"""PatchCore with image input tiling and one global memory bank."""

from __future__ import annotations

from math import ceil, log
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .features import FeatureEncoder, euclidean_dist
from .tiling import GridTiler


def kcenter_coreset(embeddings: torch.Tensor, ratio: float, eps: float = 0.9) -> torch.Tensor:
    """Sparse random projection followed by farthest-first k-center sampling."""
    n, dimension = embeddings.shape
    count = max(1, int(n * ratio))
    if count >= n:
        return embeddings
    target = min(dimension, ceil(4 * log(max(n, 2)) / (eps**2 / 2 - eps**3 / 3)))
    density = min(1.0, 1 / dimension**0.5)
    random_values = torch.rand(dimension, target, device=embeddings.device)
    signs = torch.randint(0, 2, (dimension, target), device=embeddings.device,
                          dtype=torch.int8).to(embeddings.dtype).mul_(2).sub_(1)
    projection = torch.where(random_values < density,
                             signs / (density * target) ** 0.5, 0.0)
    features = embeddings @ projection
    first = int(torch.randint(n, (1,), device=embeddings.device))
    selected = [first]
    minimum = torch.full((n,), float("inf"), device=embeddings.device)
    for _ in range(count - 1):
        distance = torch.linalg.vector_norm(features - features[selected[-1]], dim=1)
        minimum = torch.minimum(minimum, distance)
        minimum[selected] = -1
        selected.append(int(torch.argmax(minimum)))
    return embeddings[selected]


class GaussianBlur(nn.Module):
    def __init__(self, sigma: float = 4.0) -> None:
        super().__init__()
        radius = int(4 * sigma + 0.5)
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (axis / sigma).square())
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", torch.outer(kernel, kernel)[None, None])
        self.radius = radius

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mode = "reflect" if min(image.shape[-2:]) > self.radius else "replicate"
        return F.conv2d(F.pad(image, (self.radius,) * 4, mode=mode), self.kernel)


class PatchCoreModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layers = cfg["model"]["layers"]
        self.encoder = FeatureEncoder(cfg["model"]["backbone"], self.layers,
                                      cfg["model"]["pretrained"])
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.blur = GaussianBlur(4)
        self.num_neighbors = cfg["patchcore"]["num_neighbors"]
        self.register_buffer("memory_bank", torch.empty(0))
        tile_cfg = cfg["input_tiling"]
        self.tiler = (GridTiler(tile_cfg["tile_size"], tile_cfg["stride"],
                                tile_cfg["mode"], tile_cfg["remove_border_count"])
                      if tile_cfg["enabled"] else None)

    @torch.no_grad()
    def embedding(self, images: torch.Tensor) -> torch.Tensor:
        if self.tiler:
            images = self.tiler.tile(images)
        features = self.encoder(images)
        pooled = [self.pool(features[layer]) for layer in self.layers]
        base = pooled[0]
        aligned = [base] + [F.interpolate(other, size=base.shape[-2:],
                                          mode="bilinear", align_corners=False)
                            for other in pooled[1:]]
        embedding = torch.cat(aligned, dim=1)
        return self.tiler.untile(embedding) if self.tiler else embedding

    def _nearest(self, query: torch.Tensor, bank: torch.Tensor,
                 neighbors: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = [], []
        for chunk in query.split(1024):
            distances = euclidean_dist(chunk, bank)
            if neighbors == 1:
                value, index = distances.min(dim=1)
            else:
                value, index = distances.topk(neighbors, largest=False, dim=1)
            values.append(value)
            indices.append(index)
        return torch.cat(values), torch.cat(indices)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank.numel() == 0:
            raise ValueError("PatchCore memory bank has not been fitted")
        batch, _, image_h, image_w = images.shape
        embedding = self.embedding(images)
        _, channels, height, width = embedding.shape
        patches = embedding.permute(0, 2, 3, 1).reshape(-1, channels)
        distances, locations = self._nearest(patches, self.memory_bank)
        patch_scores = distances.reshape(batch, -1)
        max_score, max_index = patch_scores.max(dim=1)
        if self.num_neighbors == 1:
            image_score = max_score
        else:
            batch_index = torch.arange(batch, device=images.device)
            nearest_bank = self.memory_bank[locations.reshape(batch, -1)[batch_index, max_index]]
            _, supports = self._nearest(nearest_bank, self.memory_bank,
                                         min(self.num_neighbors, len(self.memory_bank)))
            if supports.ndim == 1:
                supports = supports.unsqueeze(1)
            query = patches.reshape(batch, -1, channels)[batch_index, max_index]
            support_distances = euclidean_dist(query.unsqueeze(1), self.memory_bank[supports]).squeeze(1)
            image_score = (1 - F.softmax(support_distances, dim=1)[:, 0]) * max_score
        patch_map = distances.reshape(batch, 1, height, width)
        anomaly_map = self.blur(F.interpolate(patch_map, size=(image_h, image_w)))
        return image_score, anomaly_map.squeeze(1)


class PatchCoreRunner:
    def __init__(self, cfg: dict, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.model = PatchCoreModel(cfg).to(device)

    @torch.no_grad()
    def fit(self, loader) -> None:
        rows = []
        self.model.eval()
        for batch in loader:
            embedding = self.model.embedding(batch["image"].to(self.device))
            rows.append(embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1]).cpu())
        bank = kcenter_coreset(torch.cat(rows), self.cfg["patchcore"]["coreset_ratio"],
                               self.cfg["patchcore"]["projection_eps"])
        self.model.memory_bank = bank.to(self.device)
        if len(bank) < self.model.num_neighbors:
            raise ValueError("Coreset is smaller than patchcore.num_neighbors")

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        return self.model.predict(image.to(self.device))

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)
