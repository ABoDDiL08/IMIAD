"""PatchCore with image input tiling and one global memory bank."""

from __future__ import annotations

from math import ceil, log
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F

from .features import FeatureEncoder, euclidean_dist
from .coreset_disk import select_coreset
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
        self.tile_batch_size = cfg["patchcore"]["tile_batch_size"]
        self.distance_bank_chunk_size = cfg["patchcore"]["distance_bank_chunk_size"]
        self.register_buffer("memory_bank", torch.empty(0))
        tile_cfg = cfg["input_tiling"]
        self.tile_resize = tuple(tile_cfg["tile_resize"])
        self.tiler = (GridTiler(tile_cfg["tile_size"], tile_cfg["stride"],
                                tile_cfg["mode"], tile_cfg["remove_border_count"])
                      if tile_cfg["enabled"] else None)

    @torch.no_grad()
    def embedding(self, images: torch.Tensor) -> torch.Tensor:
        if self.tiler is not None:
            raise RuntimeError("Use tile_embeddings for bounded-memory tiled PatchCore")
        features = self.encoder(images)
        pooled = [self.pool(features[layer]) for layer in self.layers]
        base = pooled[0]
        aligned = [base] + [F.interpolate(other, size=base.shape[-2:],
                                          mode="bilinear", align_corners=False)
                            for other in pooled[1:]]
        embedding = torch.cat(aligned, dim=1)
        return embedding

    @torch.no_grad()
    def tile_embeddings(self, images: torch.Tensor):
        """Run one shared encoder on bounded tile batches; return CPU embeddings."""
        if self.tiler is None:
            raise RuntimeError("Input Tiling is disabled")
        device = next(self.encoder.parameters()).device
        for indices, tiles in self.tiler.iter_tiles(images, self.tile_batch_size):
            if tiles.shape[-2:] != self.tile_resize:
                tiles = F.interpolate(tiles, size=self.tile_resize,
                                      mode="bilinear", align_corners=False)
            features = self.encoder(tiles.to(device))
            pooled = [self.pool(features[layer]) for layer in self.layers]
            base = pooled[0]
            aligned = [base] + [F.interpolate(other, size=base.shape[-2:],
                                              mode="bilinear", align_corners=False)
                                for other in pooled[1:]]
            yield indices, torch.cat(aligned, dim=1).cpu()

    def _nearest(self, query: torch.Tensor, bank: torch.Tensor,
                 neighbors: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = [], []
        for chunk in query.split(256):
            best_values = torch.full((len(chunk), neighbors), float("inf"), device=query.device)
            best_indices = torch.zeros((len(chunk), neighbors), dtype=torch.long, device=query.device)
            for start in range(0, len(bank), self.distance_bank_chunk_size):
                distances = euclidean_dist(chunk, bank[start:start + self.distance_bank_chunk_size])
                local_values, local_indices = distances.topk(min(neighbors, distances.shape[1]),
                                                              largest=False, dim=1)
                candidates = torch.cat((best_values, local_values), dim=1)
                candidate_indices = torch.cat((best_indices, local_indices + start), dim=1)
                best_values, positions = candidates.topk(neighbors, largest=False, dim=1)
                best_indices = candidate_indices.gather(1, positions)
            values.append(best_values.squeeze(1) if neighbors == 1 else best_values)
            indices.append(best_indices.squeeze(1) if neighbors == 1 else best_indices)
        return torch.cat(values), torch.cat(indices)

    @torch.no_grad()
    def predict_tiled(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score tile patches independently and join only their native-size maps."""
        if images.shape[0] != 1 or images.device.type != "cpu":
            raise ValueError("Tiled PatchCore inference expects one CPU image")
        accumulator = None
        max_score = torch.tensor(-float("inf"))
        max_query = None
        max_bank_index = None
        for indices, embeddings in self.tile_embeddings(images):
            if accumulator is None:
                accumulator = self.tiler.map_accumulator()
            count, channels, height, width = embeddings.shape
            patches = embeddings.permute(0, 2, 3, 1).reshape(-1, channels)
            distances, locations = self._nearest(patches, self.memory_bank)
            best, position = distances.max(dim=0)
            if best > max_score:
                max_score = best
                max_query = patches[position].clone()
                max_bank_index = locations[position]
            maps = distances.reshape(count, 1, height, width)
            maps = F.interpolate(maps, size=self.tiler.tile_size,
                                 mode="bilinear", align_corners=False)
            accumulator.add(indices, maps)
        if accumulator is None or max_query is None:
            raise ValueError("No tiles generated")
        if self.num_neighbors == 1:
            image_score = max_score
        else:
            _, supports = self._nearest(self.memory_bank[max_bank_index].unsqueeze(0),
                                         self.memory_bank,
                                         min(self.num_neighbors, len(self.memory_bank)))
            support_distances = euclidean_dist(max_query.unsqueeze(0),
                                                self.memory_bank[supports[0]]).squeeze(0)
            image_score = (1 - F.softmax(support_distances, dim=0)[0]) * max_score
        anomaly_map = self.blur(accumulator.result()).squeeze(1)
        return image_score.unsqueeze(0), anomaly_map

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank.numel() == 0:
            raise ValueError("PatchCore memory bank has not been fitted")
        if self.tiler is not None:
            return self.predict_tiled(images)
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
        if self.model.tiler is not None:
            self.model.blur.cpu()

    @torch.no_grad()
    def fit(self, loader) -> None:
        self.model.eval()
        if self.model.tiler is None:
            rows = []
            for batch in loader:
                embedding = self.model.embedding(batch["image"].to(self.device))
                rows.append(embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1]).cpu())
            bank = kcenter_coreset(torch.cat(rows), self.cfg["patchcore"]["coreset_ratio"],
                                   self.cfg["patchcore"]["projection_eps"])
            self.model.memory_bank = bank.to(self.device)
        else:
            with TemporaryDirectory(prefix="imiad_patchcore_", dir=self.cfg.get("_run_dir")) as temp:
                path = Path(temp) / "embeddings.bin"
                rows = []
                total, dimension = 0, None
                with path.open("wb") as stream:
                    for batch in loader:
                        for _, embedding in self.model.tile_embeddings(batch["image"]):
                            patches = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1]).contiguous()
                            dimension = patches.shape[1]
                            total += len(patches)
                            if self.cfg["patchcore"]["embedding_storage"] == "disk":
                                patches.numpy().tofile(stream)
                            else:
                                rows.append(patches)
                if self.cfg["patchcore"]["embedding_storage"] == "disk":
                    bank = select_coreset(path, total, dimension,
                                          self.cfg["patchcore"]["coreset_ratio"],
                                          self.cfg["patchcore"]["projection_eps"])
                else:
                    bank = kcenter_coreset(torch.cat(rows), self.cfg["patchcore"]["coreset_ratio"],
                                           self.cfg["patchcore"]["projection_eps"])
            self.model.memory_bank = bank.cpu()
        if len(bank) < self.model.num_neighbors:
            raise ValueError("Coreset is smaller than patchcore.num_neighbors")

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        return self.model.predict(image.cpu() if self.model.tiler else image.to(self.device))

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)
