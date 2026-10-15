"""Conditional flow with shared global-coordinate tiling or tiled ensemble."""

from __future__ import annotations

from math import log, pi
from pathlib import Path

import torch
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock
from torch import nn
from torch.nn import functional as F

from .features import FeatureEncoder
from .tiling import Grid, GridTiler


def positional_encoding_2d(dimension: int, height: int, width: int,
                           device: torch.device) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("Condition dimension must be divisible by four")
    half = dimension // 2
    frequencies = torch.exp(torch.arange(0, half, 2, device=device) * (-log(10000) / half))
    x = torch.arange(width, device=device).unsqueeze(1) * frequencies
    y = torch.arange(height, device=device).unsqueeze(1) * frequencies
    position = torch.zeros(dimension, height, width, device=device)
    position[:half:2] = x.sin().T.unsqueeze(1).expand(-1, height, -1)
    position[1:half:2] = x.cos().T.unsqueeze(1).expand(-1, height, -1)
    position[half::2] = y.sin().T.unsqueeze(-1).expand(-1, -1, width)
    position[half + 1::2] = y.cos().T.unsqueeze(-1).expand(-1, -1, width)
    return position


def _subnet(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, 2 * input_dim), nn.ReLU(),
                         nn.Linear(2 * input_dim, output_dim))


def global_positional_encoding_2d(dimension: int, height: int, width: int,
                                  device: torch.device, indices: list[int],
                                  grid: Grid, tile_size: tuple[int, int],
                                  stride: tuple[int, int], mode: str) -> torch.Tensor:
    """Encode feature-cell centers in original-image pixels, including tile offsets.

    Resize is accounted for by mapping the feature grid directly to the raw
    tile extent. In interpolation mode the padded canvas is mapped back to the
    original image. Padding mode retains coordinates outside the image bounds.
    """
    if dimension % 4:
        raise ValueError("Condition dimension must be divisible by four")
    half = dimension // 2
    frequencies = torch.exp(torch.arange(0, half, 2, device=device) * (-log(10000) / half))
    tile_ids = torch.tensor(indices, device=device) % (grid.rows * grid.cols)
    rows, cols = tile_ids // grid.cols, tile_ids % grid.cols
    y = rows[:, None] * stride[0] + (torch.arange(height, device=device) + 0.5) * tile_size[0] / height
    x = cols[:, None] * stride[1] + (torch.arange(width, device=device) + 0.5) * tile_size[1] / width
    if mode == "interpolation":
        y = y * grid.image_h / grid.padded_h
        x = x * grid.image_w / grid.padded_w
    y = (y - 0.5).unsqueeze(-1) * frequencies
    x = (x - 0.5).unsqueeze(-1) * frequencies
    position = torch.empty(len(indices), dimension, height, width, device=device)
    position[:, :half:2] = x.sin().transpose(1, 2).unsqueeze(2)
    position[:, 1:half:2] = x.cos().transpose(1, 2).unsqueeze(2)
    position[:, half::2] = y.sin().transpose(1, 2).unsqueeze(3)
    position[:, half + 1::2] = y.cos().transpose(1, 2).unsqueeze(3)
    return position


class CFlowModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        layers = cfg["model"]["layers"]
        flow = cfg["cflow"]
        self.layers = layers
        self.condition_vector = flow["condition_vector"]
        self.fiber_batch_size = flow["fiber_batch_size"]
        self.encoder = FeatureEncoder(cfg["model"]["backbone"], layers,
                                      cfg["model"]["pretrained"])
        self.decoders = nn.ModuleList()
        for channels in self.encoder.channels:
            decoder = SequenceINN(channels)
            for _ in range(flow["coupling_blocks"]):
                decoder.append(AllInOneBlock, cond=0,
                               cond_shape=(self.condition_vector,),
                               subnet_constructor=_subnet,
                               affine_clamping=flow["clamp_alpha"],
                               global_affine_type="SOFTPLUS", permute_soft=False)
            self.decoders.append(decoder)

    def fibers(self, images: torch.Tensor, position_context: dict | None = None):
        with torch.no_grad():
            activations = self.encoder(images)
        for layer, decoder in zip(self.layers, self.decoders):
            feature = activations[layer]
            batch, channels, height, width = feature.shape
            condition = (positional_encoding_2d(self.condition_vector, height, width,
                                                images.device).unsqueeze(0)
                         if position_context is None else global_positional_encoding_2d(
                             self.condition_vector, height, width, images.device,
                             **position_context))
            condition = condition.expand(batch, -1, -1, -1)
            condition = condition.permute(0, 2, 3, 1).reshape(-1, self.condition_vector)
            vectors = feature.permute(0, 2, 3, 1).reshape(-1, channels)
            yield vectors, condition, decoder, (batch, height, width)

    @staticmethod
    def log_probability(channels: int, latent: torch.Tensor,
                        log_jacobian: torch.Tensor) -> torch.Tensor:
        return (-channels * log(2 * pi) / 2 - latent.square().sum(dim=1) / 2
                + log_jacobian) / channels

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        probabilities = []
        for vectors, condition, decoder, (batch, height, width) in self.fibers(images):
            logp = []
            for vector_chunk, condition_chunk in zip(vectors.split(self.fiber_batch_size),
                                                       condition.split(self.fiber_batch_size)):
                latent, jacobian = decoder(vector_chunk, [condition_chunk])
                logp.append(self.log_probability(vector_chunk.shape[1], latent, jacobian))
            values = torch.cat(logp)
            probabilities.append(F.interpolate(
                (values - values.max()).exp().reshape(batch, 1, height, width),
                size=images.shape[-2:], mode="bilinear", align_corners=True,
            ))
        score_map = torch.stack(probabilities).sum(dim=0)
        anomaly_map = score_map.max() - score_map
        return anomaly_map.flatten(1).amax(dim=1), anomaly_map.squeeze(1)


class CFlowRunner:
    def __init__(self, cfg: dict, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.model = CFlowModel(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.decoders.parameters(),
                                          lr=cfg["cflow"]["learning_rate"])

    def train_batch(self, images: torch.Tensor, position_context: dict | None = None) -> float:
        model = self.model
        model.encoder.eval()
        model.decoders.train()
        losses: list[float] = []
        for vectors, condition, decoder, _ in model.fibers(images.to(self.device), position_context):
            permutation = torch.randperm(len(vectors), device=self.device)
            for ids in permutation.split(model.fiber_batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                latent, jacobian = decoder(vectors[ids], [condition[ids]])
                logp = model.log_probability(vectors.shape[1], latent, jacobian)
                loss = -F.logsigmoid(logp).mean()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))
        return sum(losses) / max(1, len(losses))

    def fit(self, loader) -> list[float]:
        history = []
        for _ in range(self.cfg["cflow"]["epochs"]):
            losses = [self.train_batch(batch["image"]) for batch in loader]
            history.append(sum(losses) / len(losses))
        self.model.eval()
        return history

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.predict(image.to(self.device))

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


class TiledCFlowRunner(CFlowRunner):
    """One shared CFLOW with global coordinates and streamed native-image tiles."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        super().__init__(cfg, device)
        tile = cfg["input_tiling"]
        self.tiler = GridTiler(tile["tile_size"], tile["stride"], tile["mode"],
                               tile["remove_border_count"])
        self.tile_resize = tuple(tile["tile_resize"])
        self.tile_batch_size = cfg["cflow"]["tile_batch_size"]

    def tile_batches(self, images: torch.Tensor):
        for indices, tiles in self.tiler.iter_tiles(images.cpu(), self.tile_batch_size):
            if tiles.shape[-2:] != self.tile_resize:
                tiles = F.interpolate(tiles, size=self.tile_resize,
                                      mode="bilinear", align_corners=False)
            context = {"indices": indices, "grid": self.tiler.grid,
                       "tile_size": self.tiler.tile_size, "stride": self.tiler.stride,
                       "mode": self.tiler.mode}
            yield indices, tiles.to(self.device), context

    def fit(self, loader) -> list[float]:
        history = []
        for _ in range(self.cfg["cflow"]["epochs"]):
            total, count = 0.0, 0
            for batch in loader:
                for indices, tiles, context in self.tile_batches(batch["image"]):
                    total += self.train_batch(tiles, context) * len(indices)
                    count += len(indices)
            history.append(total / max(1, count))
        self.model.eval()
        return history

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[0] != 1:
            raise ValueError("Tiled CFLOW inference expects one image")
        self.model.eval()
        accumulator = None
        for indices, tiles, context in self.tile_batches(image):
            if accumulator is None:
                accumulator = self.tiler.map_accumulator()
            maps = []
            for vectors, condition, decoder, (batch, height, width) in self.model.fibers(tiles, context):
                logp = []
                for vectors_chunk, condition_chunk in zip(
                        vectors.split(self.model.fiber_batch_size),
                        condition.split(self.model.fiber_batch_size)):
                    latent, jacobian = decoder(vectors_chunk, [condition_chunk])
                    logp.append(self.model.log_probability(vectors_chunk.shape[1], latent, jacobian))
                # Raw negative log likelihood keeps tiles and tile batches on
                # the same scale; no per-tile or per-batch max normalization.
                maps.append(F.interpolate(-torch.cat(logp).reshape(batch, 1, height, width),
                                          size=self.tiler.tile_size, mode="bilinear",
                                          align_corners=False))
            accumulator.add(indices, torch.stack(maps).mean(dim=0).cpu())
        if accumulator is None:
            raise ValueError("No tiles generated")
        merged = accumulator.result().squeeze(1)
        return merged.flatten(1).amax(dim=1), merged


class TiledCFlowEnsemble:
    """One independent CFlow model per tile position, merged by overlap mean."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        tile_cfg = cfg["tiled_ensemble"]
        if tile_cfg["aggregation"] != "mean":
            raise ValueError("TiledEnsemble supports mean overlap aggregation")
        self.tiler = GridTiler(tile_cfg["tile_size"], tile_cfg["stride"])
        height, width = cfg["data"]["image_size"]
        self.tiler.tile(torch.zeros(1, 3, height, width))
        self.models = [CFlowRunner(cfg, device) for _ in range(self.tiler.num_tiles)]

    def fit(self, loader) -> list[list[float]]:
        histories = []
        for tile_index, runner in enumerate(self.models):
            epoch_loss = []
            for _ in range(self.cfg["cflow"]["epochs"]):
                losses = []
                for batch in loader:
                    images = batch["image"]
                    tiles = self.tiler.tile(images).reshape(
                        images.shape[0], len(self.models), images.shape[1],
                        *self.tiler.tile_size,
                    )[:, tile_index]
                    losses.append(runner.train_batch(tiles))
                epoch_loss.append(sum(losses) / len(losses))
            runner.model.eval()
            histories.append(epoch_loss)
        return histories

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[0] != 1:
            raise ValueError("TiledCFlowEnsemble predicts one image at a time")
        tiles = self.tiler.tile(image)
        maps = []
        for runner, tile in zip(self.models, tiles):
            _, anomaly_map = runner.predict(tile.unsqueeze(0))
            maps.append(anomaly_map.unsqueeze(1))
        merged = self.tiler.untile(torch.cat(maps, dim=0)).squeeze(1)
        return merged.flatten(1).amax(dim=1), merged

    def save(self, path: Path) -> None:
        torch.save([runner.model.state_dict() for runner in self.models], path)
