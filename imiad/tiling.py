"""Grid tiling with overlap averaging and feature-resolution reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Grid:
    batch: int
    channels: int
    image_h: int
    image_w: int
    padded_h: int
    padded_w: int
    rows: int
    cols: int


class GridTiler:
    """Tiles B,C,H,W tensors; `untile` can merge lower-resolution feature maps.

    The same tile grid is reused only until the next `tile` call. This matches
    the synchronous forward path used by the PatchCore and CFlow runners.
    """

    def __init__(self, tile_size: tuple[int, int], stride: tuple[int, int],
                 mode: str = "padding", remove_border_count: int = 0) -> None:
        self.tile_size = tuple(tile_size)
        self.stride = tuple(stride)
        self.mode = mode
        self.remove_border_count = remove_border_count
        if mode not in {"padding", "interpolation"}:
            raise ValueError("Tiling mode must be padding or interpolation")
        if any(s < 1 or s > t for s, t in zip(stride, tile_size)):
            raise ValueError("Tiling stride must be in [1, tile_size]")
        self.grid: Grid | None = None

    @property
    def num_tiles(self) -> int:
        if self.grid is None:
            raise RuntimeError("Call tile before num_tiles")
        return self.grid.rows * self.grid.cols

    def tile(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("Expected B,C,H,W image tensor")
        b, c, h, w = image.shape
        th, tw = self.tile_size
        sh, sw = self.stride
        rows = max(1, ceil(max(0, h - th) / sh) + 1)
        cols = max(1, ceil(max(0, w - tw) / sw) + 1)
        ph, pw = th + (rows - 1) * sh, tw + (cols - 1) * sw
        if self.mode == "padding":
            image = F.pad(image, (0, pw - w, 0, ph - h))
        elif (ph, pw) != (h, w):
            image = F.interpolate(image, size=(ph, pw), mode="bilinear", align_corners=False)
        unfolded = F.unfold(image, kernel_size=(th, tw), stride=(sh, sw))
        self.grid = Grid(b, c, h, w, ph, pw, rows, cols)
        return unfolded.transpose(1, 2).reshape(b * rows * cols, c, th, tw)

    def untile(self, tiles: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        if grid is None:
            raise RuntimeError("Call tile before untile")
        if tiles.ndim == 3:
            tiles = tiles.unsqueeze(1)
        if tiles.ndim != 4 or tiles.shape[0] != grid.batch * grid.rows * grid.cols:
            raise ValueError("Tile count differs from most recent tile grid")
        _, channels, fh, fw = tiles.shape
        th, tw = self.tile_size
        sh, sw = self.stride
        # Fixed-size tile CNN output has a proportional spatial scale. The input
        # stride must land on a feature cell to avoid a drifting feature grid.
        if (sh * fh) % th or (sw * fw) % tw:
            raise ValueError(
                "Tile stride is not aligned with feature resolution; choose a "
                "stride divisible by the backbone feature reduction."
            )
        fsh, fsw = sh * fh // th, sw * fw // tw
        out_h = fh + (grid.rows - 1) * fsh
        out_w = fw + (grid.cols - 1) * fsw
        n = grid.rows * grid.cols
        flat = tiles.reshape(grid.batch, n, channels * fh * fw).transpose(1, 2)
        merged = F.fold(flat, output_size=(out_h, out_w),
                        kernel_size=(fh, fw), stride=(fsh, fsw))
        weight = tiles.new_ones((grid.batch, grid.rows, grid.cols, 1, fh, fw))
        border = self.remove_border_count
        if border:
            # border is expressed in feature cells, as in Anomalib's `untile`.
            if 2 * border >= min(fh, fw):
                raise ValueError("remove_border_count removes the entire feature tile")
            # Preserve image-edge cells; discard only internal tile borders.
            weight[:, 1:, :, :, :border, :] = 0
            weight[:, :-1, :, :, -border:, :] = 0
            weight[:, :, 1:, :, :, :border] = 0
            weight[:, :, :-1, :, :, -border:] = 0
            flat_weight = weight.reshape(grid.batch * n, 1, fh, fw)
            merged = F.fold((tiles * flat_weight).reshape(grid.batch, n, channels * fh * fw).transpose(1, 2),
                            output_size=(out_h, out_w), kernel_size=(fh, fw), stride=(fsh, fsw))
        count = F.fold(weight.reshape(grid.batch, n, fh * fw).transpose(1, 2),
                       output_size=(out_h, out_w), kernel_size=(fh, fw), stride=(fsh, fsw))
        crop_h = ceil(grid.image_h * fh / th)
        crop_w = ceil(grid.image_w * fw / tw)
        if torch.any(count == 0):
            raise ValueError("Tiling leaves uncovered pixels; reduce stride or border removal")
        merged = merged / count
        if self.mode == "interpolation" and (grid.padded_h, grid.padded_w) != (grid.image_h, grid.image_w):
            merged = F.interpolate(merged, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
        else:
            merged = merged[:, :, :crop_h, :crop_w]
        return merged
