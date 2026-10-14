"""Grid tiling with overlap averaging and feature-resolution reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

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

    def iter_tiles(self, image: torch.Tensor, batch_size: int):
        """Yield small CPU tile batches without materializing the whole unfolded grid."""
        if image.ndim != 4 or image.device.type != "cpu" or batch_size < 1:
            raise ValueError("iter_tiles expects a B,C,H,W CPU image and positive batch_size")
        b, c, h, w = image.shape
        th, tw = self.tile_size
        sh, sw = self.stride
        rows = max(1, ceil(max(0, h - th) / sh) + 1)
        cols = max(1, ceil(max(0, w - tw) / sw) + 1)
        ph, pw = th + (rows - 1) * sh, tw + (cols - 1) * sw
        self.grid = Grid(b, c, h, w, ph, pw, rows, cols)
        if self.mode == "interpolation" and (ph, pw) != (h, w):
            image = F.interpolate(image, size=(ph, pw), mode="bilinear", align_corners=False)
        chunk, indices = [], []
        for batch_idx in range(b):
            for row in range(rows):
                for col in range(cols):
                    top, left = row * sh, col * sw
                    tile = image[batch_idx, :, top:min(top + th, image.shape[-2]),
                                 left:min(left + tw, image.shape[-1])]
                    if tile.shape[-2:] != (th, tw):
                        tile = F.pad(tile, (0, tw - tile.shape[-1], 0, th - tile.shape[-2]))
                    chunk.append(tile)
                    indices.append(batch_idx * rows * cols + row * cols + col)
                    if len(chunk) == batch_size:
                        yield indices, torch.stack(chunk)
                        chunk, indices = [], []
        if chunk:
            yield indices, torch.stack(chunk)

    def map_accumulator(self) -> "TileMapAccumulator":
        if self.grid is None:
            raise RuntimeError("Call iter_tiles before map_accumulator")
        return TileMapAccumulator(self.grid, self.tile_size, self.stride,
                                  self.mode, self.remove_border_count)

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
        n = grid.rows * grid.cols
        border = self.remove_border_count
        if border and 2 * border >= min(fh, fw):
            raise ValueError("remove_border_count removes the entire feature tile")
        if (sh * fh) % th or (sw * fw) % tw:
            # A tile resize may put the original stride between feature cells.
            # Place each tile using its raw-image coordinates, resampling by at
            # most one feature cell at a boundary, then average overlaps.
            out_h, out_w = ceil(grid.padded_h * fh / th), ceil(grid.padded_w * fw / tw)
            merged = tiles.new_zeros((grid.batch, channels, out_h, out_w))
            count = tiles.new_zeros((grid.batch, 1, out_h, out_w))
            for batch_idx in range(grid.batch):
                for row in range(grid.rows):
                    for col in range(grid.cols):
                        idx = batch_idx * n + row * grid.cols + col
                        top = floor(row * sh * fh / th)
                        left = floor(col * sw * fw / tw)
                        bottom = ceil((row * sh + th) * fh / th)
                        right = ceil((col * sw + tw) * fw / tw)
                        size = (bottom - top, right - left)
                        weight = tiles.new_ones((1, 1, fh, fw))
                        if border:
                            if row:
                                weight[:, :, :border, :] = 0
                            if row < grid.rows - 1:
                                weight[:, :, -border:, :] = 0
                            if col:
                                weight[:, :, :, :border] = 0
                            if col < grid.cols - 1:
                                weight[:, :, :, -border:] = 0
                        scaled_tile = F.interpolate(tiles[idx:idx + 1], size=size,
                                                    mode="bilinear", align_corners=False)
                        scaled_weight = F.interpolate(weight, size=size,
                                                      mode="bilinear", align_corners=False)
                        merged[batch_idx:batch_idx + 1, :, top:bottom, left:right] += scaled_tile * scaled_weight
                        count[batch_idx:batch_idx + 1, :, top:bottom, left:right] += scaled_weight
        else:
            fsh, fsw = sh * fh // th, sw * fw // tw
            out_h = fh + (grid.rows - 1) * fsh
            out_w = fw + (grid.cols - 1) * fsw
            flat = tiles.reshape(grid.batch, n, channels * fh * fw).transpose(1, 2)
            merged = F.fold(flat, output_size=(out_h, out_w),
                            kernel_size=(fh, fw), stride=(fsh, fsw))
            weight = tiles.new_ones((grid.batch, grid.rows, grid.cols, 1, fh, fw))
            if border:
                # Keep raw image edges; suppress only internal tile borders.
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


class TileMapAccumulator:
    """Accumulate native-size anomaly maps one tile batch at a time on CPU."""

    def __init__(self, grid: Grid, tile_size: tuple[int, int], stride: tuple[int, int],
                 mode: str, border: int) -> None:
        self.grid, self.tile_size, self.stride, self.mode, self.border = (
            grid, tile_size, stride, mode, border)
        self.total = torch.zeros((grid.batch, 1, grid.padded_h, grid.padded_w))
        self.count = torch.zeros_like(self.total)

    def add(self, indices: list[int], maps: torch.Tensor) -> None:
        if maps.device.type != "cpu" or maps.shape != (len(indices), 1, *self.tile_size):
            raise ValueError("Expected CPU maps with one native-size map per tile index")
        th, tw = self.tile_size
        sh, sw = self.stride
        n = self.grid.rows * self.grid.cols
        if self.border and 2 * self.border >= min(th, tw):
            raise ValueError("remove_border_count removes the entire map tile")
        for local_idx, index in enumerate(indices):
            batch_idx, tile_idx = divmod(index, n)
            row, col = divmod(tile_idx, self.grid.cols)
            top, left = row * sh, col * sw
            weight = torch.ones((1, 1, th, tw))
            if self.border:
                if row:
                    weight[:, :, :self.border, :] = 0
                if row < self.grid.rows - 1:
                    weight[:, :, -self.border:, :] = 0
                if col:
                    weight[:, :, :, :self.border] = 0
                if col < self.grid.cols - 1:
                    weight[:, :, :, -self.border:] = 0
            self.total[batch_idx:batch_idx + 1, :, top:top + th, left:left + tw] += (
                maps[local_idx:local_idx + 1] * weight)
            self.count[batch_idx:batch_idx + 1, :, top:top + th, left:left + tw] += weight

    def result(self) -> torch.Tensor:
        if torch.any(self.count == 0):
            raise ValueError("Tiling leaves uncovered pixels; reduce stride or border removal")
        merged = self.total / self.count
        if self.mode == "interpolation" and (
            (self.grid.padded_h, self.grid.padded_w) != (self.grid.image_h, self.grid.image_w)
        ):
            return F.interpolate(merged, size=(self.grid.image_h, self.grid.image_w),
                                 mode="bilinear", align_corners=False)
        return merged[:, :, :self.grid.image_h, :self.grid.image_w]
