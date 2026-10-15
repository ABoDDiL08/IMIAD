"""Disk-backed exact farthest-first coreset selection for PatchCore embeddings."""

from __future__ import annotations

from math import ceil, log
from pathlib import Path

import numpy as np
import torch


def select_coreset(path: Path, rows: int, dimension: int, ratio: float,
                   eps: float, chunk_size: int = 4096, progress=None) -> torch.Tensor:
    """Keep embeddings and projected distances on disk until the final bank is selected."""
    count = max(1, int(rows * ratio))
    embeddings = np.memmap(path, dtype=np.float32, mode="r", shape=(rows, dimension))
    if count >= rows:
        if progress is not None:
            progress("train_coreset_selection", centers_done=rows, centers_total=rows)
        return torch.from_numpy(np.asarray(embeddings).copy())
    if progress is not None:
        progress("train_coreset_projection", candidate_patches=rows,
                 centers_total=count)
    target = min(dimension, ceil(4 * log(max(rows, 2)) / (eps**2 / 2 - eps**3 / 3)))
    density = min(1.0, 1 / dimension**0.5)
    random_values = torch.rand(dimension, target)
    signs = torch.randint(0, 2, (dimension, target), dtype=torch.int8).float().mul_(2).sub_(1)
    projection = torch.where(random_values < density,
                             signs / (density * target) ** 0.5, 0.0)
    projected_path = path.with_name("projected.bin")
    projected = np.memmap(projected_path, dtype=np.float32, mode="w+", shape=(rows, target))
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        projected[start:end] = (torch.from_numpy(np.asarray(embeddings[start:end]).copy())
                                @ projection).numpy()
    projected.flush()
    selected = [int(torch.randint(rows, (1,)))]
    minimum = np.full(rows, np.inf, dtype=np.float32)
    if progress is not None:
        progress("train_coreset_selection", centers_done=1, centers_total=count)
    report_every = max(1, count // 20)
    for _ in range(count - 1):
        center = np.asarray(projected[selected[-1]]).copy()
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            delta = np.asarray(projected[start:end]) - center
            distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            np.minimum(minimum[start:end], distances, out=minimum[start:end])
        minimum[selected] = -1
        selected.append(int(np.argmax(minimum)))
        if progress is not None and (len(selected) % report_every == 0 or len(selected) == count):
            progress("train_coreset_selection", centers_done=len(selected), centers_total=count)
    bank = torch.from_numpy(np.asarray(embeddings[selected]).copy())
    del projected
    return bank
