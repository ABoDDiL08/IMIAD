"""Select the configured model runner."""

from __future__ import annotations

import torch

from .cflow import CFlowRunner, TiledCFlowEnsemble, TiledCFlowRunner
from .patchcore import PatchCoreRunner


def build_runner(cfg: dict, device: torch.device):
    if cfg["model"]["name"] == "patchcore":
        return PatchCoreRunner(cfg, device)
    if cfg["tiled_ensemble"]["enabled"]:
        return TiledCFlowEnsemble(cfg, device)
    if cfg["input_tiling"]["enabled"]:
        return TiledCFlowRunner(cfg, device)
    return CFlowRunner(cfg, device)
