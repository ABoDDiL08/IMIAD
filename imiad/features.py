"""Frozen Timm feature stages and shared distance helpers."""

from __future__ import annotations

import timm
import torch
from torch import nn


class FeatureEncoder(nn.Module):
    """Expose ResNet feature stages by their familiar layer1..layer4 names."""

    def __init__(self, backbone: str, layers: list[str], pretrained: bool) -> None:
        super().__init__()
        if not layers or any(layer not in {"layer1", "layer2", "layer3", "layer4"} for layer in layers):
            raise ValueError("This implementation supports ResNet layer1..layer4")
        indices = tuple(int(layer[-1]) for layer in layers)
        self.layers = list(layers)
        self.network = timm.create_model(backbone, pretrained=pretrained,
                                         features_only=True, out_indices=indices)
        self.channels = self.network.feature_info.channels()
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        return dict(zip(self.layers, self.network(images)))


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    squared = x.square().sum(-1, keepdim=True) - 2 * (x @ y.transpose(-2, -1))
    squared = squared + y.square().sum(-1).unsqueeze(-2)
    return squared.clamp_min_(0).sqrt_()
