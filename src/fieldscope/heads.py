"""Matched lightweight task heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fieldscope.tokenizer import TokenizerOutput


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, variant: str = "linear"):
        super().__init__()
        if variant == "linear":
            self.network = nn.Linear(hidden_dim, num_classes)
        elif variant == "mlp":
            self.network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError("classification variant must be linear or mlp")

    def forward(self, global_token: torch.Tensor) -> torch.Tensor:
        return self.network(global_token)


class DenseHead(nn.Module):
    """Small two-block decoder shared by segmentation/depth/normal heads."""

    def __init__(self, hidden_dim: int, out_channels: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )

    def forward(
        self,
        dense_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, patches, channels = dense_tokens.shape
        if patches != grid_size[0] * grid_size[1]:
            raise ValueError("grid_size does not match dense token count")
        feature_map = dense_tokens.transpose(1, 2).reshape(batch, channels, *grid_size)
        prediction = self.decoder(feature_map)
        return F.interpolate(prediction, size=output_size, mode="bilinear", align_corners=False)


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        segmentation_classes: int,
        classification_variant: str = "linear",
    ):
        super().__init__()
        self.classification = ClassificationHead(
            hidden_dim, num_classes, classification_variant
        )
        self.segmentation = DenseHead(hidden_dim, segmentation_classes)
        self.depth = DenseHead(hidden_dim, 1)
        self.normals = DenseHead(hidden_dim, 3)

    def forward(
        self,
        tokens: TokenizerOutput,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        normals = self.normals(tokens.dense, grid_size, output_size)
        normals = F.normalize(normals, dim=1, eps=1e-6)
        depth = F.softplus(
            self.depth(tokens.dense, grid_size, output_size)
        ).clamp_min(1e-6)
        return {
            "classification": self.classification(tokens.global_token),
            "segmentation": self.segmentation(tokens.dense, grid_size, output_size),
            "depth": depth,
            "normals": normals,
        }
