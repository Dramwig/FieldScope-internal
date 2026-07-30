"""Trainable FieldScope readout over frozen cached field features."""

from __future__ import annotations

import torch
from torch import nn

from fieldscope.config import TokenizerConfig
from fieldscope.contracts import FieldFeatures
from fieldscope.heads import MultiTaskHeads
from fieldscope.tokenizer import FieldTokenizer


class FieldScopeModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        response_dim: int,
        config: TokenizerConfig,
        *,
        mode: str = "full",
        classification_variant: str = "linear",
    ):
        super().__init__()
        self.tokenizer = FieldTokenizer(
            state_dim=state_dim,
            response_dim=response_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            mode=mode,
        )
        self.heads = MultiTaskHeads(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            segmentation_classes=config.segmentation_classes,
            classification_variant=classification_variant,
        )

    def forward(
        self,
        features: FieldFeatures,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(features)
        return self.heads(tokens, features.grid_size, output_size)

