"""Lightweight state--geometry fusion tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fieldscope.contracts import FieldFeatures
from fieldscope.graph import (
    fixed_gaussian_sketch,
    fixed_identity_adjacency,
    fixed_normalized_grid_adjacency,
    normalize_adjacency,
)

_POSITIONS: dict[
    tuple[tuple[int, int], int, str, int | None, torch.dtype],
    torch.Tensor,
] = {}


def sinusoidal_2d_position(
    grid_size: tuple[int, int], hidden_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create deterministic [P,D] two-dimensional sinusoidal positions."""

    key = (grid_size, hidden_dim, device.type, device.index, dtype)
    cached = _POSITIONS.get(key)
    if cached is not None:
        return cached
    height, width = grid_size
    quarter = max(1, hidden_dim // 4)
    frequencies = torch.exp(
        -torch.arange(quarter, device=device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=device)) / max(1, quarter - 1))
    )
    yy = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequencies[None]
    xx = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequencies[None]
    y_encoding = torch.cat([yy.sin(), yy.cos()], dim=-1)
    x_encoding = torch.cat([xx.sin(), xx.cos()], dim=-1)
    position = torch.cat(
        [
            y_encoding[:, None, :].expand(-1, width, -1),
            x_encoding[None, :, :].expand(height, -1, -1),
        ],
        dim=-1,
    ).reshape(height * width, -1)
    if position.shape[-1] < hidden_dim:
        position = torch.nn.functional.pad(position, (0, hidden_dim - position.shape[-1]))
    position = position[:, :hidden_dim].to(dtype=dtype)
    _POSITIONS[key] = position
    return position


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregate = torch.bmm(adjacency, nodes)
        nodes = self.norm1(nodes + self.dropout(self.message(aggregate)))
        return self.norm2(nodes + self.dropout(self.mlp(nodes)))


@dataclass
class TokenizerOutput:
    dense: torch.Tensor
    global_token: torch.Tensor


class FieldTokenizer(nn.Module):
    """Fuse field state and response signatures with sparse graph propagation."""

    def __init__(
        self,
        state_dim: int,
        response_dim: int,
        hidden_dim: int = 128,
        input_dim: int = 768,
        num_layers: int = 3,
        dropout: float = 0.0,
        mode: str = "full",
    ):
        super().__init__()
        if mode not in {
            "full",
            "full_local",
            "full_nograph",
            "state",
            "state_nograph",
            "response",
            "response_local",
            "response_nograph",
            "state_graph",
        }:
            raise ValueError("unsupported tokenizer mode")
        self.mode = mode
        self.state_dim = state_dim
        self.response_dim = response_dim
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.pool_query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.pool_query, std=hidden_dim**-0.5)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: FieldFeatures,
        *,
        validate_features: bool = True,
    ) -> TokenizerOutput:
        if validate_features:
            features.validate()
        if self.mode in {"state", "state_graph", "state_nograph"}:
            inputs = fixed_gaussian_sketch(
                features.state,
                self.input_dim,
                seed=314159,
            )
        elif self.mode in {"response", "response_local", "response_nograph"}:
            inputs = fixed_gaussian_sketch(
                features.response,
                self.input_dim,
                seed=161803,
            )
        else:
            state = fixed_gaussian_sketch(
                features.state,
                self.input_dim,
                seed=314159,
            )
            response = fixed_gaussian_sketch(
                features.response,
                self.input_dim,
                seed=161803,
            )
            inputs = (state + response) * 2**-0.5
        nodes = self.input_projection(inputs)
        position = sinusoidal_2d_position(
            features.grid_size, nodes.shape[-1], nodes.device, nodes.dtype
        )
        nodes = nodes + position.unsqueeze(0)
        if self.mode in {"state_nograph", "response_nograph", "full_nograph"}:
            adjacency = fixed_identity_adjacency(
                nodes.shape[1],
                nodes.shape[0],
                nodes.device,
                nodes.dtype,
            )
        elif self.mode in {"state", "response_local", "full_local"}:
            adjacency = fixed_normalized_grid_adjacency(
                features.grid_size,
                nodes.shape[0],
                nodes.device,
                nodes.dtype,
            )
        else:
            adjacency = normalize_adjacency(features.adjacency.to(dtype=nodes.dtype))
        for layer in self.layers:
            nodes = layer(nodes, adjacency)
        nodes = self.output_norm(nodes)
        attention = torch.softmax(
            torch.einsum("bpd,d->bp", nodes, self.pool_query.to(dtype=nodes.dtype)),
            dim=-1,
        )
        global_token = torch.einsum("bp,bpd->bd", attention, nodes)
        return TokenizerOutput(dense=nodes, global_token=global_token)
