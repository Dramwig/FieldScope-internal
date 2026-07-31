"""Lightweight state--geometry fusion tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fieldscope.contracts import FieldFeatures
from fieldscope.graph import fixed_grid_adjacency, normalize_adjacency


def sinusoidal_2d_position(
    grid_size: tuple[int, int], hidden_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create deterministic [P,D] two-dimensional sinusoidal positions."""

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
    return position[:, :hidden_dim].to(dtype=dtype)


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
        num_layers: int = 3,
        dropout: float = 0.0,
        mode: str = "full",
    ):
        super().__init__()
        if mode not in {
            "full",
            "state",
            "response",
            "response_local",
            "state_graph",
        }:
            raise ValueError(
                "mode must be full, state, response, response_local, or state_graph"
            )
        self.mode = mode
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.response_projection = nn.Linear(response_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.pool_query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.pool_query, std=hidden_dim**-0.5)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: FieldFeatures) -> TokenizerOutput:
        features.validate()
        if self.mode in {"state", "state_graph"}:
            nodes = self.state_projection(features.state)
        elif self.mode in {"response", "response_local"}:
            nodes = self.response_projection(features.response)
        else:
            nodes = self.state_projection(features.state) + self.response_projection(
                features.response
            )
        position = sinusoidal_2d_position(
            features.grid_size, nodes.shape[-1], nodes.device, nodes.dtype
        )
        nodes = nodes + position.unsqueeze(0)
        if self.mode in {"state", "response_local"}:
            adjacency = fixed_grid_adjacency(
                features.grid_size,
                nodes.shape[0],
                nodes.device,
                nodes.dtype,
            )
        else:
            adjacency = features.adjacency.to(dtype=nodes.dtype)
        adjacency = normalize_adjacency(adjacency)
        for layer in self.layers:
            nodes = layer(nodes, adjacency)
        nodes = self.output_norm(nodes)
        attention = torch.softmax(
            torch.einsum("bpd,d->bp", nodes, self.pool_query.to(dtype=nodes.dtype)),
            dim=-1,
        )
        global_token = torch.einsum("bp,bpd->bd", attention, nodes)
        return TokenizerOutput(dense=nodes, global_token=global_token)
