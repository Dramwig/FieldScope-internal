"""Unsupervised graph diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from fieldscope.contracts import FieldFeatures
from fieldscope.graph import boundary_strength, spectral_binary_partition


def graph_diagnostics(features: FieldFeatures) -> dict[str, Any]:
    features.validate()
    adjacency = features.adjacency
    patches = adjacency.shape[-1]
    off_diagonal = ~torch.eye(patches, device=adjacency.device, dtype=torch.bool)
    positive_edges = (adjacency[:, off_diagonal] > 0).float().mean()
    partitions = spectral_binary_partition(adjacency)
    boundaries = boundary_strength(features.affinity, features.grid_size)
    return {
        "batch_size": features.state.shape[0],
        "patches": patches,
        "grid_size": list(features.grid_size),
        "positive_edge_density": float(positive_edges.item()),
        "affinity_mean": float(features.affinity.mean().item()),
        "affinity_std": float(features.affinity.float().std().item()),
        "boundary_strength_mean": float(boundaries.mean().item()),
        "partition_balance": [
            float(partition.float().mean().item()) for partition in partitions
        ],
    }

