"""Batch and baseline operations over FieldFeatures."""

from __future__ import annotations

import torch

from fieldscope.contracts import FieldFeatures


def slice_features(features: FieldFeatures, index: int) -> FieldFeatures:
    sliced = FieldFeatures(
        state=features.state[index : index + 1],
        response=features.response[index : index + 1],
        affinity=features.affinity[index : index + 1],
        adjacency=features.adjacency[index : index + 1],
        grid_size=features.grid_size,
        baselines={
            name: tensor[index : index + 1] for name, tensor in features.baselines.items()
        },
        metadata=dict(features.metadata),
    )
    sliced.validate()
    return sliced


def stack_features(items: list[FieldFeatures]) -> FieldFeatures:
    if not items:
        raise ValueError("Cannot stack an empty feature list")
    grid_size = items[0].grid_size
    baseline_names = set(items[0].baselines)
    for item in items:
        item.validate()
        if item.grid_size != grid_size or set(item.baselines) != baseline_names:
            raise ValueError("Feature batches have incompatible contracts")
    stacked = FieldFeatures(
        state=torch.cat([item.state for item in items], dim=0),
        response=torch.cat([item.response for item in items], dim=0),
        affinity=torch.cat([item.affinity for item in items], dim=0),
        adjacency=torch.cat([item.adjacency for item in items], dim=0),
        grid_size=grid_size,
        baselines={
            name: torch.cat([item.baselines[name] for item in items], dim=0)
            for name in baseline_names
        },
        metadata=dict(items[0].metadata),
    )
    stacked.validate()
    return stacked


def select_representation(
    features: FieldFeatures, representation: str
) -> tuple[FieldFeatures, str]:
    """Select a matched experiment representation and tokenizer mode."""

    if representation in {"full", "state", "response"}:
        return features, representation
    if representation not in features.baselines:
        available = ", ".join(sorted(features.baselines))
        raise ValueError(f"Unknown representation {representation!r}; baselines: {available}")
    selected = features.baselines[representation]
    baseline_features = FieldFeatures(
        state=selected,
        response=features.response,
        affinity=features.affinity,
        adjacency=features.adjacency,
        grid_size=features.grid_size,
        baselines=features.baselines,
        metadata={**features.metadata, "selected_representation": representation},
    )
    baseline_features.validate()
    return baseline_features, "state"

