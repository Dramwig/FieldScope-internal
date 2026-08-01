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
        baselines={name: tensor[index : index + 1] for name, tensor in features.baselines.items()},
        graphs={name: tensor[index : index + 1] for name, tensor in features.graphs.items()},
        metadata=dict(features.metadata),
    )
    # The source shard is fully validated by ``load_features``. Slicing is a
    # view-only operation, so only the inexpensive structural invariant needs
    # to be rechecked here.
    sliced.validate_structure()
    return sliced


def stack_features(items: list[FieldFeatures]) -> FieldFeatures:
    if not items:
        raise ValueError("Cannot stack an empty feature list")
    grid_size = items[0].grid_size
    baseline_names = set(items[0].baselines)
    graph_names = set(items[0].graphs)
    for item in items:
        item.validate_structure()
        if (
            item.grid_size != grid_size
            or set(item.baselines) != baseline_names
            or set(item.graphs) != graph_names
        ):
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
        graphs={
            name: torch.cat([item.graphs[name] for item in items], dim=0) for name in graph_names
        },
        metadata=dict(items[0].metadata),
    )
    stacked.validate_structure()
    return stacked


def select_representation(
    features: FieldFeatures, representation: str
) -> tuple[FieldFeatures, str]:
    """Select a matched experiment representation and tokenizer mode."""

    def compact(
        *,
        state: torch.Tensor,
        response: torch.Tensor,
        affinity: torch.Tensor,
        adjacency: torch.Tensor,
        mode: str,
    ) -> tuple[FieldFeatures, str]:
        selected = FieldFeatures(
            state=state,
            response=response,
            affinity=affinity,
            adjacency=adjacency,
            grid_size=features.grid_size,
            baselines={},
            graphs={},
            metadata={
                **features.metadata,
                "selected_representation": representation,
                "tokenizer_mode": mode,
            },
        )
        selected.validate_structure()
        return selected, mode

    if representation in {
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
        return compact(
            state=features.state,
            response=features.response,
            affinity=features.affinity,
            adjacency=features.adjacency,
            mode=representation,
        )
    if representation == "random_feature_local":
        return compact(
            state=features.state,
            response=features.response,
            affinity=features.affinity,
            adjacency=features.adjacency,
            mode="state",
        )
    if representation == "response_shuffled":
        return compact(
            state=features.state,
            response=features.response,
            affinity=features.affinity,
            adjacency=features.adjacency,
            mode="response",
        )
    if representation == "full_shuffled":
        return compact(
            state=features.state,
            response=features.response,
            affinity=features.affinity,
            adjacency=features.adjacency,
            mode="full",
        )
    if representation in {"dit_hidden_local", "dit_hidden_attention"}:
        if "dit_hidden" not in features.baselines:
            raise ValueError("DiT hidden features are absent from this cache")
        use_attention = representation == "dit_hidden_attention"
        if use_attention and "dit_attention_adjacency" not in features.graphs:
            raise ValueError("DiT attention graph is absent from this cache")
        selected = features.baselines["dit_hidden"]
        return compact(
            state=selected,
            response=features.response,
            affinity=(
                features.graphs.get(
                    "dit_attention",
                    features.graphs["dit_attention_adjacency"],
                )
                if use_attention
                else features.affinity
            ),
            adjacency=(
                features.graphs["dit_attention_adjacency"] if use_attention else features.adjacency
            ),
            mode="state_graph" if use_attention else "state",
        )
    if representation not in features.baselines:
        available = ", ".join(sorted(features.baselines))
        raise ValueError(f"Unknown representation {representation!r}; baselines: {available}")
    selected = features.baselines[representation]
    return compact(
        state=selected,
        response=features.response,
        affinity=features.affinity,
        adjacency=features.adjacency,
        mode="state",
    )
