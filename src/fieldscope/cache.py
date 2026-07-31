"""Portable feature-cache serialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from fieldscope.contracts import FieldFeatures
from fieldscope.graph import sparsify_affinity

FORMAT_VERSION = 4


def feature_fingerprint(features: FieldFeatures, *, format_version: int = FORMAT_VERSION) -> str:
    payload = {
        "grid_size": features.grid_size,
        "state_shape": tuple(features.state.shape),
        "response_shape": tuple(features.response.shape),
        "metadata": features.metadata,
    }
    if format_version >= 2:
        payload["baseline_shapes"] = {
            name: tuple(value.shape) for name, value in sorted(features.baselines.items())
        }
        payload["graph_shapes"] = {
            name: tuple(value.shape) for name, value in sorted(features.graphs.items())
        }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()




def pack_weighted_adjacency(adjacency: torch.Tensor) -> dict[str, Any]:
    """Losslessly pack a small batched weighted graph with uint8 COO indices."""

    if adjacency.ndim != 3 or max(adjacency.shape) > 256:
        raise ValueError("packed adjacency requires [B,P,P] dimensions at most 256")
    coordinates = torch.nonzero(adjacency != 0, as_tuple=False)
    return {
        "format": "packed_weighted_adjacency_v1",
        "shape": list(adjacency.shape),
        "indices": coordinates.transpose(0, 1).to(torch.uint8),
        "values": adjacency[tuple(coordinates.transpose(0, 1))],
    }


def unpack_weighted_adjacency(
    payload: Any,
    *,
    map_location: str | torch.device = "cpu",
) -> torch.Tensor:
    """Reconstruct a dense adjacency exactly from the packed cache payload."""

    if not isinstance(payload, dict) or payload.get("format") != (
        "packed_weighted_adjacency_v1"
    ):
        raise ValueError("Unsupported packed adjacency payload")
    shape = tuple(int(value) for value in payload["shape"])
    indices = payload["indices"].to(device=map_location, dtype=torch.long)
    values = payload["values"].to(device=map_location)
    adjacency = torch.zeros(shape, dtype=values.dtype, device=values.device)
    adjacency[tuple(indices)] = values
    return adjacency


def _fingerprint_contract(
    features: FieldFeatures, storage_policy: str
) -> FieldFeatures:
    if storage_policy == "dense":
        return features
    return FieldFeatures(
        state=features.state,
        response=features.response,
        affinity=features.adjacency,
        adjacency=features.adjacency,
        grid_size=features.grid_size,
        baselines=features.baselines,
        graphs={
            name: value
            for name, value in features.graphs.items()
            if name.endswith("_adjacency")
        },
        metadata=features.metadata,
    )


def save_features(
    path: str | Path,
    features: FieldFeatures,
    *,
    targets: dict[str, torch.Tensor] | None = None,
    sample_ids: list[str] | None = None,
    storage_policy: str = "dense",
) -> dict[str, Any]:
    features.validate()
    if storage_policy not in {"dense", "readout_sparse"}:
        raise ValueError("storage_policy must be dense or readout_sparse")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpu = features.detached_cpu()
    compressed_targets: dict[str, torch.Tensor] = {}
    for name, value in (targets or {}).items():
        tensor = value.detach().cpu()
        if (
            name == "segmentation"
            and tensor.numel()
            and tensor.min() >= 0
            and tensor.max() <= 255
        ):
            tensor = tensor.to(torch.uint8)
        elif name == "classification":
            tensor = tensor.to(torch.int32)
        compressed_targets[name] = tensor
    manifest = {
        "format_version": FORMAT_VERSION,
        "fingerprint": feature_fingerprint(
            _fingerprint_contract(cpu, storage_policy)
        ),
        "num_samples": cpu.state.shape[0],
        "grid_size": cpu.grid_size,
        "state_dim": cpu.state.shape[-1],
        "response_dim": cpu.response.shape[-1],
        "metadata": cpu.metadata,
        "sample_ids": sample_ids,
        "target_dtypes": {
            name: str(value.dtype).removeprefix("torch.")
            for name, value in compressed_targets.items()
        },
        "storage_policy": storage_policy,
        "derived_adjacency": storage_policy == "dense",
        "dense_affinity_available": storage_policy == "dense",
    }
    if storage_policy == "dense":
        serialized_features = {
            "state": cpu.state,
            "response": cpu.response,
            "affinity": cpu.affinity,
            "baselines": cpu.baselines,
            "graphs": {
                name: value
                for name, value in cpu.graphs.items()
                if not name.endswith("_adjacency")
            },
        }
    else:
        serialized_features = {
            "state": cpu.state,
            "response": cpu.response,
            "adjacency": pack_weighted_adjacency(cpu.adjacency),
            "baselines": cpu.baselines,
            "graphs": {
                name: pack_weighted_adjacency(value)
                for name, value in cpu.graphs.items()
                if name.endswith("_adjacency")
            },
        }
    payload = {
        "manifest": manifest,
        "features": serialized_features,
        "targets": compressed_targets,
    }
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def load_features(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[FieldFeatures, dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    manifest = payload["manifest"]
    format_version = int(manifest.get("format_version", 0))
    if format_version not in {1, 2, 3, FORMAT_VERSION}:
        raise ValueError(f"Unsupported cache format: {manifest.get('format_version')}")
    raw = payload["features"]
    metadata = manifest.get("metadata", {})
    probe = metadata.get("probe", {})
    storage_policy = manifest.get("storage_policy", "dense")
    affinity = raw.get("affinity")
    adjacency = raw.get("adjacency")
    if isinstance(adjacency, dict):
        adjacency = unpack_weighted_adjacency(
            adjacency, map_location=map_location
        )
    if adjacency is None:
        if affinity is None:
            raise ValueError("Cache contains neither affinity nor adjacency")
        adjacency = sparsify_affinity(
            affinity,
            tuple(manifest["grid_size"]),
            int(probe["topk"]),
            int(probe["local_radius"]),
        )
    elif adjacency.is_sparse:
        adjacency = adjacency.to_dense()
    if affinity is None:
        affinity = adjacency
    graphs = dict(raw.get("graphs", {}))
    graphs = {
        name: (
            unpack_weighted_adjacency(value, map_location=map_location)
            if isinstance(value, dict)
            else (value.to_dense() if value.is_sparse else value)
        )
        for name, value in graphs.items()
    }
    if "dit_attention" in graphs and "dit_attention_adjacency" not in graphs:
        graphs["dit_attention_adjacency"] = sparsify_affinity(
            graphs["dit_attention"],
            tuple(manifest["grid_size"]),
            int(probe["topk"]),
            int(probe["local_radius"]),
        )
    features = FieldFeatures(
        state=raw["state"],
        response=raw["response"],
        affinity=affinity,
        adjacency=adjacency,
        grid_size=tuple(manifest["grid_size"]),
        baselines=raw.get("baselines", {}),
        graphs=graphs,
        metadata=metadata,
    )
    features.validate()
    features.metadata = {**features.metadata, "cache_storage_policy": storage_policy}
    expected = manifest["fingerprint"]
    fingerprint_features = _fingerprint_contract(
        FieldFeatures(
            state=features.state,
            response=features.response,
            affinity=features.affinity,
            adjacency=features.adjacency,
            grid_size=features.grid_size,
            baselines=features.baselines,
            graphs=features.graphs,
            metadata={
                key: value
                for key, value in features.metadata.items()
                if key != "cache_storage_policy"
            },
        ),
        storage_policy,
    )
    if feature_fingerprint(fingerprint_features, format_version=format_version) != expected:
        raise ValueError("Feature cache fingerprint mismatch")
    return features, payload.get("targets", {}), manifest
