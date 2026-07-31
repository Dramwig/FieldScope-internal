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

FORMAT_VERSION = 3


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


def save_features(
    path: str | Path,
    features: FieldFeatures,
    *,
    targets: dict[str, torch.Tensor] | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    features.validate()
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
        "fingerprint": feature_fingerprint(cpu),
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
        "derived_adjacency": True,
    }
    payload = {
        "manifest": manifest,
        "features": {
            "state": cpu.state,
            "response": cpu.response,
            "affinity": cpu.affinity,
            "baselines": cpu.baselines,
            "graphs": {
                name: value
                for name, value in cpu.graphs.items()
                if not name.endswith("_adjacency")
            },
        },
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
    if format_version not in {1, 2, FORMAT_VERSION}:
        raise ValueError(f"Unsupported cache format: {manifest.get('format_version')}")
    raw = payload["features"]
    metadata = manifest.get("metadata", {})
    probe = metadata.get("probe", {})
    adjacency = raw.get("adjacency")
    if adjacency is None:
        adjacency = sparsify_affinity(
            raw["affinity"],
            tuple(manifest["grid_size"]),
            int(probe["topk"]),
            int(probe["local_radius"]),
        )
    graphs = dict(raw.get("graphs", {}))
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
        affinity=raw["affinity"],
        adjacency=adjacency,
        grid_size=tuple(manifest["grid_size"]),
        baselines=raw.get("baselines", {}),
        graphs=graphs,
        metadata=metadata,
    )
    features.validate()
    expected = manifest["fingerprint"]
    if feature_fingerprint(features, format_version=format_version) != expected:
        raise ValueError("Feature cache fingerprint mismatch")
    return features, payload.get("targets", {}), manifest
