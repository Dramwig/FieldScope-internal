"""Portable feature-cache serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from fieldscope.contracts import FieldFeatures

FORMAT_VERSION = 1


def feature_fingerprint(features: FieldFeatures) -> str:
    payload = {
        "grid_size": features.grid_size,
        "state_shape": tuple(features.state.shape),
        "response_shape": tuple(features.response.shape),
        "metadata": features.metadata,
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
    manifest = {
        "format_version": FORMAT_VERSION,
        "fingerprint": feature_fingerprint(cpu),
        "num_samples": cpu.state.shape[0],
        "grid_size": cpu.grid_size,
        "state_dim": cpu.state.shape[-1],
        "response_dim": cpu.response.shape[-1],
        "metadata": cpu.metadata,
        "sample_ids": sample_ids,
    }
    torch.save(
        {
            "manifest": manifest,
            "features": {
                "state": cpu.state,
                "response": cpu.response,
                "affinity": cpu.affinity,
                "adjacency": cpu.adjacency,
                "baselines": cpu.baselines,
            },
            "targets": {
                name: value.detach().cpu() for name, value in (targets or {}).items()
            },
        },
        destination,
    )
    return manifest


def load_features(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[FieldFeatures, dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    manifest = payload["manifest"]
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported cache format: {manifest.get('format_version')}")
    raw = payload["features"]
    features = FieldFeatures(
        state=raw["state"],
        response=raw["response"],
        affinity=raw["affinity"],
        adjacency=raw["adjacency"],
        grid_size=tuple(manifest["grid_size"]),
        baselines=raw.get("baselines", {}),
        metadata=manifest.get("metadata", {}),
    )
    features.validate()
    expected = manifest["fingerprint"]
    if feature_fingerprint(features) != expected:
        raise ValueError("Feature cache fingerprint mismatch")
    return features, payload.get("targets", {}), manifest

