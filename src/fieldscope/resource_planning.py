"""Disk-budget planning from measured FieldScope cache shards."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fieldscope.experiments import file_sha256


def measured_bytes_per_sample(
    cache_dir: Path,
    *,
    required_storage_policy: str = "readout_sparse",
) -> float:
    manifest_path = cache_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"Cache is incomplete: {cache_dir}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Cache has no shards: {cache_dir}")
    samples = 0
    cache_bytes = 0
    for shard in shards:
        path = cache_dir / str(shard.get("path", ""))
        expected_bytes = int(shard.get("bytes", -1))
        sample_count = int(shard.get("num_samples", 0))
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"Cache shard is missing or has changed size: {path}")
        expected_sha256 = shard.get("sha256")
        if expected_sha256 and file_sha256(path) != expected_sha256:
            raise ValueError(f"Cache shard checksum mismatch: {path}")
        samples += sample_count
        cache_bytes += path.stat().st_size
    if samples < 1 or cache_bytes < 1:
        raise ValueError(f"Cache has no measurable samples: {cache_dir}")
    if int(manifest.get("num_samples", samples)) != samples:
        raise ValueError("Cache manifest sample count does not match its shards")
    if required_storage_policy not in {"dense", "readout_sparse"}:
        raise ValueError("required_storage_policy must be dense or readout_sparse")
    if manifest.get("storage_policy") != required_storage_policy:
        raise ValueError(
            "Disk projection requires a "
            f"{required_storage_policy} measurement cache"
        )
    return cache_bytes / samples


def plan_cache_budget(
    *,
    measurement_cache: Path,
    target_samples: Mapping[str, int],
    filesystem_path: Path,
    reserve_bytes: int = 10 * 1024**3,
    safety_factor: float = 1.15,
    required_storage_policy: str = "readout_sparse",
    additional_required_bytes: int = 0,
) -> dict[str, Any]:
    """Project a full readout cache and decide whether extraction may start."""

    if reserve_bytes < 0 or safety_factor < 1.0 or additional_required_bytes < 0:
        raise ValueError("reserve_bytes and safety_factor are invalid")
    if not target_samples or any(value < 1 for value in target_samples.values()):
        raise ValueError("target_samples must contain positive split counts")
    measured = measured_bytes_per_sample(
        measurement_cache,
        required_storage_policy=required_storage_policy,
    )
    projected = {
        split: int(round(count * measured * safety_factor))
        for split, count in target_samples.items()
    }
    projected_total = sum(projected.values())
    free_bytes = shutil.disk_usage(filesystem_path).free
    required_bytes = projected_total + reserve_bytes + additional_required_bytes
    return {
        "status": "passed" if required_bytes <= free_bytes else "resource_blocked",
        "fits": required_bytes <= free_bytes,
        "measurement_cache": str(measurement_cache),
        "measurement_bytes_per_sample": measured,
        "required_storage_policy": required_storage_policy,
        "safety_factor": safety_factor,
        "target_samples": dict(target_samples),
        "projected_split_bytes": projected,
        "projected_total_bytes": projected_total,
        "reserve_bytes": reserve_bytes,
        "additional_required_bytes": additional_required_bytes,
        "required_bytes": required_bytes,
        "filesystem_path": str(filesystem_path),
        "free_bytes": free_bytes,
        "headroom_bytes": free_bytes - required_bytes,
    }
