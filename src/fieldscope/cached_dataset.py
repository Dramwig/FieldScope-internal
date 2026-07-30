"""Sharded FieldFeatures dataset for offline readout training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fieldscope.cache import load_features
from fieldscope.contracts import FieldFeatures
from fieldscope.feature_ops import slice_features, stack_features


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.index: list[tuple[Path, int]] = []
        for shard in self.manifest["shards"]:
            path = self.cache_dir / shard["path"]
            self.index.extend((path, index) for index in range(shard["num_samples"]))
        self._loaded_path: Path | None = None
        self._loaded_payload: tuple[
            FieldFeatures, dict[str, torch.Tensor], dict[str, Any]
        ] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load(
        self, path: Path
    ) -> tuple[FieldFeatures, dict[str, torch.Tensor], dict[str, Any]]:
        if path != self._loaded_path:
            self._loaded_payload = load_features(path)
            self._loaded_path = path
        assert self._loaded_payload is not None
        return self._loaded_payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, local_index = self.index[index]
        features, targets, manifest = self._load(path)
        sample_ids = manifest.get("sample_ids") or []
        return {
            "features": slice_features(features, local_index),
            "targets": {
                name: tensor[local_index] for name, tensor in targets.items()
            },
            "sample_id": sample_ids[local_index] if local_index < len(sample_ids) else str(index),
        }


def collate_cached(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    target_names = set(samples[0]["targets"])
    if any(set(sample["targets"]) != target_names for sample in samples):
        raise ValueError("Cached samples have inconsistent targets")
    return {
        "features": stack_features([sample["features"] for sample in samples]),
        "targets": {
            name: torch.stack([sample["targets"][name] for sample in samples])
            for name in target_names
        },
        "sample_ids": [sample["sample_id"] for sample in samples],
    }

