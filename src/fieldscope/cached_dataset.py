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


class ShuffledResponseCachedDataset(Dataset[dict[str, Any]]):
    """Replace response signatures/graphs with a deterministic donor sample."""

    def __init__(self, cache_dir: str | Path, seed: int):
        self.dataset = CachedFeatureDataset(cache_dir)
        if len(self.dataset) < 2:
            raise ValueError("Response shuffling requires at least two samples")
        order = torch.randperm(
            len(self.dataset),
            generator=torch.Generator().manual_seed(seed),
        )
        donors = order.roll(1)
        self.donor_for_index = torch.empty_like(order)
        self.donor_for_index[order] = donors
        if torch.any(self.donor_for_index == torch.arange(len(self.dataset))):
            raise AssertionError("Response shuffle unexpectedly contains a fixed point")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        receiver = self.dataset[index]
        donor_index = int(self.donor_for_index[index].item())
        donor = self.dataset[donor_index]
        receiver_features = receiver["features"]
        donor_features = donor["features"]
        features = FieldFeatures(
            state=receiver_features.state,
            response=donor_features.response,
            affinity=donor_features.affinity,
            adjacency=donor_features.adjacency,
            grid_size=receiver_features.grid_size,
            baselines=receiver_features.baselines,
            graphs=receiver_features.graphs,
            metadata={
                **receiver_features.metadata,
                "response_donor_sample_id": donor["sample_id"],
            },
        )
        features.validate()
        return {
            "features": features,
            "targets": receiver["targets"],
            "sample_id": receiver["sample_id"],
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
