"""Sharded FieldFeatures dataset for offline readout training."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

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
        self.shard_ranges: list[tuple[int, int]] = []
        for shard in self.manifest["shards"]:
            path = self.cache_dir / shard["path"]
            start = len(self.index)
            self.index.extend((path, index) for index in range(shard["num_samples"]))
            self.shard_ranges.append((start, len(self.index)))
        self._loaded_payloads: OrderedDict[
            Path,
            tuple[FieldFeatures, dict[str, torch.Tensor], dict[str, Any]],
        ] = OrderedDict()
        self._max_loaded_shards = 4

    def __len__(self) -> int:
        return len(self.index)

    def _load(
        self, path: Path
    ) -> tuple[FieldFeatures, dict[str, torch.Tensor], dict[str, Any]]:
        if path in self._loaded_payloads:
            payload = self._loaded_payloads.pop(path)
            self._loaded_payloads[path] = payload
            return payload
        payload = load_features(path)
        self._loaded_payloads[path] = payload
        while len(self._loaded_payloads) > self._max_loaded_shards:
            self._loaded_payloads.popitem(last=False)
        return payload

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
        self.shard_ranges = self.dataset.shard_ranges
        generator = torch.Generator().manual_seed(seed)
        self.donor_for_index = torch.empty(len(self.dataset), dtype=torch.long)
        if len(self.shard_ranges) == 1:
            order = torch.randperm(len(self.dataset), generator=generator)
            self.donor_for_index[order] = order.roll(1)
        else:
            shard_order = torch.randperm(len(self.shard_ranges), generator=generator)
            donor_for_shard = torch.empty_like(shard_order)
            donor_for_shard[shard_order] = shard_order.roll(1)
            for shard_index, (start, stop) in enumerate(self.shard_ranges):
                donor_start, donor_stop = self.shard_ranges[
                    int(donor_for_shard[shard_index].item())
                ]
                donor_count = donor_stop - donor_start
                rotation = int(
                    torch.randint(donor_count, (1,), generator=generator).item()
                )
                local = torch.arange(stop - start)
                self.donor_for_index[start:stop] = (
                    donor_start + (local + rotation) % donor_count
                )
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


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shard order and samples while keeping cache reads localized."""

    def __init__(self, dataset: Any, seed: int):
        self.shard_ranges = list(dataset.shard_ranges)
        self.seed = seed
        self.length = sum(stop - start for start, stop in self.shard_ranges)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        shard_order = torch.randperm(len(self.shard_ranges), generator=generator)
        for shard_index in shard_order.tolist():
            start, stop = self.shard_ranges[shard_index]
            local_order = torch.randperm(stop - start, generator=generator)
            yield from (start + local_order).tolist()


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
