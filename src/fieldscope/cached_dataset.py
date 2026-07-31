"""Sharded FieldFeatures dataset for offline readout training."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from fieldscope.cache import load_features
from fieldscope.contracts import FieldFeatures
from fieldscope.feature_ops import select_representation, slice_features, stack_features

_SHARED_PAYLOADS: OrderedDict[
    Path,
    tuple[FieldFeatures, dict[str, torch.Tensor], dict[str, Any]],
] = OrderedDict()
_SHARED_PAYLOAD_BYTES: dict[Path, int] = {}
_SHARED_TOTAL_BYTES = 0
_MAX_SHARDS_PER_RESPONSE_POOL = 32


def _manifest_shard_ranges(manifest: dict[str, Any]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    stop = 0
    for shard in manifest["shards"]:
        start = stop
        stop += int(shard["num_samples"])
        ranges.append((start, stop))
    return ranges


def _response_shuffle_assignment(
    shard_ranges: list[tuple[int, int]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    length = sum(stop - start for start, stop in shard_ranges)
    if length < 2:
        raise ValueError("Response shuffling requires at least two samples")
    generator = torch.Generator().manual_seed(seed)
    shard_order = torch.randperm(len(shard_ranges), generator=generator)
    pool_count = max(
        1,
        (len(shard_ranges) + _MAX_SHARDS_PER_RESPONSE_POOL - 1)
        // _MAX_SHARDS_PER_RESPONSE_POOL,
    )
    shard_pools = torch.tensor_split(shard_order, pool_count)
    donor_for_index = torch.empty(length, dtype=torch.long)
    pool_for_index = torch.empty(length, dtype=torch.long)
    for pool_index, shard_pool in enumerate(shard_pools):
        pooled_indices = torch.cat(
            [
                torch.arange(*shard_ranges[int(shard_index)])
                for shard_index in shard_pool.tolist()
            ]
        )
        if pooled_indices.numel() < 2:
            raise AssertionError("Response shuffle pool has fewer than two samples")
        cycle = pooled_indices[
            torch.randperm(pooled_indices.numel(), generator=generator)
        ]
        donor_for_index[cycle] = cycle.roll(1)
        pool_for_index[pooled_indices] = pool_index
    if not torch.equal(
        donor_for_index.sort().values,
        torch.arange(length, dtype=torch.long),
    ):
        raise AssertionError("Response shuffle is not a one-to-one permutation")
    if torch.any(donor_for_index == torch.arange(length)):
        raise AssertionError("Response shuffle unexpectedly contains a fixed point")
    if torch.any(pool_for_index != pool_for_index[donor_for_index]):
        raise AssertionError("Response shuffle crossed a cache-locality pool")
    return donor_for_index, pool_for_index, pool_count


def cached_control_contract_for_cache(
    cache_dir: str | Path,
    representation: str,
    seed: int,
) -> dict[str, Any]:
    manifest = json.loads(
        (Path(cache_dir) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    num_samples = sum(int(shard["num_samples"]) for shard in manifest["shards"])
    if representation in {"response_shuffled", "full_shuffled"}:
        donor_for_index, _, pool_count = _response_shuffle_assignment(
            _manifest_shard_ranges(manifest),
            seed,
        )
        return {
            "response_shuffle": {
                "policy": "seeded_random_pooled_derangement_v2",
                "seed": seed,
                "pool_count": pool_count,
                "max_shards_per_pool": _MAX_SHARDS_PER_RESPONSE_POOL,
                "num_samples": num_samples,
                "donor_permutation_sha256": hashlib.sha256(
                    donor_for_index.numpy().tobytes()
                ).hexdigest(),
            }
        }
    if representation == "random_feature_local":
        return {
            "random_feature": {
                "policy": "sample_id_sha256_seeded_v1",
                "seed": seed,
                "num_samples": num_samples,
            }
        }
    return {}


def shared_memory_cache_stats() -> dict[str, int]:
    return {
        "shards": len(_SHARED_PAYLOADS),
        "estimated_bytes": _SHARED_TOTAL_BYTES,
    }


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        memory_cache_bytes: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.memory_cache_bytes = memory_cache_bytes
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
        global _SHARED_TOTAL_BYTES
        if self.memory_cache_bytes > 0 and path in _SHARED_PAYLOADS:
            payload = _SHARED_PAYLOADS.pop(path)
            _SHARED_PAYLOADS[path] = payload
            return payload
        if path in self._loaded_payloads:
            payload = self._loaded_payloads.pop(path)
            self._loaded_payloads[path] = payload
            return payload
        payload = load_features(path)
        if self.memory_cache_bytes > 0:
            estimated_bytes = path.stat().st_size
            _SHARED_PAYLOADS[path] = payload
            _SHARED_PAYLOAD_BYTES[path] = estimated_bytes
            _SHARED_TOTAL_BYTES += estimated_bytes
            while (
                _SHARED_TOTAL_BYTES > self.memory_cache_bytes
                and len(_SHARED_PAYLOADS) > 1
            ):
                evicted_path, _ = _SHARED_PAYLOADS.popitem(last=False)
                _SHARED_TOTAL_BYTES -= _SHARED_PAYLOAD_BYTES.pop(evicted_path)
            return payload
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

    _MAX_SHARDS_PER_POOL = _MAX_SHARDS_PER_RESPONSE_POOL

    def __init__(
        self,
        cache_dir: str | Path,
        seed: int,
        *,
        memory_cache_bytes: int = 0,
    ):
        self.dataset = CachedFeatureDataset(
            cache_dir,
            memory_cache_bytes=memory_cache_bytes,
        )
        self.shard_ranges = self.dataset.shard_ranges
        (
            self.donor_for_index,
            self.pool_for_index,
            pool_count,
        ) = _response_shuffle_assignment(self.shard_ranges, seed)
        self.shuffle_policy = "seeded_random_pooled_derangement_v2"
        self.shuffle_seed = seed
        self.shuffle_pool_count = pool_count
        self.donor_permutation_sha256 = hashlib.sha256(
            self.donor_for_index.numpy().tobytes()
        ).hexdigest()

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
                "response_shuffle_policy": self.shuffle_policy,
                "response_shuffle_seed": self.shuffle_seed,
                "response_shuffle_pool": int(self.pool_for_index[index].item()),
                "response_shuffle_pool_count": self.shuffle_pool_count,
                "response_shuffle_max_shards_per_pool": self._MAX_SHARDS_PER_POOL,
                "response_shuffle_permutation_sha256": self.donor_permutation_sha256,
            },
        )
        features.validate()
        return {
            "features": features,
            "targets": receiver["targets"],
            "sample_id": receiver["sample_id"],
        }


class RandomFeatureCachedDataset(Dataset[dict[str, Any]]):
    """Replace visual evidence with deterministic sample-ID random patch features."""

    def __init__(
        self,
        cache_dir: str | Path,
        seed: int,
        *,
        memory_cache_bytes: int = 0,
    ):
        self.dataset = CachedFeatureDataset(
            cache_dir,
            memory_cache_bytes=memory_cache_bytes,
        )
        self.seed = seed
        self.shard_ranges = self.dataset.shard_ranges

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        source = sample["features"]
        sample_id = str(sample["sample_id"])
        encoded = f"{self.seed}\0{sample_id}".encode()
        random_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        random_features = torch.randn(
            (1, source.state.shape[1], 768),
            generator=generator,
            dtype=torch.float32,
        )
        features = FieldFeatures(
            state=random_features,
            response=source.response.float(),
            affinity=source.affinity.float(),
            adjacency=source.adjacency.float(),
            grid_size=source.grid_size,
            baselines=source.baselines,
            graphs=source.graphs,
            metadata={
                **source.metadata,
                "random_feature_policy": "sample_id_sha256_seeded_v1",
            },
        )
        features.validate()
        return {
            "features": features,
            "targets": sample["targets"],
            "sample_id": sample_id,
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


def cached_control_contract(dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, ShuffledResponseCachedDataset):
        return {
            "response_shuffle": {
                "policy": dataset.shuffle_policy,
                "seed": dataset.shuffle_seed,
                "pool_count": dataset.shuffle_pool_count,
                "max_shards_per_pool": dataset._MAX_SHARDS_PER_POOL,
                "num_samples": len(dataset),
                "donor_permutation_sha256": dataset.donor_permutation_sha256,
            }
        }
    if isinstance(dataset, RandomFeatureCachedDataset):
        return {
            "random_feature": {
                "policy": "sample_id_sha256_seeded_v1",
                "seed": dataset.seed,
                "num_samples": len(dataset),
            }
        }
    return {}


def collate_cached(
    samples: list[dict[str, Any]],
    *,
    representation: str | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    target_names = set(samples[0]["targets"])
    if any(set(sample["targets"]) != target_names for sample in samples):
        raise ValueError("Cached samples have inconsistent targets")
    selected_features = [sample["features"] for sample in samples]
    tokenizer_mode = None
    if representation is not None:
        selected = [
            select_representation(features, representation)
            for features in selected_features
        ]
        modes = {mode for _, mode in selected}
        if len(modes) != 1:
            raise ValueError("Cached samples selected inconsistent tokenizer modes")
        selected_features = [features for features, _ in selected]
        tokenizer_mode = modes.pop()
    return {
        "features": stack_features(selected_features),
        "targets": {
            name: torch.stack([sample["targets"][name] for sample in samples])
            for name in target_names
        },
        "sample_ids": [sample["sample_id"] for sample in samples],
        "tokenizer_mode": tokenizer_mode,
    }
