import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.cache import save_features
from fieldscope.cached_dataset import (
    CachedFeatureDataset,
    RandomFeatureCachedDataset,
    ShardShuffleSampler,
    ShuffledResponseCachedDataset,
    cached_control_contract,
    collate_cached,
    shared_memory_cache_stats,
)
from fieldscope.cli import train_cache
from fieldscope.config import ProbeConfig, RunConfig
from fieldscope.feature_ops import select_representation, slice_features, stack_features
from fieldscope.response import FieldResponseExtractor


def _write_cache(tmp_path: Path) -> tuple[Path, RunConfig]:
    probe = ProbeConfig(
        times=(0.5,),
        num_directions=2,
        graph_grid=(4, 4),
        topk=3,
        probe_batch_size=4,
        antithetic_noise=False,
        seed=7,
    )
    features = FieldResponseExtractor(ToyFieldBackend(image_size=32), probe).extract(
        torch.rand(3, 3, 32, 32)
    )
    cache_dir = tmp_path / "cache"
    labels = torch.tensor([0, 1, 0])
    sample_ids = ["a", "b", "c"]
    shards = []
    for shard_index, (start, stop) in enumerate(((0, 2), (2, 3))):
        path = cache_dir / f"shard-{shard_index:06d}.pt"
        shard_features = stack_features(
            [slice_features(features, index) for index in range(start, stop)]
        )
        save_features(
            path,
            shard_features,
            targets={"classification": labels[start:stop]},
            sample_ids=sample_ids[start:stop],
        )
        shards.append({"path": path.name, "num_samples": stop - start})
    (cache_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": "synthetic",
                "split": "train",
                "num_samples": 3,
                "complete": True,
                "storage_policy": "dense",
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    config = RunConfig.from_mapping(
        {
            "backend": {"name": "toy", "device": "cpu", "image_size": 32},
            "probe": {
                "times": [0.5],
                "num_directions": 2,
                "graph_grid": [4, 4],
            },
            "tokenizer": {
                "hidden_dim": 16,
                "num_layers": 1,
                "num_classes": 2,
                "segmentation_classes": 3,
            },
            "runtime": {
                "output_dir": str(tmp_path / "train"),
                "batch_size": 2,
            },
        }
    )
    return cache_dir, config


def test_feature_batch_ops_and_baselines(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    features = CachedFeatureDataset(cache_dir)[0]["features"]
    doubled = stack_features([features, features])
    assert doubled.state.shape[0] == 2
    assert slice_features(doubled, 1).state.shape[0] == 1
    baseline, mode = select_representation(features, "velocity")
    assert mode == "state"
    assert baseline.state.shape[-1] == 4
    hidden, mode = select_representation(features, "dit_hidden_local")
    assert mode == "state"
    assert hidden.state.shape[-1] == 768
    hidden_attention, mode = select_representation(
        features,
        "dit_hidden_attention",
    )
    assert mode == "state_graph"
    assert torch.equal(
        hidden_attention.adjacency,
        features.graphs["dit_attention_adjacency"],
    )


def test_cached_dataset_collate_and_train(tmp_path: Path) -> None:
    cache_dir, config = _write_cache(tmp_path)
    dataset = CachedFeatureDataset(cache_dir)
    assert len(dataset) == 3
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_cached)))
    assert batch["features"].state.shape[0] == 2
    report = train_cache(
        config,
        cache_dir=cache_dir,
        task="classification",
        representation="full",
        epochs=1,
        learning_rate=1e-3,
        batch_size=2,
    )
    assert report["status"] == "passed"
    assert Path(report["checkpoint"]).is_file()
    assert report["train_samples"] == 3
    assert report["optimizer_steps_per_epoch"] == 2
    assert report["completed_optimizer_steps"] == 2
    assert report["completed_training_sample_exposures"] == 3
    assert report["history"][0]["train_samples"] == 3
    assert report["history"][0]["optimizer_steps"] == 2
    assert report["history"][0]["train_samples_per_second"] > 0
    assert report["runtime"]["elapsed_seconds"] > 0
    resumed = train_cache(
        config,
        cache_dir=cache_dir,
        val_cache_dir=cache_dir,
        task="classification",
        representation="full",
        epochs=1,
        learning_rate=1e-3,
        resume_checkpoint=Path(report["last_checkpoint"]),
    )
    assert resumed["status"] == "passed"
    assert resumed["validation"]["num_samples"] == 3


def test_random_feature_control_is_sample_id_deterministic(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    first = RandomFeatureCachedDataset(cache_dir, seed=17)
    second = RandomFeatureCachedDataset(cache_dir, seed=17)
    different = RandomFeatureCachedDataset(cache_dir, seed=19)
    assert torch.equal(first[0]["features"].state, second[0]["features"].state)
    assert not torch.equal(first[0]["features"].state, different[0]["features"].state)
    assert first[0]["features"].state.shape[-1] == 768


def test_shuffled_response_dataset_has_no_self_donors(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    regular = CachedFeatureDataset(cache_dir)
    shuffled = ShuffledResponseCachedDataset(cache_dir, seed=17)
    repeated = ShuffledResponseCachedDataset(cache_dir, seed=17)
    different = ShuffledResponseCachedDataset(cache_dir, seed=19)
    assert torch.all(shuffled.donor_for_index != torch.arange(len(shuffled)))
    assert torch.equal(
        shuffled.donor_for_index.sort().values,
        torch.arange(len(shuffled)),
    )
    assert torch.equal(shuffled.donor_for_index, repeated.donor_for_index)
    assert not torch.equal(shuffled.donor_for_index, different.donor_for_index)
    assert torch.equal(
        shuffled.pool_for_index,
        shuffled.pool_for_index[shuffled.donor_for_index],
    )
    sample = shuffled[0]
    assert torch.equal(sample["features"].state, regular[0]["features"].state)
    assert sample["features"].metadata["response_donor_sample_id"] != sample["sample_id"]
    assert sample["features"].metadata["response_shuffle_policy"] == (
        "seeded_random_pooled_derangement_v2"
    )
    assert sample["features"].metadata["response_shuffle_seed"] == 17
    assert sample["features"].metadata["response_shuffle_pool_count"] == 1
    assert sample["features"].metadata["response_shuffle_max_shards_per_pool"] == 32
    assert sample["features"].metadata["response_shuffle_permutation_sha256"] == (
        shuffled.donor_permutation_sha256
    )
    contract = cached_control_contract(shuffled)["response_shuffle"]
    assert contract["seed"] == 17
    assert contract["num_samples"] == len(shuffled)
    assert contract["donor_permutation_sha256"] == shuffled.donor_permutation_sha256
    _, mode = select_representation(sample["features"], "full_shuffled")
    assert mode == "full"


def test_representation_collation_drops_unused_cached_fields(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    dataset = CachedFeatureDataset(cache_dir)
    batch = collate_cached(
        [dataset[0], dataset[1]],
        representation="z0",
    )
    assert batch["tokenizer_mode"] == "state"
    assert batch["features"].state.shape[-1] == 4
    assert batch["features"].response.shape[-1] == 0
    assert batch["features"].adjacency.untyped_storage().nbytes() == 4
    assert batch["features"].baselines == {}
    assert batch["features"].graphs == {}

    full = collate_cached(
        [dataset[0], dataset[1]],
        representation="full",
    )
    assert full["features"].state.shape[-1] > 0
    assert full["features"].response.shape[-1] > 0
    assert full["features"].adjacency.untyped_storage().nbytes() > 4

    local = collate_cached(
        [dataset[0], dataset[1]],
        representation="full_local",
    )
    assert local["features"].state.shape[-1] > 0
    assert local["features"].response.shape[-1] > 0
    assert local["features"].adjacency.untyped_storage().nbytes() == 4


@pytest.mark.parametrize(
    ("representation", "expected_mode"),
    [
        ("random_feature_local", "state"),
        ("z0", "state"),
        ("zt", "state"),
        ("trajectory", "state"),
        ("velocity", "state"),
        ("mismatch", "state"),
        ("endpoint", "state"),
        ("state", "state"),
        ("state_nograph", "state_nograph"),
        ("state_graph", "state_graph"),
        ("response_nograph", "response_nograph"),
        ("response_local", "response_local"),
        ("response", "response"),
        ("full_nograph", "full_nograph"),
        ("full_local", "full_local"),
        ("full", "full"),
        ("dit_hidden_local", "state"),
        ("dit_hidden_attention", "state_graph"),
        ("response_shuffled", "response"),
        ("full_shuffled", "full"),
    ],
)
def test_compact_collation_preserves_all_consumed_registered_tensors(
    tmp_path: Path,
    representation: str,
    expected_mode: str,
) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    dataset = CachedFeatureDataset(cache_dir)
    samples = [dataset[0], dataset[1]]
    selected = [
        select_representation(sample["features"], representation)[0]
        for sample in samples
    ]
    standard = stack_features(selected)
    compact_batch = collate_cached(samples, representation=representation)
    compact = compact_batch["features"]

    assert compact_batch["tokenizer_mode"] == expected_mode
    assert compact.grid_size == standard.grid_size
    if expected_mode in {
        "state",
        "state_graph",
        "state_nograph",
        "full",
        "full_local",
        "full_nograph",
    }:
        assert torch.equal(compact.state, standard.state)
    else:
        assert compact.state.shape[-1] == 0
    if expected_mode in {
        "response",
        "response_local",
        "response_nograph",
        "full",
        "full_local",
        "full_nograph",
    }:
        assert torch.equal(compact.response, standard.response)
    else:
        assert compact.response.shape[-1] == 0
    if expected_mode in {"state_graph", "response", "full"}:
        assert torch.equal(compact.adjacency, standard.adjacency)
    else:
        assert compact.adjacency.untyped_storage().nbytes() == 4


def test_shard_shuffle_sampler_is_deterministic_and_complete(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    dataset = CachedFeatureDataset(cache_dir)
    first = list(ShardShuffleSampler(dataset, seed=5))
    second = list(ShardShuffleSampler(dataset, seed=5))
    assert first == second
    assert sorted(first) == list(range(len(dataset)))


def test_shared_memory_cache_reuses_loaded_shards(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    before = shared_memory_cache_stats()
    first = CachedFeatureDataset(cache_dir, memory_cache_bytes=1024**3)
    _ = first[0]
    after_first = shared_memory_cache_stats()
    second = CachedFeatureDataset(cache_dir, memory_cache_bytes=1024**3)
    _ = second[1]
    after_second = shared_memory_cache_stats()
    assert after_first["shards"] == before["shards"] + 1
    assert after_second == after_first
