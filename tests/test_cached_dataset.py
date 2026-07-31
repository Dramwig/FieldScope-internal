import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.cache import save_features
from fieldscope.cached_dataset import (
    CachedFeatureDataset,
    ShuffledResponseCachedDataset,
    collate_cached,
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
    path = cache_dir / "shard-000000.pt"
    save_features(
        path,
        features,
        targets={"classification": torch.tensor([0, 1, 0])},
        sample_ids=["a", "b", "c"],
    )
    (cache_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": "synthetic",
                "split": "train",
                "num_samples": 3,
                "shards": [{"path": path.name, "num_samples": 3}],
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
    batch = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_cached))
    )
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


def test_shuffled_response_dataset_has_no_self_donors(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path)
    regular = CachedFeatureDataset(cache_dir)
    shuffled = ShuffledResponseCachedDataset(cache_dir, seed=17)
    assert torch.all(
        shuffled.donor_for_index != torch.arange(len(shuffled))
    )
    sample = shuffled[0]
    assert torch.equal(sample["features"].state, regular[0]["features"].state)
    assert (
        sample["features"].metadata["response_donor_sample_id"]
        != sample["sample_id"]
    )
    _, mode = select_representation(sample["features"], "full_shuffled")
    assert mode == "full"
