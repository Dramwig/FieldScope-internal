import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from fieldscope.cached_dataset import CachedFeatureDataset
from fieldscope.config import RunConfig
from fieldscope.experiments import extract_dataset_cache, run_readout_matrix


class _SmallDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.full((3, 16, 16), index / 4),
            "classification": torch.tensor(index % 2),
            "sample_id": f"sample-{index}",
        }


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "backend": {"name": "toy", "device": "cpu", "image_size": 16},
            "probe": {
                "times": [0.5],
                "num_directions": 1,
                "graph_grid": [2, 2],
                "topk": 2,
                "probe_batch_size": 2,
                "antithetic_noise": False,
                "seed": 11,
            },
            "tokenizer": {
                "hidden_dim": 8,
                "num_layers": 1,
                "num_classes": 2,
                "segmentation_classes": 2,
            },
            "runtime": {
                "output_dir": str(tmp_path / "outputs"),
                "batch_size": 2,
                "cache_shard_size": 3,
                "num_workers": 0,
            },
        }
    )


def test_extract_dataset_cache_resumes_verified_shards(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    output = tmp_path / "cache"
    arguments = {
        "dataset_name": "synthetic-test",
        "dataset_root": tmp_path,
        "split": "train",
        "output_dir": output,
        "limit": 4,
    }
    first = extract_dataset_cache(_config(tmp_path), **arguments)
    assert first["complete"] is True
    assert [shard["status"] for shard in first["shards"]] == ["written", "written"]
    assert [shard["num_samples"] for shard in first["shards"]] == [3, 1]
    assert all(shard["sha256"] for shard in first["shards"])
    assert first["code_tree_sha256"]

    manifest_path = output / "dataset_manifest.json"
    interrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
    interrupted["complete"] = False
    manifest_path.write_text(json.dumps(interrupted), encoding="utf-8")
    resumed = extract_dataset_cache(_config(tmp_path), resume=True, **arguments)
    assert resumed["complete"] is True
    assert [shard["status"] for shard in resumed["shards"]] == ["reused", "reused"]
    complete = extract_dataset_cache(_config(tmp_path), resume=True, **arguments)
    assert complete["resume_status"] == "already_complete"

    first_shard = output / first["shards"][0]["path"]
    first_shard.write_bytes(first_shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="integrity"):
        extract_dataset_cache(_config(tmp_path), resume=True, **arguments)


def test_dataset_cache_is_invariant_to_batch_and_shard_layout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    first_mapping = _config(tmp_path).to_dict()
    first_mapping["runtime"]["batch_size"] = 1
    first_mapping["runtime"]["cache_shard_size"] = 2
    second_mapping = _config(tmp_path).to_dict()
    second_mapping["runtime"]["batch_size"] = 2
    second_mapping["runtime"]["cache_shard_size"] = 3
    first_dir = tmp_path / "cache-layout-a"
    second_dir = tmp_path / "cache-layout-b"
    common = {
        "dataset_name": "synthetic-test",
        "dataset_root": tmp_path,
        "split": "train",
        "limit": 4,
    }
    first_report = extract_dataset_cache(
        RunConfig.from_mapping(first_mapping),
        output_dir=first_dir,
        **common,
    )
    second_report = extract_dataset_cache(
        RunConfig.from_mapping(second_mapping),
        output_dir=second_dir,
        **common,
    )
    assert first_report["randomness"] == second_report["randomness"]
    assert first_report["randomness"]["path_noise"] == "sample_id_sha256_seeded_v1"
    first = CachedFeatureDataset(first_dir)
    second = CachedFeatureDataset(second_dir)
    assert len(first) == len(second) == 4
    for index in range(4):
        first_sample = first[index]
        second_sample = second[index]
        assert first_sample["sample_id"] == second_sample["sample_id"]
        first_features = first_sample["features"]
        second_features = second_sample["features"]
        torch.testing.assert_close(first_features.state, second_features.state)
        torch.testing.assert_close(first_features.response, second_features.response)
        torch.testing.assert_close(first_features.affinity, second_features.affinity)
        torch.testing.assert_close(first_features.adjacency, second_features.adjacency)
        for name in first_features.baselines:
            torch.testing.assert_close(
                first_features.baselines[name],
                second_features.baselines[name],
            )
        for name in first_features.graphs:
            torch.testing.assert_close(
                first_features.graphs[name],
                second_features.graphs[name],
            )


def test_readout_matrix_runs_and_resumes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    config = _config(tmp_path)
    caches = {}
    for split in ("train", "val", "test"):
        cache_dir = tmp_path / f"cache-{split}"
        extract_dataset_cache(
            config,
            dataset_name="synthetic-test",
            dataset_root=tmp_path,
            split=split,
            output_dir=cache_dir,
            limit=4,
        )
        caches[split] = cache_dir
    arguments = {
        "train_cache_dir": caches["train"],
        "val_cache_dir": caches["val"],
        "test_cache_dir": caches["test"],
        "output_dir": tmp_path / "matrix",
        "task": "classification",
        "representations": ["state", "full"],
        "seeds": [3],
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 2,
        "reference": "state",
    }
    first = run_readout_matrix(config, **arguments)
    second = run_readout_matrix(config, **arguments)
    assert first["status"] == "passed"
    assert second["status"] == "passed"
    assert len(first["runs"]) == 2
    assert "classification/full-minus-state" in first["summary"]["comparisons"]
