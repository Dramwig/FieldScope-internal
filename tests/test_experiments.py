from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

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

    resumed = extract_dataset_cache(_config(tmp_path), resume=True, **arguments)
    assert resumed["complete"] is True
    assert [shard["status"] for shard in resumed["shards"]] == ["reused", "reused"]


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
