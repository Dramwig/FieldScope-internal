import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import fieldscope.experiments as experiments
from fieldscope.cache import save_features
from fieldscope.cached_dataset import CachedFeatureDataset
from fieldscope.config import RunConfig
from fieldscope.contracts import FieldFeatures
from fieldscope.experiments import (
    _task_loss_targets,
    diagnose_segmentation_cache,
    evaluate_checkpoint,
    extract_dataset_cache,
    run_readout_matrix,
    run_readout_matrix_seed_parallel,
    train_cached_readout,
)


class _SmallDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.full((3, 16, 16), index / 4),
            "classification": torch.tensor(index % 2),
            "sample_id": f"sample-{index}",
        }


def test_dense_loss_targets_use_patch_grid_without_filling_invalid_depth() -> None:
    segmentation = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ]
        ]
    )
    segmentation_target = _task_loss_targets(
        "segmentation",
        {"segmentation": segmentation},
        (2, 2),
    )["segmentation"]
    assert torch.equal(
        segmentation_target,
        torch.tensor([[[0, 1], [2, 3]]]),
    )

    depth = torch.tensor([[[[1.0, 0.0], [3.0, float("nan")]]]])
    depth_target = _task_loss_targets(
        "depth",
        {"depth": depth},
        (1, 1),
    )["depth"]
    assert depth_target.item() == pytest.approx(2.0)


def _write_full_resolution_dense_cache(
    tmp_path: Path,
    task: str,
) -> Path:
    patches = 16 * 16
    adjacency = torch.eye(patches).unsqueeze(0)
    features = FieldFeatures(
        state=torch.zeros(1, patches, 4),
        response=torch.zeros(1, patches, 4),
        affinity=adjacency,
        adjacency=adjacency,
        grid_size=(16, 16),
    )
    target = (
        torch.ones(1, 512, 512, dtype=torch.long)
        if task == "segmentation"
        else torch.full((1, 1, 512, 512), 2.0)
    )
    cache_dir = tmp_path / f"dense-{task}-cache"
    shard_path = cache_dir / "shard-000000.pt"
    save_features(
        shard_path,
        features,
        targets={task: target},
        sample_ids=[f"{task}-sample"],
        storage_policy="readout_sparse",
    )
    (cache_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": f"synthetic-{task}",
                "split": "validation",
                "num_samples": 1,
                "complete": True,
                "storage_policy": "readout_sparse",
                "shards": [{"path": shard_path.name, "num_samples": 1}],
            }
        ),
        encoding="utf-8",
    )
    return cache_dir


@pytest.mark.parametrize("task", ["segmentation", "depth"])
def test_dense_readout_trains_on_patch_grid_and_scores_full_resolution(
    tmp_path: Path,
    monkeypatch: Any,
    task: str,
) -> None:
    cache_dir = _write_full_resolution_dense_cache(tmp_path, task)
    mapping = _config(tmp_path).to_dict()
    mapping["probe"]["graph_grid"] = [16, 16]
    mapping["tokenizer"]["input_dim"] = 8
    mapping["runtime"]["batch_size"] = 1
    mapping["runtime"]["cache_shard_size"] = 1
    config = RunConfig.from_mapping(mapping)

    loss_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    metric_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_loss = experiments.multitask_loss
    original_update = experiments._update_meter

    def tracked_loss(
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_shapes.append((tuple(predictions[task].shape), tuple(targets[task].shape)))
        return original_loss(predictions, targets, **kwargs)

    def tracked_update(
        meter: Any,
        active_task: str,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        metric_shapes.append(
            (
                tuple(predictions[active_task].shape),
                tuple(targets[active_task].shape),
            )
        )
        original_update(meter, active_task, predictions, targets)

    monkeypatch.setattr(experiments, "multitask_loss", tracked_loss)
    monkeypatch.setattr(experiments, "_update_meter", tracked_update)
    report = train_cached_readout(
        config,
        train_cache_dir=cache_dir,
        val_cache_dir=cache_dir,
        output_dir=tmp_path / f"dense-{task}-readout",
        task=task,
        representation="state",
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=17,
        batch_size=1,
    )

    output_channels = 2 if task == "segmentation" else 1
    target_channels = () if task == "segmentation" else (1,)
    assert loss_shapes
    assert all(
        prediction_shape == (1, output_channels, 16, 16)
        and target_shape == (1, *target_channels, 16, 16)
        for prediction_shape, target_shape in loss_shapes
    )
    expected_metric_shape = (1, 512, 512) if task == "segmentation" else (1, 1, 512, 512)
    assert metric_shapes == [(expected_metric_shape, expected_metric_shape)]
    validation = report["history"][0]["validation"]
    assert validation["num_samples"] == 1
    assert all(torch.isfinite(torch.tensor(list(validation["metrics"].values()))))


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


def _classification_caches(
    tmp_path: Path, monkeypatch: Any, prefix: str
) -> tuple[RunConfig, dict[str, Path]]:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    config = _config(tmp_path)
    caches = {}
    for split in ("train", "val"):
        cache_dir = tmp_path / f"{prefix}-cache-{split}"
        extract_dataset_cache(
            config,
            dataset_name="synthetic-test",
            dataset_root=tmp_path,
            split=split,
            output_dir=cache_dir,
            limit=4,
        )
        caches[split] = cache_dir
    return config, caches


def test_extract_dataset_cache_resumes_verified_shards(tmp_path: Path, monkeypatch: Any) -> None:
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
    assert complete["storage_policy"] == "dense"
    with pytest.raises(ValueError, match="different extraction"):
        extract_dataset_cache(
            _config(tmp_path),
            resume=True,
            storage_policy="readout_sparse",
            **arguments,
        )

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


def test_seed_parallel_matrix_matches_serial_checkpoints(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    mapping = _config(tmp_path).to_dict()
    mapping["tokenizer"]["dropout"] = 0.25
    config = RunConfig.from_mapping(mapping)
    caches = {}
    for split in ("train", "val", "test"):
        cache_dir = tmp_path / f"parallel-cache-{split}"
        extract_dataset_cache(
            config,
            dataset_name="synthetic-test",
            dataset_root=tmp_path,
            split=split,
            output_dir=cache_dir,
            limit=4,
        )
        caches[split] = cache_dir
    common = {
        "train_cache_dir": caches["train"],
        "val_cache_dir": caches["val"],
        "test_cache_dir": caches["test"],
        "task": "classification",
        "representations": ["full"],
        "seeds": [3, 5, 7],
        "epochs": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 2,
        "reference": "full",
    }
    serial_root = tmp_path / "serial-matrix"
    parallel_root = tmp_path / "parallel-matrix"
    serial = run_readout_matrix(config, output_dir=serial_root, **common)
    runtime_profile = {
        "sha256": "runtime-profile",
        "selected_profile": {"seed_workers": 3},
    }
    parallel = run_readout_matrix_seed_parallel(
        config,
        output_dir=parallel_root,
        seed_workers=3,
        readout_runtime_profile=runtime_profile,
        **common,
    )
    assert serial["status"] == parallel["status"] == "passed"
    assert parallel["readout_runtime_profile"] == runtime_profile
    assert not list(parallel_root.glob(".matrix_report.seed-*.json"))
    persisted = json.loads((parallel_root / "matrix_report.json").read_text(encoding="utf-8"))
    assert persisted["readout_runtime_profile"] == runtime_profile
    serial_metrics = {
        (run["representation"], run["seed"]): run["test_metric"] for run in serial["runs"]
    }
    parallel_metrics = {
        (run["representation"], run["seed"]): run["test_metric"] for run in parallel["runs"]
    }
    assert serial_metrics == parallel_metrics
    for seed in common["seeds"]:
        relative = Path("full") / f"seed-{seed}"
        serial_dir = serial_root / relative
        parallel_dir = parallel_root / relative
        for suffix in ("best.pt", "last.pt"):
            serial_checkpoint = torch.load(
                serial_dir / f"classification_full_seed{seed}_{suffix}",
                map_location="cpu",
                weights_only=False,
            )
            parallel_checkpoint = torch.load(
                parallel_dir / f"classification_full_seed{seed}_{suffix}",
                map_location="cpu",
                weights_only=False,
            )
            for key in ("model", "optimizer", "scheduler", "rng_state"):
                _assert_nested_equal(serial_checkpoint[key], parallel_checkpoint[key])
            for key in ("epoch", "best_epoch", "best_primary_metric"):
                assert serial_checkpoint[key] == parallel_checkpoint[key]
            for serial_epoch, parallel_epoch in zip(
                serial_checkpoint["history"],
                parallel_checkpoint["history"],
                strict=True,
            ):
                for key in ("epoch", "learning_rate", "train_loss", "validation"):
                    _assert_nested_equal(serial_epoch[key], parallel_epoch[key])


def test_readout_matrix_rejects_stale_report_revision(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    config = _config(tmp_path)
    caches = {}
    for split in ("train", "val", "test"):
        cache_dir = tmp_path / f"stale-cache-{split}"
        extract_dataset_cache(
            config,
            dataset_name="synthetic-test",
            dataset_root=tmp_path,
            split=split,
            output_dir=cache_dir,
            limit=4,
        )
        caches[split] = cache_dir
    output_dir = tmp_path / "stale-matrix"
    arguments = {
        "train_cache_dir": caches["train"],
        "val_cache_dir": caches["val"],
        "test_cache_dir": caches["test"],
        "output_dir": output_dir,
        "task": "classification",
        "representations": ["state"],
        "seeds": [3],
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 2,
    }
    run_readout_matrix(config, **arguments)
    report_path = output_dir / "state" / "seed-3" / "classification_state_seed3_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["code_revision"] = "stale-revision"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Stale training report revision"):
        run_readout_matrix(config, **arguments)


def _assert_nested_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert isinstance(second, dict)
        assert set(first) == set(second)
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second, strict=True):
            _assert_nested_equal(first_item, second_item)
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray)
        assert np.array_equal(first, second)
    else:
        assert first == second


def test_readout_resume_matches_uninterrupted_training(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    mapping = _config(tmp_path).to_dict()
    mapping["tokenizer"]["dropout"] = 0.25
    config = RunConfig.from_mapping(mapping)
    caches = {}
    for split in ("train", "val"):
        cache_dir = tmp_path / f"resume-cache-{split}"
        extract_dataset_cache(
            config,
            dataset_name="synthetic-test",
            dataset_root=tmp_path,
            split=split,
            output_dir=cache_dir,
            limit=4,
        )
        caches[split] = cache_dir

    common = {
        "config": config,
        "train_cache_dir": caches["train"],
        "val_cache_dir": caches["val"],
        "task": "classification",
        "representation": "full",
        "epochs": 3,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 17,
        "batch_size": 2,
    }
    continuous_dir = tmp_path / "continuous"
    continuous = train_cached_readout(output_dir=continuous_dir, **common)

    import fieldscope.experiments as experiments

    original_save = experiments.atomic_torch_save
    interrupted_dir = tmp_path / "interrupted"

    def interrupt_after_first_epoch(path: Path, payload: dict[str, Any]) -> None:
        original_save(path, payload)
        if path.name.endswith("_last.pt") and int(payload["epoch"]) == 1:
            raise RuntimeError("simulated process loss after epoch commit")

    monkeypatch.setattr(experiments, "atomic_torch_save", interrupt_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        train_cached_readout(output_dir=interrupted_dir, **common)
    interrupted_checkpoint = interrupted_dir / "classification_full_seed17_last.pt"
    assert interrupted_checkpoint.is_file()
    assert (interrupted_dir / "classification_full_seed17_best.pt").is_file()

    monkeypatch.setattr(experiments, "atomic_torch_save", original_save)
    resumed = train_cached_readout(
        output_dir=interrupted_dir,
        resume_checkpoint=interrupted_checkpoint,
        **common,
    )
    continuous_payload = torch.load(
        continuous["last_checkpoint"], map_location="cpu", weights_only=False
    )
    resumed_payload = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "rng_state"):
        _assert_nested_equal(continuous_payload[key], resumed_payload[key])
    assert continuous_payload["best_epoch"] == resumed_payload["best_epoch"]
    assert continuous_payload["best_primary_metric"] == resumed_payload["best_primary_metric"]
    for first, second in zip(continuous["history"], resumed["history"], strict=True):
        assert first["epoch"] == second["epoch"]
        assert first["learning_rate"] == second["learning_rate"]
        assert first["train_loss"] == second["train_loss"]
        assert first["validation"] == second["validation"]


def test_readout_resume_after_final_epoch_commit_rebuilds_passed_report(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    config = _config(tmp_path)
    caches = {}
    for split in ("train", "val"):
        cache_dir = tmp_path / f"final-commit-cache-{split}"
        extract_dataset_cache(
            config,
            dataset_name="synthetic-test",
            dataset_root=tmp_path,
            split=split,
            output_dir=cache_dir,
            limit=4,
        )
        caches[split] = cache_dir

    common = {
        "config": config,
        "train_cache_dir": caches["train"],
        "val_cache_dir": caches["val"],
        "task": "classification",
        "representation": "full",
        "epochs": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 17,
        "batch_size": 2,
    }
    output_dir = tmp_path / "final-commit-interrupted"
    report_path = output_dir / "classification_full_seed17_report.json"
    last_checkpoint = output_dir / "classification_full_seed17_last.pt"
    original_save = experiments.atomic_torch_save

    def interrupt_after_final_epoch_commit(path: Path, payload: dict[str, Any]) -> None:
        original_save(path, payload)
        if path == last_checkpoint and int(payload["epoch"]) == 2:
            raise RuntimeError("simulated process loss after final epoch commit")

    monkeypatch.setattr(
        experiments,
        "atomic_torch_save",
        interrupt_after_final_epoch_commit,
    )
    with pytest.raises(RuntimeError, match="after final epoch commit"):
        train_cached_readout(output_dir=output_dir, **common)
    assert last_checkpoint.is_file()
    stale = json.loads(report_path.read_text(encoding="utf-8"))
    assert stale["status"] == "running"
    assert len(stale["history"]) == 1

    monkeypatch.setattr(experiments, "atomic_torch_save", original_save)
    resumed = train_cached_readout(
        output_dir=output_dir,
        resume_checkpoint=last_checkpoint,
        **common,
    )
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert resumed["status"] == persisted["status"] == "passed"
    assert persisted["history"] == checkpoint["history"]
    assert persisted["best_epoch"] == checkpoint["best_epoch"]
    assert persisted["best_primary_metric"] == checkpoint["best_primary_metric"]
    assert persisted["last_checkpoint"] == str(last_checkpoint)


def test_readout_rejects_nonfinite_training_loss_before_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config, caches = _classification_caches(tmp_path, monkeypatch, "nonfinite-loss")

    def nonfinite_loss(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        loss = torch.tensor(float("nan"), requires_grad=True)
        return loss, {"classification": loss}

    monkeypatch.setattr(experiments, "multitask_loss", nonfinite_loss)
    output_dir = tmp_path / "nonfinite-loss"
    with pytest.raises(ValueError, match="Non-finite training loss"):
        train_cached_readout(
            config,
            train_cache_dir=caches["train"],
            val_cache_dir=caches["val"],
            output_dir=output_dir,
            task="classification",
            representation="full",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=17,
            batch_size=2,
        )
    assert not list(output_dir.glob("*.pt"))


def test_readout_rejects_nonfinite_validation_metric_before_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config, caches = _classification_caches(tmp_path, monkeypatch, "nonfinite-validation")
    monkeypatch.setattr(
        experiments,
        "evaluate_cached_readout",
        lambda *args, **kwargs: {
            "loss": 0.0,
            "num_samples": 4,
            "metrics": {"top1": float("nan")},
        },
    )
    output_dir = tmp_path / "nonfinite-validation"
    with pytest.raises(ValueError, match="Non-finite validation primary metric"):
        train_cached_readout(
            config,
            train_cache_dir=caches["train"],
            val_cache_dir=caches["val"],
            output_dir=output_dir,
            task="classification",
            representation="full",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=17,
            batch_size=2,
        )
    assert not list(output_dir.glob("*.pt"))


def test_readout_rejects_nonfinite_gradient_before_optimizer_step(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config, caches = _classification_caches(tmp_path, monkeypatch, "nonfinite-gradient")

    def nonfinite_gradient_loss(
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        finite_value_with_nan_gradient = torch.sqrt(predictions["classification"].sum() * 0.0)
        return finite_value_with_nan_gradient, {"classification": finite_value_with_nan_gradient}

    monkeypatch.setattr(
        experiments,
        "multitask_loss",
        nonfinite_gradient_loss,
    )
    output_dir = tmp_path / "nonfinite-gradient"
    with pytest.raises(ValueError, match="Non-finite gradient norm"):
        train_cached_readout(
            config,
            train_cache_dir=caches["train"],
            val_cache_dir=caches["val"],
            output_dir=output_dir,
            task="classification",
            representation="full",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=17,
            batch_size=2,
        )
    assert not list(output_dir.glob("*.pt"))


def test_held_out_evaluation_rejects_nonfinite_primary_metric(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config, caches = _classification_caches(tmp_path, monkeypatch, "nonfinite-test")
    report = train_cached_readout(
        config,
        train_cache_dir=caches["train"],
        val_cache_dir=caches["val"],
        output_dir=tmp_path / "nonfinite-test-training",
        task="classification",
        representation="full",
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=17,
        batch_size=2,
    )
    monkeypatch.setattr(
        experiments,
        "evaluate_cached_readout",
        lambda *args, **kwargs: {
            "loss": 0.0,
            "num_samples": 4,
            "metrics": {"top1": float("nan")},
        },
    )
    with pytest.raises(ValueError, match="Non-finite held-out test primary metric"):
        evaluate_checkpoint(
            config,
            checkpoint=Path(report["best_checkpoint"]),
            cache_dir=caches["val"],
            batch_size=2,
        )


def test_sparse_readout_cache_rejects_unsupervised_diagnosis(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "fieldscope.experiments.build_vision_dataset",
        lambda *args, **kwargs: _SmallDataset(),
    )
    output = tmp_path / "sparse-cache"
    report = extract_dataset_cache(
        _config(tmp_path),
        dataset_name="synthetic-test",
        dataset_root=tmp_path,
        split="train",
        output_dir=output,
        limit=4,
        storage_policy="readout_sparse",
    )
    assert report["storage_policy"] == "readout_sparse"
    assert report["dense_affinity_available"] is False
    dataset = CachedFeatureDataset(output)
    assert torch.equal(
        dataset[0]["features"].affinity,
        dataset[0]["features"].adjacency,
    )
    with pytest.raises(ValueError, match="dense-affinity"):
        diagnose_segmentation_cache(output)
