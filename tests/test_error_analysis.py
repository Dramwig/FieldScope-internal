import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

import fieldscope.error_analysis as error_analysis
from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.cache import save_features
from fieldscope.config import ProbeConfig, RunConfig
from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.error_analysis import (
    _classification_records,
    _depth_record,
    _segmentation_record,
    _source_code_tree_sha256,
    audit_replay_core_compatibility,
    audit_source_repository,
    replay_readout_errors,
)
from fieldscope.experiments import (
    cache_identity,
    evaluate_checkpoint,
    file_sha256,
    train_cached_readout,
)
from fieldscope.feature_ops import slice_features, stack_features
from fieldscope.response import FieldResponseExtractor


def _head_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_registered_per_sample_metrics() -> None:
    classification = _classification_records(
        torch.tensor([[3.0, 1.0], [0.0, 2.0]]),
        torch.tensor([0, 0]),
        ["a", "b"],
    )
    assert classification[0]["top1_correct"] == 1
    assert classification[1]["top1_correct"] == 0
    assert classification[0]["top1_logit_margin"] == pytest.approx(2.0)
    assert classification[0]["negative_log_likelihood"] > 0

    segmentation = _segmentation_record(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[0, 0], [1, 255]]),
        sample_id="seg",
        num_classes=2,
    )
    assert segmentation["mean_iou"] == pytest.approx(0.5)
    assert segmentation["pixel_accuracy"] == pytest.approx(2 / 3)
    assert segmentation["valid_fraction"] == pytest.approx(0.75)
    assert segmentation["semantic_class_count"] == 2
    assert segmentation["dominant_class_fraction"] == pytest.approx(2 / 3)
    assert segmentation["boundary_density"] == pytest.approx(0.5)

    depth = _depth_record(
        torch.tensor([[[1.0, 2.0], [2.0, 3.0]]]),
        torch.tensor([[[1.0, 1.0], [2.0, float("nan")]]]),
        sample_id="depth",
    )
    assert depth["abs_rel"] == pytest.approx(1 / 3)
    assert depth["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert depth["delta1"] == pytest.approx(2 / 3)
    assert depth["valid_fraction"] == pytest.approx(0.75)
    assert depth["depth_p05_p95_range"] == pytest.approx(0.9)
    assert depth["depth_discontinuity_density"] == pytest.approx(0.5)


def test_replay_core_compatibility_accepts_same_revision() -> None:
    repository = Path(__file__).resolve().parents[1]
    revision = _head_revision(repository)
    expected_tree = _source_code_tree_sha256(repository, revision)
    report = audit_replay_core_compatibility(
        revision,
        expected_source_tree_sha256=expected_tree,
        repository_root=repository,
    )
    assert report["status"] == "passed"
    assert report["changed_paths"] == []
    assert report["dirty_paths"] == []
    assert report["source_code_tree_sha256"] == expected_tree


def test_replay_core_compatibility_rejects_changed_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(["", "analyzer", "src/fieldscope/model.py", ""])
    monkeypatch.setattr(error_analysis, "_run_git", lambda *_args, **_kwargs: next(outputs))
    with pytest.raises(ValueError, match="Replay core differs"):
        audit_replay_core_compatibility("source")


def test_source_repository_audit_binds_clean_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = iter(["formal-revision", ""])
    monkeypatch.setattr(error_analysis, "_run_git", lambda *_args, **_kwargs: next(outputs))
    report = audit_source_repository(
        "formal-revision",
        repository_root=tmp_path,
    )
    assert report["status"] == "passed"
    assert report["path"] == str(tmp_path.resolve())
    assert report["dirty_paths"] == []


def test_source_repository_audit_rejects_wrong_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(error_analysis, "_run_git", lambda *_args, **_kwargs: "other")
    with pytest.raises(ValueError, match="Source repository revision mismatch"):
        audit_source_repository("formal-revision", repository_root=tmp_path)


def _write_classification_cache(
    tmp_path: Path,
    *,
    source_revision: str,
    source_tree_sha256: str,
) -> tuple[Path, RunConfig]:
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
    cache_dir.mkdir()
    sample_ids = ["a", "b", "c"]
    labels = torch.tensor([0, 1, 0])
    shard_path = cache_dir / "shard-000000.pt"
    save_features(
        shard_path,
        stack_features([slice_features(features, index) for index in range(3)]),
        targets={"classification": labels},
        sample_ids=sample_ids,
    )
    (cache_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "passed",
                "complete": True,
                "dataset": "synthetic",
                "split": "test",
                "num_samples": 3,
                "sample_ids_sha256": sample_ids_sha256(sample_ids),
                "storage_policy": "dense",
                "extraction_signature": "test",
                "code_revision": source_revision,
                "code_tree_sha256": source_tree_sha256,
                "shards": [{"path": shard_path.name, "num_samples": 3}],
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
                "topk": 3,
                "probe_batch_size": 4,
                "antithetic_noise": False,
            },
            "tokenizer": {
                "hidden_dim": 16,
                "input_dim": 16,
                "num_layers": 1,
                "num_classes": 2,
                "segmentation_classes": 2,
            },
            "runtime": {
                "output_dir": str(tmp_path / "output"),
                "cache_dir": str(cache_dir),
                "batch_size": 2,
                "num_workers": 0,
                "deterministic": True,
            },
        }
    )
    return cache_dir, config


def test_replay_readout_errors_binds_artifacts_and_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_revision = _head_revision(repository)
    source_tree = _source_code_tree_sha256(repository, source_revision)
    cache_dir, config = _write_classification_cache(
        tmp_path,
        source_revision=source_revision,
        source_tree_sha256=source_tree,
    )
    output_dir = tmp_path / "training"
    training = train_cached_readout(
        config,
        train_cache_dir=cache_dir,
        val_cache_dir=cache_dir,
        output_dir=output_dir,
        task="classification",
        representation="full",
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=7,
        batch_size=2,
    )
    checkpoint = Path(training["best_checkpoint"])
    test = evaluate_checkpoint(config, checkpoint=checkpoint, cache_dir=cache_dir, batch_size=2)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_payload.update(
        {
            "code_revision": source_revision,
            "code_tree_sha256": source_tree,
            "code_dirty": False,
        }
    )
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)

    training_path = output_dir / "classification_full_seed7_report.json"
    training_payload = json.loads(training_path.read_text(encoding="utf-8"))
    training_payload.update(
        {
            "code_revision": source_revision,
            "code_tree_sha256": source_tree,
            "code_dirty": False,
            "best_checkpoint_sha256": checkpoint_sha256,
        }
    )
    training_path.write_text(json.dumps(training_payload), encoding="utf-8")
    test.update(
        {
            "code_revision": source_revision,
            "code_tree_sha256": source_tree,
            "code_dirty": False,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps(test), encoding="utf-8")
    matrix = {
        "status": "passed",
        "code_revision": source_revision,
        "code_tree_sha256": source_tree,
        "code_dirty": False,
        "task": "classification",
        "config": config.to_dict(),
        "test_cache": cache_identity(cache_dir),
        "runs": [
            {
                "representation": "full",
                "seed": 7,
                "best_checkpoint": str(checkpoint),
                "best_checkpoint_sha256": checkpoint_sha256,
                "training_report": str(training_path),
                "test_report": str(test_path),
            }
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    monkeypatch.setattr(
        error_analysis,
        "code_provenance",
        lambda: {
            "code_revision": source_revision,
            "code_tree_sha256": "analyzer-tree",
            "code_dirty": False,
        },
    )
    monkeypatch.setattr(
        error_analysis,
        "audit_source_repository",
        lambda revision, *, repository_root: {
            "status": "passed",
            "path": str(repository_root),
            "source_revision": revision,
            "actual_revision": revision,
            "dirty_paths": [],
        },
    )
    report = replay_readout_errors(
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        matrix_report_path=matrix_path,
        training_report_path=training_path,
        test_report_path=test_path,
        batch_size=2,
        command=["fieldscope", "analyze-readout-errors"],
    )
    assert report["status"] == "passed"
    assert report["changes_main_verdict"] is False
    assert report["num_samples"] == 3
    assert len(report["per_sample"]) == 3
    assert report["sample_ids_sha256"] == cache_identity(cache_dir)["sample_ids_sha256"]
    assert report["checkpoint"]["sha256"] == checkpoint_sha256
    assert report["source_repository"]["status"] == "passed"
    assert report["replayed_aggregate_metrics"] == test["evaluation"]["metrics"]

    tampered = json.loads(test_path.read_text(encoding="utf-8"))
    tampered["checkpoint_sha256"] = "0" * 64
    test_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="test checkpoint SHA-256 mismatch"):
        replay_readout_errors(
            checkpoint=checkpoint,
            cache_dir=cache_dir,
            matrix_report_path=matrix_path,
            training_report_path=training_path,
            test_report_path=test_path,
            batch_size=2,
        )
