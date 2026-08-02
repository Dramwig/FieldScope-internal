"""Fail-closed per-sample replay for the registered supervised error analysis."""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import tarfile
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fieldscope.cached_dataset import cached_control_contract, collate_cached
from fieldscope.config import RunConfig
from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.evaluation import ClassificationMeter, DepthMeter, SegmentationMeter
from fieldscope.experiments import (
    _cached_dataset,
    _readout_features_to_device,
    _task_output_size,
    cache_identity,
    code_provenance,
    file_sha256,
)
from fieldscope.model import FieldScopeModel

SCHEMA_VERSION = 1
EVIDENCE_SCOPE = "prospective_secondary_supervised_error_analysis"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPLAY_CORE_PATHS = (
    "src/fieldscope/cache.py",
    "src/fieldscope/cached_dataset.py",
    "src/fieldscope/config.py",
    "src/fieldscope/contracts.py",
    "src/fieldscope/evaluation.py",
    "src/fieldscope/experiments.py",
    "src/fieldscope/feature_ops.py",
    "src/fieldscope/graph.py",
    "src/fieldscope/heads.py",
    "src/fieldscope/losses.py",
    "src/fieldscope/model.py",
    "src/fieldscope/tokenizer.py",
)


def _run_git(repository_root: Path, arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Unable to audit replay revisions: {error}") from error
    return result.stdout.strip()


def _source_code_tree_sha256(repository_root: Path, revision: str) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "archive",
                "--format=tar",
                revision,
                "src/fieldscope",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Unable to hash checkpoint source revision: {error}") from error
    digest = hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".py")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Unable to read archived source file: {member.name}")
            digest.update(member.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(handle.read())
            digest.update(b"\0")
    return digest.hexdigest()


def audit_replay_core_compatibility(
    source_revision: str,
    *,
    expected_source_tree_sha256: str | None = None,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Prove that cached-readout inference code is unchanged since a checkpoint."""

    _run_git(repository_root, ["cat-file", "-e", f"{source_revision}^{{commit}}"])
    analyzer_revision = _run_git(repository_root, ["rev-parse", "HEAD"])
    changed_output = _run_git(
        repository_root,
        ["diff", "--name-only", source_revision, analyzer_revision, "--", *_REPLAY_CORE_PATHS],
    )
    dirty_output = _run_git(
        repository_root,
        ["status", "--porcelain", "--", *_REPLAY_CORE_PATHS],
    )
    changed_paths = [line for line in changed_output.splitlines() if line]
    dirty_paths = [line for line in dirty_output.splitlines() if line]
    if changed_paths:
        raise ValueError(
            "Replay core differs from checkpoint revision: " + ", ".join(changed_paths)
        )
    if dirty_paths:
        raise ValueError("Replay core has uncommitted changes: " + ", ".join(dirty_paths))
    source_tree_sha256 = _source_code_tree_sha256(repository_root, source_revision)
    if (
        expected_source_tree_sha256 is not None
        and source_tree_sha256 != expected_source_tree_sha256
    ):
        raise ValueError("Checkpoint source code-tree SHA-256 does not match its revision")
    return {
        "status": "passed",
        "source_revision": source_revision,
        "analyzer_revision": analyzer_revision,
        "source_code_tree_sha256": source_tree_sha256,
        "core_paths": list(_REPLAY_CORE_PATHS),
        "changed_paths": [],
        "dirty_paths": [],
    }


def audit_source_repository(
    source_revision: str,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Bind relative artifact paths to a clean checkout of the formal revision."""

    repository_root = repository_root.resolve()
    actual_revision = _run_git(repository_root, ["rev-parse", "HEAD"])
    if actual_revision != source_revision:
        raise ValueError(
            "Source repository revision mismatch: "
            f"expected={source_revision} actual={actual_revision}"
        )
    dirty_output = _run_git(repository_root, ["status", "--porcelain"])
    dirty_paths = [line for line in dirty_output.splitlines() if line]
    if dirty_paths:
        raise ValueError("Source repository worktree is dirty")
    return {
        "status": "passed",
        "path": str(repository_root),
        "source_revision": source_revision,
        "actual_revision": actual_revision,
        "dirty_paths": [],
    }


def _read_json_object(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{kind} must be a JSON object: {path}")
    return payload


def _resolve_reported_path(value: Any, repository_root: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _require_report_source(
    payload: Mapping[str, Any],
    *,
    kind: str,
    source_revision: str,
    source_tree_sha256: str,
) -> None:
    _require_equal(payload.get("status"), "passed", f"{kind} is not passed")
    _require_equal(
        payload.get("code_revision"), source_revision, f"{kind} revision mismatch"
    )
    _require_equal(
        payload.get("code_tree_sha256"),
        source_tree_sha256,
        f"{kind} code tree mismatch",
    )
    _require_equal(payload.get("code_dirty"), False, f"{kind} came from a dirty worktree")


def _validate_source_artifacts(
    *,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
    cache_dir: Path,
    matrix_report_path: Path,
    training_report_path: Path,
    test_report_path: Path,
    source_repository_root: Path,
) -> dict[str, Any]:
    matrix = _read_json_object(matrix_report_path, "matrix report")
    training = _read_json_object(training_report_path, "training report")
    test = _read_json_object(test_report_path, "test report")
    source_revision = str(checkpoint_payload.get("code_revision", ""))
    source_tree_sha256 = str(checkpoint_payload.get("code_tree_sha256", ""))
    if not source_revision or not source_tree_sha256:
        raise ValueError("Checkpoint source provenance is missing")
    _require_equal(
        checkpoint_payload.get("code_dirty"), False, "Checkpoint came from a dirty worktree"
    )
    for kind, payload in (
        ("matrix report", matrix),
        ("training report", training),
        ("test report", test),
    ):
        _require_report_source(
            payload,
            kind=kind,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
        )

    task = str(checkpoint_payload.get("task", ""))
    representation = str(checkpoint_payload.get("representation", ""))
    seed = int(checkpoint_payload.get("seed", -1))
    config = checkpoint_payload.get("config")
    report_config = _json_compatible(config)
    for kind, payload in (("training report", training), ("test report", test)):
        _require_equal(payload.get("task"), task, f"{kind} task mismatch")
        _require_equal(
            payload.get("representation"), representation, f"{kind} representation mismatch"
        )
        _require_equal(int(payload.get("seed", -1)), seed, f"{kind} seed mismatch")
        _require_equal(payload.get("config"), report_config, f"{kind} config mismatch")
    _require_equal(matrix.get("task"), task, "Matrix task mismatch")
    _require_equal(matrix.get("config"), report_config, "Matrix config mismatch")

    matching_runs = [
        run
        for run in matrix.get("runs", [])
        if str(run.get("representation")) == representation and int(run.get("seed", -1)) == seed
    ]
    if len(matching_runs) != 1:
        raise ValueError("Matrix does not contain exactly one matching representation/seed run")
    run = matching_runs[0]
    checkpoint = checkpoint.resolve()
    checkpoint_sha256 = file_sha256(checkpoint)
    for kind, reported in (
        ("matrix", run.get("best_checkpoint")),
        ("training", training.get("best_checkpoint")),
        ("test", test.get("checkpoint")),
    ):
        _require_equal(
            _resolve_reported_path(reported, source_repository_root),
            checkpoint,
            f"{kind} checkpoint path mismatch",
        )
    for kind, reported_sha256 in (
        ("matrix", run.get("best_checkpoint_sha256")),
        ("training", training.get("best_checkpoint_sha256")),
        ("test", test.get("checkpoint_sha256")),
    ):
        _require_equal(reported_sha256, checkpoint_sha256, f"{kind} checkpoint SHA-256 mismatch")
    for kind, reported_path, expected_path in (
        ("matrix training", run.get("training_report"), training_report_path),
        ("matrix test", run.get("test_report"), test_report_path),
    ):
        _require_equal(
            _resolve_reported_path(reported_path, source_repository_root),
            expected_path.resolve(),
            f"{kind} report path mismatch",
        )

    test_cache = cache_identity(cache_dir)
    _require_equal(
        test_cache.get("code_revision"), source_revision, "Test cache revision mismatch"
    )
    _require_equal(
        test_cache.get("code_tree_sha256"),
        source_tree_sha256,
        "Test cache code tree mismatch",
    )
    _require_equal(matrix.get("test_cache"), test_cache, "Matrix test-cache identity mismatch")
    _require_equal(test.get("test_cache"), test_cache, "Test report cache identity mismatch")
    return {
        "matrix": matrix,
        "training": training,
        "test": test,
        "test_cache": test_cache,
        "checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "source_files": {
            "matrix_report": {
                "path": str(matrix_report_path.resolve()),
                "sha256": file_sha256(matrix_report_path),
            },
            "training_report": {
                "path": str(training_report_path.resolve()),
                "sha256": file_sha256(training_report_path),
            },
            "test_report": {
                "path": str(test_report_path.resolve()),
                "sha256": file_sha256(test_report_path),
            },
        },
    }


def _classification_records(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    logits = logits.detach().float().cpu()
    targets = targets.detach().long().cpu()
    probabilities = torch.softmax(logits, dim=1)
    topk = min(5, logits.shape[1])
    top_indices = logits.topk(topk, dim=1).indices
    top_values = logits.topk(min(2, logits.shape[1]), dim=1).values
    records = []
    for index, sample_id in enumerate(sample_ids):
        target = int(targets[index].item())
        true_probability = float(probabilities[index, target].item())
        margin = (
            float((top_values[index, 0] - top_values[index, 1]).item())
            if top_values.shape[1] > 1
            else 0.0
        )
        records.append(
            {
                "sample_id": sample_id,
                "target_class": target,
                "predicted_class": int(top_indices[index, 0].item()),
                "top1_correct": int(top_indices[index, 0].item() == target),
                "top5_correct": int(target in top_indices[index].tolist()),
                "true_class_probability": true_probability,
                "negative_log_likelihood": -math.log(max(true_probability, 1e-30)),
                "top1_logit_margin": margin,
            }
        )
    return records


def _segmentation_record(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_id: str,
    num_classes: int,
) -> dict[str, Any]:
    prediction = prediction.detach().long().cpu()
    target = target.detach().long().cpu()
    valid = (target != 255) & (target >= 0) & (target < num_classes)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError(f"Segmentation sample has no valid pixels: {sample_id}")
    intersections = []
    unions = []
    for label in range(num_classes):
        predicted = (prediction == label) & valid
        expected = (target == label) & valid
        union = int((predicted | expected).sum().item())
        if union > 0:
            intersections.append(int((predicted & expected).sum().item()))
            unions.append(union)
    mean_iou = sum(i / u for i, u in zip(intersections, unions, strict=True)) / len(unions)
    labels, counts = torch.unique(target[valid], return_counts=True)

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    adjacent_count = int(horizontal_valid.sum().item() + vertical_valid.sum().item())
    boundary_count = int(
        ((target[:, :-1] != target[:, 1:]) & horizontal_valid).sum().item()
        + ((target[:-1, :] != target[1:, :]) & vertical_valid).sum().item()
    )
    return {
        "sample_id": sample_id,
        "mean_iou": mean_iou,
        "pixel_accuracy": float(((prediction == target) & valid).sum().item() / valid_count),
        "valid_fraction": float(valid_count / target.numel()),
        "semantic_class_count": int(labels.numel()),
        "dominant_class_fraction": float(counts.max().item() / valid_count),
        "boundary_density": float(boundary_count / max(1, adjacent_count)),
    }


def _depth_record(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_id: str,
) -> dict[str, Any]:
    prediction = prediction.detach().float().cpu().squeeze(0)
    target = target.detach().float().cpu().squeeze(0)
    valid = torch.isfinite(target) & (target > 0)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError(f"Depth sample has no valid pixels: {sample_id}")
    predicted = prediction[valid].clamp_min(1e-6)
    expected = target[valid].clamp_min(1e-6)
    difference = predicted - expected
    ratio = torch.maximum(predicted / expected, expected / predicted)
    quantiles = torch.quantile(expected, torch.tensor([0.05, 0.95]))

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    horizontal_denominator = torch.minimum(target[:, :-1], target[:, 1:]).clamp_min(1e-6)
    vertical_denominator = torch.minimum(target[:-1, :], target[1:, :]).clamp_min(1e-6)
    horizontal_jump = (
        (target[:, :-1] - target[:, 1:]).abs() / horizontal_denominator > 0.1
    ) & horizontal_valid
    vertical_jump = (
        (target[:-1, :] - target[1:, :]).abs() / vertical_denominator > 0.1
    ) & vertical_valid
    adjacent_count = int(horizontal_valid.sum().item() + vertical_valid.sum().item())
    jump_count = int(horizontal_jump.sum().item() + vertical_jump.sum().item())
    return {
        "sample_id": sample_id,
        "abs_rel": float((difference.abs() / expected).mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "delta1": float((ratio < 1.25).float().mean().item()),
        "valid_fraction": float(valid_count / target.numel()),
        "depth_p05_p95_range": float((quantiles[1] - quantiles[0]).item()),
        "depth_discontinuity_density": float(jump_count / max(1, adjacent_count)),
    }


def _require_finite_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        for name, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite per-sample value {record.get('sample_id')}/{name}")


def _compare_replayed_metrics(
    replayed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for name, expected_value in expected.items():
        if not isinstance(expected_value, (int, float)) or name not in replayed:
            continue
        actual_value = float(replayed[name])
        expected_value = float(expected_value)
        if not math.isfinite(actual_value) or not math.isfinite(expected_value):
            raise ValueError(f"Non-finite replayed aggregate metric: {name}")
        if not math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"Replayed aggregate metric mismatch {name}: "
                f"expected={expected_value} actual={actual_value}"
            )


@torch.no_grad()
def replay_readout_errors(
    *,
    checkpoint: Path,
    cache_dir: Path,
    matrix_report_path: Path,
    training_report_path: Path,
    test_report_path: Path,
    source_repository_root: Path | None = None,
    batch_size: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Replay one formal test checkpoint and emit aligned per-sample diagnostics."""

    started = time.perf_counter()
    checkpoint = checkpoint.resolve()
    cache_dir = cache_dir.resolve()
    source_repository_root = (
        source_repository_root or _REPOSITORY_ROOT
    ).resolve()
    analyzer = code_provenance()
    if analyzer.get("code_dirty") is not False:
        raise ValueError("Supervised error analysis requires a clean analyzer worktree")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    task = str(payload.get("task", ""))
    if task not in {"classification", "segmentation", "depth"}:
        raise ValueError(f"Unsupported registered error-analysis task: {task}")
    artifacts = _validate_source_artifacts(
        checkpoint=checkpoint,
        checkpoint_payload=payload,
        cache_dir=cache_dir,
        matrix_report_path=matrix_report_path,
        training_report_path=training_report_path,
        test_report_path=test_report_path,
        source_repository_root=source_repository_root,
    )
    source_repository = audit_source_repository(
        artifacts["source_revision"],
        repository_root=source_repository_root,
    )
    compatibility = audit_replay_core_compatibility(
        artifacts["source_revision"],
        expected_source_tree_sha256=artifacts["source_tree_sha256"],
    )

    checkpoint_config = RunConfig.from_mapping(payload["config"])
    device = torch.device(checkpoint_config.backend.device)
    model = FieldScopeModel(
        int(payload["state_dim"]),
        int(payload["response_dim"]),
        checkpoint_config.tokenizer,
        mode=str(payload["mode"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    representation = str(payload["representation"])
    seed = int(payload["seed"])
    dataset = _cached_dataset(
        cache_dir,
        representation,
        seed,
        checkpoint_config.runtime.readout_memory_cache_gib,
    )
    control_contract = cached_control_contract(dataset)
    _require_equal(
        artifacts["test"].get("test_control_contract"),
        control_contract,
        "Test control contract mismatch",
    )
    effective_batch_size = batch_size or int(artifacts["test"].get("batch_size", 0))
    if effective_batch_size < 1:
        raise ValueError("Replay batch size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_cached, representation=representation),
    )
    if task == "classification":
        meter: Any = ClassificationMeter()
    elif task == "segmentation":
        meter = SegmentationMeter(checkpoint_config.tokenizer.segmentation_classes)
    else:
        meter = DepthMeter()
    records: list[dict[str, Any]] = []
    for batch in loader:
        mode = str(batch["tokenizer_mode"])
        features = _readout_features_to_device(
            batch["features"],
            mode,
            device,
            validate_model_features=False,
        )
        targets = {name: value.to(device) for name, value in batch["targets"].items()}
        if task not in targets:
            raise ValueError(f"Requested task {task!r} is absent from cache")
        predictions = model(
            features,
            output_size=_task_output_size(task, targets, features.grid_size),
            task=task,
            validate_features=False,
        )[task]
        sample_ids = [str(value) for value in batch["sample_ids"]]
        if task == "classification":
            meter.update(predictions, targets[task])
            records.extend(_classification_records(predictions, targets[task], sample_ids))
            continue
        if tuple(predictions.shape[-2:]) != tuple(targets[task].shape[-2:]):
            if task == "segmentation":
                predictions = torch.nn.functional.interpolate(
                    predictions.argmax(dim=1, keepdim=True).float(),
                    size=targets[task].shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
            else:
                predictions = torch.nn.functional.interpolate(
                    predictions,
                    size=targets[task].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        if task == "segmentation":
            meter.update(predictions, targets[task])
            records.extend(
                _segmentation_record(
                    predictions[index],
                    targets[task][index],
                    sample_id=sample_id,
                    num_classes=checkpoint_config.tokenizer.segmentation_classes,
                )
                for index, sample_id in enumerate(sample_ids)
            )
        else:
            meter.update(predictions, targets[task])
            records.extend(
                _depth_record(
                    predictions[index],
                    targets[task][index],
                    sample_id=sample_id,
                )
                for index, sample_id in enumerate(sample_ids)
            )

    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Replay produced duplicate sample IDs")
    expected_samples = int(artifacts["test_cache"].get("num_samples", -1))
    _require_equal(len(records), expected_samples, "Replay sample count mismatch")
    replay_sample_ids_sha256 = sample_ids_sha256(sample_ids)
    _require_equal(
        replay_sample_ids_sha256,
        artifacts["test_cache"].get("sample_ids_sha256"),
        "Replay sample-ID SHA-256 mismatch",
    )
    _require_finite_records(records)
    replayed_metrics = meter.compute()
    expected_metrics = artifacts["test"].get("evaluation", {}).get("metrics", {})
    if not isinstance(expected_metrics, Mapping):
        raise ValueError("Test report aggregate metrics are missing")
    _compare_replayed_metrics(replayed_metrics, expected_metrics)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "evidence_scope": EVIDENCE_SCOPE,
        "changes_main_verdict": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": artifacts["checkpoint_sha256"],
            "code_revision": artifacts["source_revision"],
            "code_tree_sha256": artifacts["source_tree_sha256"],
            "task": task,
            "representation": representation,
            "seed": seed,
            "mode": payload["mode"],
            "state_dim": int(payload["state_dim"]),
            "response_dim": int(payload["response_dim"]),
            "config": payload["config"],
        },
        "source_files": artifacts["source_files"],
        "source_repository": source_repository,
        "test_cache": artifacts["test_cache"],
        "test_control_contract": control_contract,
        "analyzer": {
            **analyzer,
            "command": list(command or []),
            "batch_size": effective_batch_size,
            "replay_core_compatibility": compatibility,
        },
        "num_samples": len(records),
        "sample_ids_sha256": replay_sample_ids_sha256,
        "replayed_aggregate_metrics": replayed_metrics,
        "per_sample": records,
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
        "research_boundary": (
            "This prospective secondary analysis explains completed formal results; "
            "it does not alter the registered main or final verdict."
        ),
    }
