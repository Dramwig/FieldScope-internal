"""Resumable orchestration for the registered supervised error analysis."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fieldscope.error_analysis import (
    EVIDENCE_SCOPE,
    audit_replay_core_compatibility,
    audit_source_repository,
    replay_readout_errors,
)
from fieldscope.error_summary import (
    REGISTERED_REPRESENTATIONS,
    REGISTERED_SEEDS,
    summarize_readout_errors,
)
from fieldscope.experiments import (
    atomic_json_dump,
    cache_identity,
    code_provenance,
    file_sha256,
)

FORMAL_REPRESENTATIONS = {
    "random_feature_local",
    "z0",
    "zt",
    "trajectory",
    "velocity",
    "mismatch",
    "endpoint",
    "state",
    "state_nograph",
    "state_graph",
    "response_nograph",
    "response_local",
    "response",
    "full_nograph",
    "full_local",
    "full",
    "dit_hidden_local",
    "dit_hidden_attention",
    "response_shuffled",
    "full_shuffled",
}


def _read_json_object(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{kind} must be a JSON object: {path}")
    return payload


def _resolve_source_path(value: Any, source_repository_root: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (source_repository_root / path).resolve()


def _formal_runs(matrix: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    if matrix.get("status") != "passed":
        raise ValueError("Formal readout matrix is not passed")
    if set(matrix.get("representations", [])) != FORMAL_REPRESENTATIONS:
        raise ValueError("Formal readout matrix representation contract mismatch")
    if set(matrix.get("seeds", [])) != set(REGISTERED_SEEDS):
        raise ValueError("Formal readout matrix seed contract mismatch")
    runs: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in matrix.get("runs", []):
        if not isinstance(run, Mapping):
            raise ValueError("Formal readout matrix contains an invalid run")
        key = (str(run.get("representation", "")), int(run.get("seed", -1)))
        if key in runs:
            raise ValueError(f"Duplicate formal matrix run: {key[0]}/seed{key[1]}")
        runs[key] = run
    expected = {
        (representation, seed)
        for representation in FORMAL_REPRESENTATIONS
        for seed in REGISTERED_SEEDS
    }
    if set(runs) != expected:
        raise ValueError("Formal readout matrix does not contain exactly 60 registered runs")
    return runs


def _reusable_report(
    path: Path,
    *,
    analyzer: Mapping[str, Any],
    source_revision: str,
    source_repository_root: Path,
    matrix_report_path: Path,
    checkpoint: Path,
    training_report_path: Path,
    test_report_path: Path,
    cache_dir: Path,
    task: str,
    representation: str,
    seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json_object(path, "existing supervised error report")
        checkpoint_payload = payload["checkpoint"]
        source_files = payload["source_files"]
        source_repository = payload["source_repository"]
        replay_compatibility = payload["analyzer"]["replay_core_compatibility"]
        expected_sources = {
            "matrix_report": matrix_report_path,
            "training_report": training_report_path,
            "test_report": test_report_path,
        }
        if (
            payload.get("status") != "passed"
            or payload.get("evidence_scope") != EVIDENCE_SCOPE
            or payload.get("changes_main_verdict") is not False
            or checkpoint_payload.get("task") != task
            or checkpoint_payload.get("representation") != representation
            or int(checkpoint_payload.get("seed", -1)) != seed
            or checkpoint_payload.get("code_revision") != source_revision
            or Path(str(checkpoint_payload.get("path", ""))).resolve() != checkpoint
            or checkpoint_payload.get("sha256") != file_sha256(checkpoint)
            or payload.get("test_cache") != cache_identity(cache_dir)
            or source_repository.get("status") != "passed"
            or Path(str(source_repository.get("path", ""))).resolve()
            != source_repository_root
            or source_repository.get("source_revision") != source_revision
            or source_repository.get("actual_revision") != source_revision
            or source_repository.get("dirty_paths") != []
            or payload.get("analyzer", {}).get("code_revision")
            != analyzer.get("code_revision")
            or payload.get("analyzer", {}).get("code_tree_sha256")
            != analyzer.get("code_tree_sha256")
            or payload.get("analyzer", {}).get("code_dirty") is not False
            or replay_compatibility.get("status") != "passed"
            or replay_compatibility.get("source_revision") != source_revision
            or replay_compatibility.get("analyzer_revision")
            != analyzer.get("code_revision")
        ):
            return False
        for kind, source_path in expected_sources.items():
            recorded = source_files[kind]
            if (
                Path(str(recorded.get("path", ""))).resolve() != source_path
                or recorded.get("sha256") != file_sha256(source_path)
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def run_readout_error_analysis(
    *,
    matrix_report_path: Path,
    source_repository_root: Path,
    output_dir: Path,
    batch_size: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Replay and summarize the registered 33-report subset of one formal matrix."""

    started = time.perf_counter()
    matrix_report_path = matrix_report_path.resolve()
    source_repository_root = source_repository_root.resolve()
    output_dir = output_dir.resolve()
    analyzer = code_provenance()
    if analyzer.get("code_dirty") is not False:
        raise ValueError("Supervised error pipeline requires a clean analyzer worktree")
    matrix = _read_json_object(matrix_report_path, "formal matrix report")
    runs = _formal_runs(matrix)
    task = str(matrix.get("task", ""))
    if task not in {"classification", "segmentation", "depth"}:
        raise ValueError(f"Unsupported formal error-analysis task: {task}")
    source_revision = str(matrix.get("code_revision", ""))
    source_tree_sha256 = str(matrix.get("code_tree_sha256", ""))
    if not source_revision or not source_tree_sha256:
        raise ValueError("Formal matrix source provenance is missing")
    source_repository = audit_source_repository(
        source_revision,
        repository_root=source_repository_root,
    )
    compatibility = audit_replay_core_compatibility(
        source_revision,
        expected_source_tree_sha256=source_tree_sha256,
    )
    cache_dir = _resolve_source_path(
        matrix.get("test_cache_dir", ""), source_repository_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    generated = []
    reused = []
    for representation in REGISTERED_REPRESENTATIONS:
        for seed in REGISTERED_SEEDS:
            run = runs[(representation, seed)]
            checkpoint = _resolve_source_path(
                run.get("best_checkpoint", ""), source_repository_root
            )
            training_report_path = _resolve_source_path(
                run.get("training_report", ""), source_repository_root
            )
            test_report_path = _resolve_source_path(
                run.get("test_report", ""), source_repository_root
            )
            report_path = output_dir / "per_run" / representation / f"seed-{seed}.json"
            report_paths.append(report_path)
            identity = {"representation": representation, "seed": seed}
            if _reusable_report(
                report_path,
                analyzer=analyzer,
                source_revision=source_revision,
                source_repository_root=source_repository_root,
                matrix_report_path=matrix_report_path,
                checkpoint=checkpoint,
                training_report_path=training_report_path,
                test_report_path=test_report_path,
                cache_dir=cache_dir,
                task=task,
                representation=representation,
                seed=seed,
            ):
                reused.append(identity)
                continue
            replay_command = [
                "fieldscope",
                "analyze-readout-errors",
                "--checkpoint",
                str(checkpoint),
                "--cache-dir",
                str(cache_dir),
                "--matrix-report",
                str(matrix_report_path),
                "--training-report",
                str(training_report_path),
                "--test-report",
                str(test_report_path),
                "--source-repository-root",
                str(source_repository_root),
                "--output",
                str(report_path),
            ]
            if batch_size is not None:
                replay_command.extend(("--batch-size", str(batch_size)))
            report = replay_readout_errors(
                checkpoint=checkpoint,
                cache_dir=cache_dir,
                matrix_report_path=matrix_report_path,
                training_report_path=training_report_path,
                test_report_path=test_report_path,
                source_repository_root=source_repository_root,
                batch_size=batch_size,
                command=replay_command,
            )
            atomic_json_dump(report_path, report)
            generated.append(identity)
    summary = summarize_readout_errors(report_paths, command=command)
    summary["pipeline"] = {
        "status": "passed",
        "matrix_report": {
            "path": str(matrix_report_path),
            "sha256": file_sha256(matrix_report_path),
        },
        "source_repository": source_repository,
        "replay_core_compatibility": compatibility,
        "output_dir": str(output_dir),
        "generated_reports": generated,
        "reused_reports": reused,
        "generated_count": len(generated),
        "reused_count": len(reused),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    return summary
