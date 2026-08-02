"""Audit that every registered formal and secondary artifact is complete."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fieldscope.error_analysis import audit_source_repository
from fieldscope.experiments import code_provenance, file_sha256

_TASKS = ("imagenet100", "voc2012", "ade20k", "nyuv2")
_TASK_TYPES = {
    "imagenet100": "classification",
    "voc2012": "segmentation",
    "ade20k": "segmentation",
    "nyuv2": "depth",
}
_CAUSAL_SOURCES = {
    "empty_prompt",
    "random_flow",
    "spatially_shuffled_probe",
    "neutral_prompt",
    "unrelated_prompt",
}
_FINAL_VERDICTS = {
    "supports_core_hypothesis_with_scaling_extension",
    "supports_core_hypothesis_limited_scaling",
    "main_task_gain_not_causally_attributed",
    "limited_or_negative",
}
_POSITIVE_MAIN_VERDICT = "main_tasks_supported_pending_causal_audits"


def _read_json(path: Path, kind: str, problems: list[str]) -> dict[str, Any]:
    if not path.is_file():
        problems.append(f"missing {kind}: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"invalid {kind}: {path}: {error}")
        return {}
    if not isinstance(payload, dict):
        problems.append(f"{kind} must be a JSON object: {path}")
        return {}
    return payload


def _resolve_source_path(value: Any, source_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def _source_tree_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    package_root = source_root / "src" / "fieldscope"
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _artifact(path: Path | None, kind: str, problems: list[str]) -> dict[str, Any]:
    if path is None:
        problems.append(f"missing {kind} path")
        return {"path": None, "sha256": None}
    if not path.is_file():
        problems.append(f"missing {kind}: {path}")
        return {"path": str(path), "sha256": None}
    return {"path": str(path), "sha256": file_sha256(path)}


def _require_provenance(
    payload: Mapping[str, Any],
    *,
    kind: str,
    revision: str,
    tree_sha256: str,
    problems: list[str],
) -> None:
    if payload.get("code_revision") != revision:
        problems.append(f"{kind} revision mismatch")
    if payload.get("code_tree_sha256") != tree_sha256:
        problems.append(f"{kind} code-tree mismatch")
    if payload.get("code_dirty") is not False:
        problems.append(f"{kind} came from a dirty worktree")


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _audit_training_execution(
    report: Mapping[str, Any], problems: list[str]
) -> None:
    required_steps = _positive_integer(report.get("required_optimizer_steps"))
    reported_steps = _positive_integer(report.get("reported_completed_optimizer_steps"))
    if required_steps is None or reported_steps != required_steps:
        problems.append("formal optimizer-step execution is incomplete")
    required_exposures = _positive_integer(
        report.get("required_training_sample_exposures")
    )
    reported_exposures = _positive_integer(
        report.get("reported_completed_training_sample_exposures")
    )
    if required_exposures is None or reported_exposures != required_exposures:
        problems.append("formal training sample exposure is incomplete")


def _audit_main(
    main: Mapping[str, Any],
    *,
    final: Mapping[str, Any],
    revision: str,
    tree_sha256: str,
    problems: list[str],
) -> None:
    _require_provenance(
        main,
        kind="main evidence",
        revision=revision,
        tree_sha256=tree_sha256,
        problems=problems,
    )
    if main.get("verdict") not in {_POSITIVE_MAIN_VERDICT, "limited_or_negative"}:
        problems.append("main evidence verdict is incomplete")
    if main.get("verdict") != final.get("main_verdict"):
        problems.append("main evidence verdict does not match final decision")
    if main.get("status") not in {"passed", "failed"}:
        problems.append("main evidence status is incomplete")
    if main.get("problems") != []:
        problems.append("main evidence reports unresolved problems")
    tasks = main.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(_TASKS):
        problems.append("main evidence does not contain the exact four registered tasks")
        tasks = {}
    for dataset in _TASKS:
        task = tasks.get(dataset)
        if not isinstance(task, dict):
            continue
        if task.get("run_count") != 60:
            problems.append(f"main matrix run count mismatch for {dataset}")
        _audit_training_execution(task, problems)
    execution = main.get("training_execution")
    if not isinstance(execution, dict):
        problems.append("main evidence lacks aggregate training execution")
    else:
        _audit_training_execution(execution, problems)
    unsupervised = main.get("unsupervised")
    if not isinstance(unsupervised, dict) or unsupervised.get("num_images") != 1449:
        problems.append("main VOC unsupervised evidence is not complete for 1449 images")


def _audit_causal(
    causal: Mapping[str, Any],
    *,
    final: Mapping[str, Any],
    source_root: Path,
    revision: str,
    tree_sha256: str,
    problems: list[str],
) -> dict[str, Any]:
    _require_provenance(
        causal,
        kind="causal evidence",
        revision=revision,
        tree_sha256=tree_sha256,
        problems=problems,
    )
    if causal.get("verdict") != final.get("causal_verdict"):
        problems.append("causal evidence verdict does not match final decision")
    if causal.get("verdict") in {None, "incomplete"}:
        problems.append("causal evidence verdict is incomplete")
    if causal.get("status") not in {"passed", "failed"}:
        problems.append("causal evidence status is incomplete")
    if causal.get("problems") != []:
        problems.append("causal evidence reports unresolved problems")
    sources = causal.get("sources")
    if not isinstance(sources, dict) or set(sources) != _CAUSAL_SOURCES:
        problems.append("causal evidence does not contain the exact registered sources")
        sources = {}
    audited_sources: dict[str, Any] = {}
    for name in sorted(_CAUSAL_SOURCES):
        source = sources.get(name)
        if not isinstance(source, dict):
            continue
        report_path = _resolve_source_path(source.get("path"), source_root)
        cache_path = _resolve_source_path(source.get("cache_dir"), source_root)
        report = _artifact(report_path, f"causal source report {name}", problems)
        if cache_path is None or not cache_path.is_dir():
            problems.append(f"missing causal source cache {name}: {cache_path}")
        audited_sources[name] = {**report, "cache_dir": str(cache_path) if cache_path else None}
    return audited_sources


def _audit_extension(
    extension: Mapping[str, Any],
    *,
    revision: str,
    tree_sha256: str,
    problems: list[str],
) -> None:
    _require_provenance(
        extension,
        kind="extension evidence",
        revision=revision,
        tree_sha256=tree_sha256,
        problems=problems,
    )
    if extension.get("status") not in {"passed", "failed"}:
        problems.append("extension evidence status is incomplete")
    if extension.get("verdict") not in {"supported", "mixed_or_negative"}:
        problems.append("extension evidence verdict is incomplete")
    if extension.get("problems") != []:
        problems.append("extension evidence reports unresolved problems")
    imagenet = extension.get("imagenet1k")
    matrix = imagenet.get("matrix", {}) if isinstance(imagenet, dict) else {}
    if not isinstance(matrix, dict) or matrix.get("run_count") != 60:
        problems.append("ImageNet-1k extension does not contain 60 runs")
    ablations = extension.get("high_cost_ablations")
    if not isinstance(ablations, dict):
        problems.append("extension lacks high-cost ablation evidence")
        return
    if ablations.get("required_variant_count") != 15:
        problems.append("extension high-cost ablation contract is not 15 variants")
    for key in ("sources", "comparisons"):
        values = ablations.get(key)
        if not isinstance(values, dict) or len(values) != 15:
            problems.append(f"extension high-cost ablation {key} count is not 15")


def _audit_error_registry(
    registry: Mapping[str, Any],
    *,
    registry_path: Path,
    final_path: Path,
    source_revision: str,
    analyzer: Mapping[str, Any],
    problems: list[str],
) -> dict[str, Any]:
    if registry.get("status") != "passed":
        problems.append("secondary error-analysis registry is not passed")
    if registry.get("evidence_scope") != "prospective_secondary_supervised_error_analysis":
        problems.append("secondary error-analysis registry scope mismatch")
    if registry.get("changes_main_verdict") is not False:
        problems.append("secondary error analysis is not verdict-preserving")
    if registry.get("formal_source_revision") != source_revision:
        problems.append("secondary error-analysis formal revision mismatch")
    registry_analyzer = registry.get("analyzer")
    if not isinstance(registry_analyzer, dict):
        problems.append("secondary error-analysis analyzer provenance is missing")
        registry_analyzer = {}
    for key in ("code_revision", "code_tree_sha256", "code_dirty"):
        if registry_analyzer.get(key) != analyzer.get(key):
            problems.append(f"secondary error-analysis analyzer {key} mismatch")
    formal_final = registry.get("formal_final_decision")
    if not isinstance(formal_final, dict):
        problems.append("secondary error-analysis final-decision identity is missing")
    else:
        recorded_path = Path(str(formal_final.get("path", ""))).resolve()
        actual_final_sha256 = file_sha256(final_path) if final_path.is_file() else None
        if (
            recorded_path != final_path
            or formal_final.get("sha256") != actual_final_sha256
        ):
            problems.append("secondary error-analysis final-decision identity mismatch")
    tasks = registry.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(_TASKS):
        problems.append("secondary error analysis does not contain the exact four tasks")
        tasks = {}
    summaries: dict[str, Any] = {}
    for dataset in _TASKS:
        entry = tasks.get(dataset)
        if not isinstance(entry, dict):
            continue
        summary_path = Path(str(entry.get("path", "")))
        if not summary_path.is_absolute():
            summary_path = (registry_path.parent / summary_path).resolve()
        else:
            summary_path = summary_path.resolve()
        summary = _read_json(summary_path, f"secondary summary {dataset}", problems)
        actual_sha256 = file_sha256(summary_path) if summary_path.is_file() else None
        if entry.get("sha256") != actual_sha256:
            problems.append(f"secondary summary SHA-256 mismatch for {dataset}")
        if summary.get("status") != "passed":
            problems.append(f"secondary summary is not passed for {dataset}")
        if summary.get("changes_main_verdict") is not False:
            problems.append(f"secondary summary changes the verdict for {dataset}")
        formal_source = summary.get("formal_source")
        if (
            not isinstance(formal_source, dict)
            or formal_source.get("code_revision") != source_revision
        ):
            problems.append(f"secondary summary formal revision mismatch for {dataset}")
        if summary.get("task") != _TASK_TYPES[dataset]:
            problems.append(f"secondary summary task mismatch for {dataset}")
        input_reports = summary.get("input_reports")
        if not isinstance(input_reports, list) or len(input_reports) != 33:
            problems.append(f"secondary summary report count mismatch for {dataset}")
        comparisons = summary.get("comparisons")
        if not isinstance(comparisons, dict) or len(comparisons) != 12:
            problems.append(f"secondary summary comparison count mismatch for {dataset}")
        summary_analyzer = summary.get("analyzer", {})
        for key in ("code_revision", "code_tree_sha256", "code_dirty"):
            if summary_analyzer.get(key) != analyzer.get(key):
                problems.append(f"secondary summary analyzer {key} mismatch for {dataset}")
        summaries[dataset] = {"path": str(summary_path), "sha256": actual_sha256}
    return summaries


def audit_research_completion(
    *,
    source_repository_root: Path,
    final_decision_path: Path,
    error_registry_path: Path,
) -> dict[str, Any]:
    """Check execution completeness without reinterpreting the scientific verdict."""

    source_root = source_repository_root.resolve()
    final_path = final_decision_path.resolve()
    registry_path = error_registry_path.resolve()
    problems: list[str] = []
    analyzer = code_provenance()
    if analyzer.get("code_dirty") is not False:
        problems.append("completion audit requires a clean analyzer worktree")
    final = _read_json(final_path, "formal final decision", problems)
    final_identity = _artifact(final_path, "formal final decision", problems)
    source_revision = str(final.get("code_revision", ""))
    source_tree = str(final.get("code_tree_sha256", ""))
    if final.get("verdict") not in _FINAL_VERDICTS:
        problems.append("formal final decision verdict is incomplete")
    if final.get("status") not in {"passed", "failed"}:
        problems.append("formal final decision status is incomplete")
    if final.get("problems") != []:
        problems.append("formal final decision reports unresolved problems")
    if not source_revision or not source_tree:
        problems.append("formal final decision source provenance is missing")
    if final.get("code_dirty") is not False:
        problems.append("formal final decision came from a dirty worktree")
    source_repository: dict[str, Any] = {}
    if source_revision:
        try:
            source_repository = audit_source_repository(
                source_revision,
                repository_root=source_root,
            )
        except (OSError, ValueError) as error:
            problems.append(f"formal source repository audit failed: {error}")
    try:
        actual_source_tree = _source_tree_sha256(source_root)
    except OSError as error:
        actual_source_tree = ""
        problems.append(f"formal source code tree could not be hashed: {error}")
    if source_tree and actual_source_tree != source_tree:
        problems.append("formal source code-tree SHA-256 mismatch")

    main_path = _resolve_source_path(final.get("main_evidence"), source_root)
    causal_path = _resolve_source_path(final.get("causal_evidence"), source_root)
    extension_path = _resolve_source_path(final.get("extension_evidence"), source_root)
    main_identity = _artifact(main_path, "main evidence", problems)
    causal_identity = _artifact(causal_path, "causal evidence", problems)
    main = _read_json(main_path, "main evidence", problems) if main_path else {}
    causal = _read_json(causal_path, "causal evidence", problems) if causal_path else {}
    _audit_main(
        main,
        final=final,
        revision=source_revision,
        tree_sha256=source_tree,
        problems=problems,
    )
    causal_sources = _audit_causal(
        causal,
        final=final,
        source_root=source_root,
        revision=source_revision,
        tree_sha256=source_tree,
        problems=problems,
    )

    extension_identity: dict[str, Any] | None = None
    extension: dict[str, Any] = {}
    if main.get("verdict") == _POSITIVE_MAIN_VERDICT:
        extension_identity = _artifact(extension_path, "extension evidence", problems)
        extension = (
            _read_json(extension_path, "extension evidence", problems)
            if extension_path
            else {}
        )
        _audit_extension(
            extension,
            revision=source_revision,
            tree_sha256=source_tree,
            problems=problems,
        )
        if extension.get("verdict") != final.get("extension_verdict"):
            problems.append("extension verdict does not match final decision")
    elif extension_path is not None:
        problems.append("negative main evidence must not have a conditional extension")

    expected_final_verdict: str | None = None
    if main.get("verdict") == "limited_or_negative" and causal.get("verdict") == (
        "limited_or_negative"
    ):
        expected_final_verdict = "limited_or_negative"
    elif main.get("verdict") == _POSITIVE_MAIN_VERDICT:
        if causal.get("verdict") == "main_task_gain_not_causally_attributed":
            expected_final_verdict = "main_task_gain_not_causally_attributed"
        elif causal.get("verdict") == "supports_core_hypothesis":
            if extension.get("verdict") == "supported":
                expected_final_verdict = (
                    "supports_core_hypothesis_with_scaling_extension"
                )
            elif extension.get("verdict") == "mixed_or_negative":
                expected_final_verdict = "supports_core_hypothesis_limited_scaling"
    if (
        expected_final_verdict is not None
        and final.get("verdict") != expected_final_verdict
    ):
        problems.append("formal final verdict is inconsistent with component verdicts")

    registry = _read_json(registry_path, "secondary error-analysis registry", problems)
    registry_identity = _artifact(
        registry_path, "secondary error-analysis registry", problems
    )
    summaries = _audit_error_registry(
        registry,
        registry_path=registry_path,
        final_path=final_path,
        source_revision=source_revision,
        analyzer=analyzer,
        problems=problems,
    )
    unique_problems = sorted(set(problems))
    complete = not unique_problems
    return {
        "schema_version": 1,
        "status": "passed" if complete else "incomplete",
        "execution_complete": complete,
        "scientific_verdict": final.get("verdict"),
        "changes_scientific_verdict": False,
        "problems": unique_problems,
        "formal_source": {
            "repository": source_repository,
            "code_revision": source_revision,
            "code_tree_sha256": source_tree,
            "actual_code_tree_sha256": actual_source_tree,
        },
        "formal_artifacts": {
            "final": final_identity,
            "main": main_identity,
            "causal": causal_identity,
            "extension": extension_identity,
            "causal_sources": causal_sources,
        },
        "secondary_error_analysis": {
            "registry": registry_identity,
            "summaries": summaries,
        },
        "analyzer": analyzer,
        "research_boundary": (
            "This audit verifies registered execution and provenance completeness only; "
            "it preserves and does not reinterpret the formal scientific verdict."
        ),
    }
