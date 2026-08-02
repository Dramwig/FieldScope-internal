"""Registered cross-run aggregation for supervised per-sample error reports."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.error_analysis import EVIDENCE_SCOPE, SCHEMA_VERSION
from fieldscope.experiments import code_provenance, file_sha256
from fieldscope.statistics import bootstrap_mean_interval

REGISTERED_SEEDS = (4121, 7319, 104729)
REGISTERED_COMPARISONS = {
    "response": (
        "response_shuffled",
        "state",
        "dit_hidden_local",
        "dit_hidden_attention",
        "response_local",
        "response_nograph",
    ),
    "full": (
        "full_shuffled",
        "state",
        "dit_hidden_local",
        "dit_hidden_attention",
        "full_local",
        "full_nograph",
    ),
}
REGISTERED_REPRESENTATIONS = tuple(
    sorted(
        {
            representation
            for candidate, controls in REGISTERED_COMPARISONS.items()
            for representation in (candidate, *controls)
        }
    )
)
BOOTSTRAP_SEED = 4121
BOOTSTRAP_RESAMPLES = 2000

_TASK_METRICS = {
    "classification": ("top1_correct", 1.0),
    "segmentation": ("mean_iou", 1.0),
    "depth": ("abs_rel", -1.0),
}
_TARGET_DESCRIPTORS = {
    "classification": ("target_class",),
    "segmentation": (
        "boundary_density",
        "semantic_class_count",
        "dominant_class_fraction",
    ),
    "depth": (
        "depth_discontinuity_density",
        "valid_fraction",
        "depth_p05_p95_range",
    ),
}


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing supervised error report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Supervised error report must be a JSON object: {path}")
    return payload


def _mapping(value: Any, name: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is missing or invalid: {path}")
    return value


def _finite_number(value: Any, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric: {path}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {path}")
    return result


def _validate_report(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported supervised error report schema: {path}")
    if payload.get("status") != "passed":
        raise ValueError(f"Supervised error report is not passed: {path}")
    if payload.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError(f"Supervised error report evidence scope mismatch: {path}")
    if payload.get("changes_main_verdict") is not False:
        raise ValueError(f"Supervised error report may not change the main verdict: {path}")

    checkpoint = _mapping(payload.get("checkpoint"), "checkpoint provenance", path)
    task = str(checkpoint.get("task", ""))
    representation = str(checkpoint.get("representation", ""))
    seed = checkpoint.get("seed")
    if task not in _TASK_METRICS:
        raise ValueError(f"Unsupported supervised error task {task!r}: {path}")
    if representation not in REGISTERED_REPRESENTATIONS:
        raise ValueError(f"Unregistered representation {representation!r}: {path}")
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"Unregistered seed {seed!r}: {path}")
    for field in ("path", "sha256", "code_revision", "code_tree_sha256"):
        if not checkpoint.get(field):
            raise ValueError(f"Checkpoint {field} is missing: {path}")

    source_files = _mapping(payload.get("source_files"), "source files", path)
    for kind in ("matrix_report", "training_report", "test_report"):
        artifact = _mapping(source_files.get(kind), f"source file {kind}", path)
        if not artifact.get("path") or not artifact.get("sha256"):
            raise ValueError(f"Source file {kind} provenance is incomplete: {path}")

    cache = _mapping(payload.get("test_cache"), "test cache identity", path)
    for field in ("path", "manifest_sha256", "num_samples", "sample_ids_sha256"):
        if cache.get(field) in (None, ""):
            raise ValueError(f"Test cache {field} is missing: {path}")
    analyzer = _mapping(payload.get("analyzer"), "analyzer provenance", path)
    if analyzer.get("code_dirty") is not False:
        raise ValueError(f"Per-sample analyzer was not clean: {path}")
    for field in ("code_revision", "code_tree_sha256"):
        if not analyzer.get(field):
            raise ValueError(f"Analyzer {field} is missing: {path}")
    compatibility = _mapping(
        analyzer.get("replay_core_compatibility"), "replay core compatibility", path
    )
    if compatibility.get("status") != "passed":
        raise ValueError(f"Replay core compatibility did not pass: {path}")

    records = payload.get("per_sample")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Per-sample records are missing: {path}")
    if payload.get("num_samples") != len(records) or cache.get("num_samples") != len(
        records
    ):
        raise ValueError(f"Per-sample count mismatch: {path}")
    sample_ids: list[str] = []
    metric, _ = _TASK_METRICS[task]
    descriptors = _TARGET_DESCRIPTORS[task]
    for record in records:
        record = _mapping(record, "per-sample record", path)
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Sample ID is missing or invalid: {path}")
        sample_ids.append(sample_id)
        metric_value = _finite_number(
            record.get(metric), f"sample {sample_id}/{metric}", path
        )
        if task in {"classification", "segmentation"} and not 0 <= metric_value <= 1:
            raise ValueError(f"Sample {sample_id}/{metric} is outside [0, 1]: {path}")
        if task == "classification" and metric_value not in {0.0, 1.0}:
            raise ValueError(
                f"Sample {sample_id}/top1_correct is not a binary indicator: {path}"
            )
        if task == "depth" and metric_value < 0:
            raise ValueError(f"Sample {sample_id}/abs_rel is negative: {path}")
        for descriptor in descriptors:
            descriptor_value = _finite_number(
                record.get(descriptor), f"sample {sample_id}/{descriptor}", path
            )
            if descriptor == "target_class" and (
                descriptor_value < 0 or not descriptor_value.is_integer()
            ):
                raise ValueError(
                    f"Sample {sample_id}/target_class is not a non-negative integer: {path}"
                )
            if descriptor == "semantic_class_count" and (
                descriptor_value < 1 or not descriptor_value.is_integer()
            ):
                raise ValueError(
                    f"Sample {sample_id}/semantic_class_count is not a positive integer: "
                    f"{path}"
                )
            if descriptor in {
                "boundary_density",
                "dominant_class_fraction",
                "depth_discontinuity_density",
                "valid_fraction",
            } and not 0 <= descriptor_value <= 1:
                raise ValueError(
                    f"Sample {sample_id}/{descriptor} is outside [0, 1]: {path}"
                )
            if descriptor == "depth_p05_p95_range" and descriptor_value < 0:
                raise ValueError(
                    f"Sample {sample_id}/depth_p05_p95_range is negative: {path}"
                )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in supervised error report: {path}")
    digest = sample_ids_sha256(sample_ids)
    if payload.get("sample_ids_sha256") != digest or cache.get(
        "sample_ids_sha256"
    ) != digest:
        raise ValueError(f"Sample-ID SHA-256 mismatch: {path}")
    return {
        "task": task,
        "representation": representation,
        "seed": int(seed),
        "checkpoint": checkpoint,
        "cache": cache,
        "test_control_contract": payload.get("test_control_contract"),
        "records": records,
        "sample_ids": sample_ids,
    }


def _summary(values: Sequence[float], *, bootstrap: bool) -> dict[str, Any]:
    if not values:
        return {
            "num_samples": 0,
            "mean_delta": None,
            "candidate_better_fraction": None,
            "paired_sample_bootstrap": None,
        }
    result: dict[str, Any] = {
        "num_samples": len(values),
        "mean_delta": mean(values),
        "candidate_better_fraction": sum(value > 0 for value in values) / len(values),
    }
    if bootstrap:
        result["paired_sample_bootstrap"] = {
            **bootstrap_mean_interval(
                list(values), seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES
            ),
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "aligned_test_sample",
        }
    return result


def _rank_quartiles(
    descriptors: Mapping[str, float],
) -> dict[str, list[str]]:
    ordered = sorted(descriptors, key=lambda sample_id: (descriptors[sample_id], sample_id))
    count = len(ordered)
    groups = {f"Q{index}": [] for index in range(1, 5)}
    for rank, sample_id in enumerate(ordered):
        quartile = min(4, (rank * 4 // count) + 1)
        groups[f"Q{quartile}"].append(sample_id)
    return groups


def _stratum(
    sample_ids: Sequence[str],
    deltas: Mapping[str, float],
    descriptors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    samples = []
    for sample_id in sample_ids:
        sample = {"sample_id": sample_id, "delta": deltas[sample_id]}
        if descriptors is not None:
            sample["descriptor"] = descriptors[sample_id]
        samples.append(sample)
    values = [deltas[sample_id] for sample_id in sample_ids]
    result = {"samples": samples, "summary": _summary(values, bootstrap=True)}
    if descriptors is not None and sample_ids:
        descriptor_values = [descriptors[sample_id] for sample_id in sample_ids]
        result["descriptor_range"] = [min(descriptor_values), max(descriptor_values)]
    return result


def _quartile_analysis(
    descriptor_name: str,
    targets: Mapping[str, Mapping[str, float]],
    deltas: Mapping[str, float],
) -> dict[str, Any]:
    descriptor_values = {
        sample_id: float(values[descriptor_name]) for sample_id, values in targets.items()
    }
    groups = _rank_quartiles(descriptor_values)
    return {
        "descriptor": descriptor_name,
        "assignment": (
            "stable rank quartiles over the whole fixed test split, ordered by "
            "(descriptor, sample_id)"
        ),
        "strata": {
            name: _stratum(sample_ids, deltas, descriptor_values)
            for name, sample_ids in groups.items()
        },
    }


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return {"status": "undefined_insufficient_samples", "coefficient": None, "n": len(x_values)}
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    x_sum = sum((value - x_mean) ** 2 for value in x_values)
    y_sum = sum((value - y_mean) ** 2 for value in y_values)
    denominator = math.sqrt(x_sum * y_sum)
    if denominator == 0:
        return {
            "status": "undefined_constant_input",
            "coefficient": None,
            "n": len(x_values),
        }
    return {"status": "passed", "coefficient": numerator / denominator, "n": len(x_values)}


def _classification_analysis(
    targets: Mapping[str, Mapping[str, float]], deltas: Mapping[str, float]
) -> dict[str, Any]:
    class_samples: dict[int, list[str]] = {}
    for sample_id, values in targets.items():
        target_class = int(values["target_class"])
        class_samples.setdefault(target_class, []).append(sample_id)
    per_class = {
        str(target_class): _stratum(sample_ids, deltas)
        for target_class, sample_ids in sorted(class_samples.items())
    }
    class_mean_deltas = [
        float(details["summary"]["mean_delta"]) for details in per_class.values()
    ]
    return {
        "per_class": per_class,
        "macro_class_distribution": {
            "num_classes": len(class_mean_deltas),
            "class_mean_deltas": class_mean_deltas,
            "macro_mean_delta": mean(class_mean_deltas),
            "candidate_better_class_fraction": sum(
                value > 0 for value in class_mean_deltas
            )
            / len(class_mean_deltas),
            "class_bootstrap": {
                **bootstrap_mean_interval(
                    class_mean_deltas,
                    seed=BOOTSTRAP_SEED,
                    resamples=BOOTSTRAP_RESAMPLES,
                ),
                "seed": BOOTSTRAP_SEED,
                "resampling_unit": "target_class",
            },
        },
    }


def _segmentation_analysis(
    targets: Mapping[str, Mapping[str, float]], deltas: Mapping[str, float]
) -> dict[str, Any]:
    class_count_groups = {"1": [], "2": [], "3+": []}
    for sample_id, values in targets.items():
        count = int(values["semantic_class_count"])
        class_count_groups[str(count) if count < 3 else "3+"].append(sample_id)
    sample_ids = list(targets)
    correlations = {}
    for descriptor in (
        "boundary_density",
        "semantic_class_count",
        "dominant_class_fraction",
    ):
        correlations[descriptor] = _pearson(
            [float(targets[sample_id][descriptor]) for sample_id in sample_ids],
            [deltas[sample_id] for sample_id in sample_ids],
        )
    return {
        "quartiles": {
            descriptor: _quartile_analysis(descriptor, targets, deltas)
            for descriptor in ("boundary_density", "dominant_class_fraction")
        },
        "semantic_class_count_groups": {
            name: _stratum(ids, deltas) for name, ids in class_count_groups.items()
        },
        "pearson_correlations": correlations,
    }


def _depth_analysis(
    targets: Mapping[str, Mapping[str, float]], deltas: Mapping[str, float]
) -> dict[str, Any]:
    descriptors = (
        "depth_discontinuity_density",
        "valid_fraction",
        "depth_p05_p95_range",
    )
    sample_ids = list(targets)
    return {
        "quartiles": {
            descriptor: _quartile_analysis(descriptor, targets, deltas)
            for descriptor in descriptors
        },
        "pearson_correlations": {
            descriptor: _pearson(
                [float(targets[sample_id][descriptor]) for sample_id in sample_ids],
                [deltas[sample_id] for sample_id in sample_ids],
            )
            for descriptor in descriptors
        },
    }


def summarize_readout_errors(
    report_paths: Sequence[Path], *, command: Sequence[str] | None = None
) -> dict[str, Any]:
    """Aggregate the complete registered 11-representation by three-seed report set."""

    started = time.perf_counter()
    analyzer = code_provenance()
    if analyzer.get("code_dirty") is not False:
        raise ValueError("Supervised error aggregation requires a clean analyzer worktree")
    expected_count = len(REGISTERED_REPRESENTATIONS) * len(REGISTERED_SEEDS)
    if len(report_paths) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} registered reports, got {len(report_paths)}"
        )

    reports: dict[tuple[str, int], dict[str, Any]] = {}
    inputs = []
    for original_path in report_paths:
        path = original_path.resolve()
        payload = _read_report(path)
        validated = _validate_report(path, payload)
        key = (validated["representation"], validated["seed"])
        if key in reports:
            raise ValueError(
                f"Duplicate representation/seed report: {key[0]}/seed{key[1]}"
            )
        reports[key] = validated
        inputs.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "task": validated["task"],
                "representation": key[0],
                "seed": key[1],
            }
        )
    expected_keys = {
        (representation, seed)
        for representation in REGISTERED_REPRESENTATIONS
        for seed in REGISTERED_SEEDS
    }
    missing = sorted(expected_keys - reports.keys())
    unexpected = sorted(reports.keys() - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"Incomplete registered report matrix; missing={missing}, unexpected={unexpected}"
        )

    first = reports[next(iter(sorted(reports)))]
    task = first["task"]
    sample_ids = first["sample_ids"]
    source_revision = first["checkpoint"]["code_revision"]
    source_tree = first["checkpoint"]["code_tree_sha256"]
    cache = first["cache"]
    control_contract = first["test_control_contract"]
    descriptors = _TARGET_DESCRIPTORS[task]
    targets: dict[str, dict[str, float]] = {}
    for record in first["records"]:
        targets[str(record["sample_id"])] = {
            descriptor: float(record[descriptor]) for descriptor in descriptors
        }
    for (representation, seed), report in sorted(reports.items()):
        if report["task"] != task:
            raise ValueError("All registered reports must belong to one task")
        if report["checkpoint"]["code_revision"] != source_revision or report[
            "checkpoint"
        ]["code_tree_sha256"] != source_tree:
            raise ValueError("Formal checkpoint provenance differs across reports")
        if report["cache"] != cache:
            raise ValueError("Test cache identity differs across reports")
        if report["test_control_contract"] != control_contract:
            raise ValueError("Test control contract differs across reports")
        if report["sample_ids"] != sample_ids:
            raise ValueError(
                f"Sample IDs are not exactly aligned for {representation}/seed{seed}"
            )
        for record in report["records"]:
            sample_id = str(record["sample_id"])
            actual = {descriptor: float(record[descriptor]) for descriptor in descriptors}
            if actual != targets[sample_id]:
                raise ValueError(
                    f"Target descriptors differ for {sample_id} in "
                    f"{representation}/seed{seed}"
                )

    metric, direction = _TASK_METRICS[task]
    comparisons: dict[str, Any] = {}
    for candidate, controls in REGISTERED_COMPARISONS.items():
        for control in controls:
            name = f"{candidate}-vs-{control}"
            seed_results: dict[str, Any] = {}
            seed_deltas: dict[int, dict[str, float]] = {}
            for seed in REGISTERED_SEEDS:
                candidate_records = {
                    str(record["sample_id"]): record
                    for record in reports[(candidate, seed)]["records"]
                }
                control_records = {
                    str(record["sample_id"]): record
                    for record in reports[(control, seed)]["records"]
                }
                samples = []
                deltas = {}
                for sample_id in sample_ids:
                    candidate_value = float(candidate_records[sample_id][metric])
                    control_value = float(control_records[sample_id][metric])
                    delta = direction * (candidate_value - control_value)
                    deltas[sample_id] = delta
                    samples.append(
                        {
                            "sample_id": sample_id,
                            "candidate_value": candidate_value,
                            "control_value": control_value,
                            "delta": delta,
                        }
                    )
                seed_deltas[seed] = deltas
                seed_results[str(seed)] = {
                    "samples": samples,
                    "summary": _summary(list(deltas.values()), bootstrap=False),
                }
            cross_seed_deltas = {
                sample_id: mean(seed_deltas[seed][sample_id] for seed in REGISTERED_SEEDS)
                for sample_id in sample_ids
            }
            fixed_analysis: dict[str, Any]
            if task == "classification":
                fixed_analysis = _classification_analysis(targets, cross_seed_deltas)
            elif task == "segmentation":
                fixed_analysis = _segmentation_analysis(targets, cross_seed_deltas)
            else:
                fixed_analysis = _depth_analysis(targets, cross_seed_deltas)
            comparisons[name] = {
                "candidate": candidate,
                "control": control,
                "metric": metric,
                "delta_definition": (
                    f"{candidate}_{metric} - {control}_{metric}"
                    if direction > 0
                    else f"{control}_{metric} - {candidate}_{metric}"
                ),
                "positive_means_candidate_better": True,
                "seeds": seed_results,
                "cross_seed": {
                    "samples": [
                        {"sample_id": sample_id, "mean_delta": cross_seed_deltas[sample_id]}
                        for sample_id in sample_ids
                    ],
                    "summary": _summary(
                        list(cross_seed_deltas.values()), bootstrap=True
                    ),
                    "fixed_analysis": fixed_analysis,
                },
            }

    return {
        "schema_version": 1,
        "status": "passed",
        "evidence_scope": EVIDENCE_SCOPE,
        "changes_main_verdict": False,
        "task": task,
        "formal_source": {
            "code_revision": source_revision,
            "code_tree_sha256": source_tree,
        },
        "test_cache": cache,
        "test_control_contract": control_contract,
        "registered_contract": {
            "representations": list(REGISTERED_REPRESENTATIONS),
            "seeds": list(REGISTERED_SEEDS),
            "comparisons": {
                candidate: list(controls)
                for candidate, controls in REGISTERED_COMPARISONS.items()
            },
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "analyzer": {**analyzer, "command": list(command or [])},
        "input_reports": sorted(
            inputs, key=lambda item: (item["representation"], item["seed"])
        ),
        "num_samples": len(sample_ids),
        "sample_ids_sha256": sample_ids_sha256(sample_ids),
        "target_descriptors": [
            {"sample_id": sample_id, **targets[sample_id]} for sample_id in sample_ids
        ],
        "comparisons": comparisons,
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
        "research_boundary": (
            "This prospective secondary analysis explains completed formal results; "
            "it does not alter the registered main or final verdict."
        ),
    }
