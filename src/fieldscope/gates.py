"""Pre-registered promotion gates for expensive FieldScope experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fieldscope.cache import load_features
from fieldscope.experiments import code_provenance, file_sha256

_RANDOMNESS_POLICY = {
    "path_noise": "sample_id_sha256_seeded_v1",
    "probe_basis": "shared_fixed_seed_v1",
}
_SIGNAL_SEEDS = {4121, 7319, 104729}
_SIGNAL_REPRESENTATIONS = {
    "state",
    "response",
    "full",
    "dit_hidden_local",
    "dit_hidden_attention",
    "response_shuffled",
    "full_shuffled",
}
_VOC_METRICS = ("boundary_average_precision", "pairwise_auroc")
_MIN_CLASSIFICATION_GAIN = 0.005


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def audit_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    """Verify a completed cache, shard digests, and the current randomness contract."""

    manifest_path = cache_dir / "dataset_manifest.json"
    problems: list[str] = []
    if not manifest_path.is_file():
        return {
            "status": "incomplete",
            "cache_dir": str(cache_dir),
            "problems": [f"missing {manifest_path}"],
        }
    manifest = _read_json(manifest_path)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        problems.append("manifest has no shards")
        shards = []
    expected_samples = int(manifest.get("num_samples", -1))
    shard_samples = sum(int(shard.get("num_samples", 0)) for shard in shards)
    if manifest.get("complete") is not True:
        problems.append("manifest is not complete")
    if expected_samples < 1 or shard_samples != expected_samples:
        problems.append(
            f"sample count mismatch manifest={expected_samples} shards={shard_samples}"
        )
    randomness = manifest.get("randomness", {})
    for name, expected in _RANDOMNESS_POLICY.items():
        if randomness.get(name) != expected:
            problems.append(
                f"randomness.{name}={randomness.get(name)!r}, expected {expected!r}"
            )
    code_revision = manifest.get("code_revision")
    if not isinstance(code_revision, str) or not code_revision:
        problems.append("manifest code_revision is missing")
    for shard in shards:
        shard_path = cache_dir / str(shard.get("path", ""))
        if not shard_path.is_file():
            problems.append(f"missing shard {shard_path.name}")
            continue
        expected_bytes = int(shard.get("bytes", -1))
        if expected_bytes < 0 or shard_path.stat().st_size != expected_bytes:
            problems.append(f"shard size mismatch {shard_path.name}")
            continue
        expected_sha256 = shard.get("sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            problems.append(f"missing shard sha256 {shard_path.name}")
            continue
        if file_sha256(shard_path) != expected_sha256:
            problems.append(f"shard sha256 mismatch {shard_path.name}")
            continue
        _, _, shard_manifest = load_features(shard_path)
        metadata = shard_manifest.get("metadata", {})
        if metadata.get("noise_policy") != _RANDOMNESS_POLICY["path_noise"]:
            problems.append(f"shard noise policy mismatch {shard_path.name}")
        if metadata.get("probe_basis") != _RANDOMNESS_POLICY["probe_basis"]:
            problems.append(f"shard probe basis mismatch {shard_path.name}")
    return {
        "status": "passed" if not problems else "failed",
        "cache_dir": str(cache_dir),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "num_samples": expected_samples,
        "code_revision": code_revision,
        "randomness": randomness,
        "problems": problems,
    }


def _finite_metric(report: Mapping[str, Any], representation: str, metric: str) -> float:
    try:
        value = float(report["representations"][representation]["means"][metric])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing {representation}/{metric}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {representation}/{metric}")
    return value


def _voc_signal_checks(report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    problems: list[str] = []
    if report.get("status") != "passed":
        problems.append("VOC diagnosis is not a passed report")
    comparisons = {
        "response_gt_shuffled": ("response", "response_shuffled"),
        "response_gt_state": ("response", "state"),
        "response_gt_dit_hidden": ("response", "dit_hidden"),
        "response_gt_dit_attention": ("response", "dit_attention"),
    }
    for metric in _VOC_METRICS:
        for name, (candidate, reference) in comparisons.items():
            if metric == "pairwise_auroc" and name in {
                "response_gt_dit_hidden",
                "response_gt_dit_attention",
            }:
                continue
            try:
                candidate_value = _finite_metric(report, candidate, metric)
                reference_value = _finite_metric(report, reference, metric)
            except ValueError as error:
                problems.append(str(error))
                continue
            checks.append(
                {
                    "name": f"voc_{metric}_{name}",
                    "candidate": candidate_value,
                    "reference": reference_value,
                    "difference": candidate_value - reference_value,
                    "passed": candidate_value > reference_value,
                }
            )
    return checks, problems


def _matrix_observations(report: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    observations: dict[str, dict[int, float]] = {}
    for run in report.get("runs", []):
        representation = str(run.get("representation"))
        seed = int(run.get("seed"))
        value = float(run.get("test_metric"))
        if not math.isfinite(value):
            raise ValueError(f"non-finite CIFAR metric for {representation}/seed-{seed}")
        observations.setdefault(representation, {})[seed] = value
    return observations


def _classification_signal_checks(
    report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    problems: list[str] = []
    checks: list[dict[str, Any]] = []
    if report.get("status") != "passed" or report.get("metric") != "top1":
        problems.append("CIFAR matrix is not a passed top1 report")
    try:
        observations = _matrix_observations(report)
    except (TypeError, ValueError) as error:
        return checks, [str(error)]
    for representation in sorted(_SIGNAL_REPRESENTATIONS):
        seeds = set(observations.get(representation, {}))
        if seeds != _SIGNAL_SEEDS:
            problems.append(
                f"CIFAR {representation} seeds={sorted(seeds)}, "
                f"expected={sorted(_SIGNAL_SEEDS)}"
            )
    if problems:
        return checks, problems
    static_controls = ("state", "dit_hidden_local", "dit_hidden_attention")
    for candidate in ("response", "full"):
        shuffled = (
            "response_shuffled" if candidate == "response" else "full_shuffled"
        )
        for reference_group, references in (
            ("shuffled", (shuffled,)),
            ("best_static_or_hidden", static_controls),
        ):
            differences: list[float] = []
            for seed in sorted(_SIGNAL_SEEDS):
                reference_value = max(observations[name][seed] for name in references)
                differences.append(observations[candidate][seed] - reference_value)
            mean_difference = sum(differences) / len(differences)
            positive_seeds = sum(value > 0 for value in differences)
            checks.append(
                {
                    "name": f"cifar_{candidate}_gt_{reference_group}",
                    "differences": differences,
                    "mean_difference": mean_difference,
                    "minimum_required_mean_difference": _MIN_CLASSIFICATION_GAIN,
                    "positive_seeds": positive_seeds,
                    "minimum_positive_seeds": 2,
                    "passed": (
                        mean_difference >= _MIN_CLASSIFICATION_GAIN
                        and positive_seeds >= 2
                    ),
                }
            )
    return checks, problems


def audit_signal_gate(
    *,
    cache_dirs: Sequence[Path],
    voc_report_path: Path,
    cifar_matrix_path: Path,
) -> dict[str, Any]:
    """Return an objective promotion verdict without treating the gate as paper evidence."""

    provenance = code_provenance()
    cache_audits = [audit_cache_manifest(path) for path in cache_dirs]
    input_problems: list[str] = []
    for audit in cache_audits:
        if audit["status"] != "passed":
            input_problems.extend(audit["problems"])
        revision = audit.get("code_revision")
        if revision and revision != provenance["code_revision"]:
            input_problems.append(
                f"cache revision {revision} != current {provenance['code_revision']}"
            )
    if not voc_report_path.is_file():
        input_problems.append(f"missing {voc_report_path}")
    if not cifar_matrix_path.is_file():
        input_problems.append(f"missing {cifar_matrix_path}")
    if input_problems:
        return {
            "status": "incomplete",
            "verdict": "incomplete",
            **provenance,
            "paper_evidence": False,
            "cache_audits": cache_audits,
            "input_problems": sorted(set(input_problems)),
            "checks": [],
        }
    voc_report = _read_json(voc_report_path)
    cifar_report = _read_json(cifar_matrix_path)
    for name, report in (("VOC", voc_report), ("CIFAR", cifar_report)):
        revision = report.get("code_revision")
        if revision != provenance["code_revision"]:
            input_problems.append(
                f"{name} report revision {revision} != current {provenance['code_revision']}"
            )
    voc_checks, voc_problems = _voc_signal_checks(voc_report)
    cifar_checks, cifar_problems = _classification_signal_checks(cifar_report)
    input_problems.extend(voc_problems)
    input_problems.extend(cifar_problems)
    checks = voc_checks + cifar_checks
    if input_problems:
        verdict = "incomplete"
        status = "incomplete"
    elif checks and all(check["passed"] for check in checks):
        verdict = "proceed"
        status = "passed"
    else:
        verdict = "stop_or_redesign"
        status = "failed"
    return {
        "status": status,
        "verdict": verdict,
        **provenance,
        "paper_evidence": False,
        "minimum_classification_gain": _MIN_CLASSIFICATION_GAIN,
        "cache_audits": cache_audits,
        "input_problems": sorted(set(input_problems)),
        "checks": checks,
        "sources": {
            "voc_report": str(voc_report_path),
            "cifar_matrix": str(cifar_matrix_path),
        },
    }
