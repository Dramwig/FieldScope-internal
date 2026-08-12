"""Registered assets and configurations for conditional extension validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from fieldscope.cache import load_features
from fieldscope.config import (
    RunConfig,
    apply_runtime_profile,
    runtime_profile_identity,
)
from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.evidence import (
    _GRAPH_CONTROLS,
    _REPRESENTATIONS,
    _SEEDS,
    _SHUFFLED_CONTROL,
    _STATIC_CONTROLS,
    _paired_gain,
    _validate_matrix,
    _validate_runtime_profile,
)
from fieldscope.experiments import cache_identity, code_provenance, file_sha256
from fieldscope.readout_runtime_gate import readout_runtime_profile_identity
from fieldscope.runtime_gate import load_runtime_gate_report
from fieldscope.statistics import bootstrap_mean_interval

IMAGENET1K_MANIFEST_BYTES = 405_484_553
IMAGENET1K_MANIFEST_SHA256 = (
    "9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0"
)
IMAGENET1K_SOURCE_COUNTS = {"train": 1_281_167, "val": 50_000}
IMAGENET1K_CLASS_COUNT = 1_000
IMAGENET1K_SOURCE_SNAPSHOT = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"

HIGH_COST_ABLATION_VARIANTS: dict[str, tuple[str, Any]] = {
    "time_020": ("probe.times", [0.2]),
    "time_050": ("probe.times", [0.5]),
    "time_080": ("probe.times", [0.8]),
    "directions_1": ("probe.num_directions", 1),
    "directions_2": ("probe.num_directions", 2),
    "directions_4": ("probe.num_directions", 4),
    "difference_forward": ("probe.difference", "forward"),
    "probe_gaussian": ("probe.probe_type", "gaussian"),
    "eta_0015": ("probe.eta", 0.015),
    "eta_0060": ("probe.eta", 0.06),
    "topk_0": ("probe.topk", 0),
    "topk_8": ("probe.topk", 8),
    "topk_32": ("probe.topk", 32),
    "radius_0": ("probe.local_radius", 0),
    "radius_2": ("probe.local_radius", 2),
}


def _registered_main_config(mapping: dict[str, Any]) -> bool:
    backend = mapping.get("backend", {})
    probe = mapping.get("probe", {})
    return bool(
        backend.get("name") == "auraflow"
        and backend.get("variant") == "fp16"
        and backend.get("image_size") == 512
        and backend.get("prompt") == ""
        and probe.get("times") == [0.2, 0.5, 0.8]
        and probe.get("num_directions") == 8
        and probe.get("probe_type") == "structured"
        and probe.get("eta") == 0.03
        and probe.get("difference") == "central"
        and probe.get("graph_grid") == [16, 16]
        and probe.get("topk") == 16
        and probe.get("local_radius") == 1
    )


def build_high_cost_ablation_configs(
    base_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize the 15 pre-registered one-factor configs and their hashes."""

    raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not _registered_main_config(raw):
        raise ValueError("base config does not match the registered 512 px main setting")
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    for name, (field_name, value) in HIGH_COST_ABLATION_VARIANTS.items():
        section, key = field_name.split(".", maxsplit=1)
        mapping = copy.deepcopy(raw)
        baseline_value = mapping[section][key]
        mapping[section][key] = value
        mapping["runtime"]["output_dir"] = (
            f"outputs/full_validation/auraflow_v03/extension/ablations/{name}"
        )
        mapping["runtime"]["cache_dir"] = (
            "${FIELDSCOPE_DATASETS_ROOT}/feature_cache/"
            f"auraflow_v03_extension/ablations/{name}"
        )
        RunConfig.from_mapping(mapping)
        config_path = output_dir / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        variants.append(
            {
                "name": name,
                "changed_field": field_name,
                "baseline_value": baseline_value,
                "value": value,
                "config_path": str(config_path.resolve()),
                "config_sha256": file_sha256(config_path),
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "evidence_scope": "configuration_and_provenance_only",
        "method_effectiveness_conclusion": None,
        "base_config_path": str(base_config_path.resolve()),
        "base_config_sha256": file_sha256(base_config_path),
        "variant_count": len(variants),
        "variants": variants,
        **code_provenance(),
    }


def audit_imagenet1k_asset(
    *,
    dataset_root: Path,
    manifest_path: Path,
    export_summary_path: Path,
    expected_manifest_bytes: int = IMAGENET1K_MANIFEST_BYTES,
    expected_manifest_sha256: str = IMAGENET1K_MANIFEST_SHA256,
    expected_source_counts: dict[str, int] | None = None,
    expected_class_count: int = IMAGENET1K_CLASS_COUNT,
) -> dict[str, Any]:
    """Verify the registered full ImageNet-1k export without decoding images."""

    expected_counts = expected_source_counts or IMAGENET1K_SOURCE_COUNTS
    problems: list[str] = []
    if not dataset_root.is_dir():
        problems.append(f"missing dataset root {dataset_root}")
    if not manifest_path.is_file():
        problems.append(f"missing manifest {manifest_path}")
    if not export_summary_path.is_file():
        problems.append(f"missing export summary {export_summary_path}")
    if problems:
        return {
            "status": "failed",
            "problems": problems,
            "dataset_root": str(dataset_root),
            "manifest_path": str(manifest_path),
            "export_summary_path": str(export_summary_path),
            **code_provenance(),
        }

    summary = json.loads(export_summary_path.read_text(encoding="utf-8"))
    if summary.get("split_counts") != expected_counts:
        problems.append("export summary split counts mismatch")
    if summary.get("manifest_sha256") != expected_manifest_sha256:
        problems.append("export summary manifest SHA-256 mismatch")
    if summary.get("image_size") != 256:
        problems.append("export summary image size mismatch")

    digest = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = {
        split: Counter() for split in expected_counts
    }
    label_to_wnid: dict[int, str] = {}
    seen_paths: set[str] = set()
    missing_examples: list[str] = []
    invalid_examples: list[str] = []
    manifest_bytes = 0
    with manifest_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            manifest_bytes += len(line)
            try:
                record = json.loads(line)
                split = str(record["split"])
                wnid = str(record["wnid"])
                label = int(record["label"])
                relative_path = str(record["path"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                if len(invalid_examples) < 10:
                    invalid_examples.append(f"line {line_number}: invalid JSON record")
                continue
            expected_prefix = f"extracted/{split}/{wnid}/"
            valid = bool(
                split in expected_counts
                and relative_path.startswith(expected_prefix)
                and relative_path.lower().endswith(('.jpg', '.jpeg'))
                and int(record.get("height", -1)) == 256
                and int(record.get("width", -1)) == 256
            )
            previous_wnid = label_to_wnid.setdefault(label, wnid)
            if previous_wnid != wnid:
                valid = False
            if relative_path in seen_paths:
                valid = False
            seen_paths.add(relative_path)
            if not valid:
                if len(invalid_examples) < 10:
                    invalid_examples.append(f"line {line_number}: {relative_path}")
                continue
            split_counts[split] += 1
            class_counts[split][wnid] += 1
            image_path = dataset_root.parent / relative_path
            if not image_path.is_file() and len(missing_examples) < 10:
                missing_examples.append(relative_path)

    actual_sha256 = digest.hexdigest()
    if manifest_bytes != expected_manifest_bytes:
        problems.append(
            f"manifest bytes {manifest_bytes} != expected {expected_manifest_bytes}"
        )
    if actual_sha256 != expected_manifest_sha256:
        problems.append("manifest SHA-256 mismatch")
    if dict(split_counts) != expected_counts:
        problems.append(f"manifest split counts mismatch: {dict(split_counts)}")
    if invalid_examples:
        problems.append("manifest contains invalid or duplicate records")
    if missing_examples:
        problems.append("manifest references missing image files")
    if set(label_to_wnid) != set(range(expected_class_count)):
        problems.append("manifest labels are not contiguous over the registered classes")
    class_count_report: dict[str, int] = {}
    for split in expected_counts:
        split_root = dataset_root / split
        if not split_root.is_dir():
            problems.append(f"missing {split} directory")
            class_dirs: set[str] = set()
        else:
            class_dirs = {path.name for path in split_root.iterdir() if path.is_dir()}
        class_count_report[split] = len(class_dirs)
        if len(class_dirs) != expected_class_count:
            problems.append(f"{split} class-directory count mismatch")
        if set(class_counts[split]) != class_dirs:
            problems.append(f"{split} manifest classes do not match class directories")

    holdout_count = sum(
        max(1, round(count * 0.1)) for count in class_counts["train"].values()
    )
    fieldscope_split_counts = {
        "train": expected_counts["train"] - holdout_count,
        "val": holdout_count,
        "test": expected_counts["val"],
    }
    return {
        "schema_version": 1,
        "status": "passed" if not problems else "failed",
        "evidence_scope": "asset_identity_and_split_counts_only",
        "method_effectiveness_conclusion": None,
        "dataset_root": str(dataset_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": actual_sha256,
        "export_summary_path": str(export_summary_path.resolve()),
        "source_snapshot": IMAGENET1K_SOURCE_SNAPSHOT,
        "source_split_counts": dict(split_counts),
        "class_directory_counts": class_count_report,
        "fieldscope_split_counts": fieldscope_split_counts,
        "manifest_unique_paths": len(seen_paths),
        "invalid_examples": invalid_examples,
        "missing_examples": missing_examples,
        "problems": problems,
        **code_provenance(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _resolve_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _validate_dense_voc_ablation(
    *,
    report_path: Path,
    config_path: Path,
    runtime_profile_path: Path,
    provenance: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], list[str]]:
    problems: list[str] = []
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = RunConfig.from_mapping(raw_config)
    try:
        load_runtime_gate_report(config, runtime_profile_path)
        runtime_identity = runtime_profile_identity(runtime_profile_path)
        expected_config = apply_runtime_profile(config, runtime_profile_path)
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {}, {"report_path": str(report_path)}, [
            f"invalid ablation runtime profile {runtime_profile_path}: {error}"
        ]
    if runtime_identity.get("code_revision") != provenance["code_revision"]:
        problems.append(f"ablation runtime revision mismatch {runtime_profile_path}")
    if runtime_identity.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append(f"ablation runtime code tree mismatch {runtime_profile_path}")
    if not report_path.is_file():
        return {}, {"report_path": str(report_path)}, problems + [
            f"missing ablation report {report_path}"
        ]
    report = _read_json(report_path)
    if report.get("status") != "passed":
        problems.append(f"ablation report is not passed {report_path}")
    for key in ("code_revision", "code_tree_sha256"):
        if report.get(key) != provenance[key]:
            problems.append(f"ablation report {key} mismatch {report_path}")
    if report.get("code_dirty") is not False:
        problems.append(f"ablation report came from a dirty worktree {report_path}")
    cache_dir = _resolve_path(str(report.get("cache_dir", "")), repository_root)
    manifest_path = cache_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {}, {"report_path": str(report_path), "cache_dir": str(cache_dir)}, problems + [
            f"missing ablation cache manifest {manifest_path}"
        ]
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "passed" or manifest.get("complete") is not True:
        problems.append(f"ablation cache is incomplete {cache_dir}")
    if (
        manifest.get("dataset") != "voc2012"
        or manifest.get("split") != "test"
        or int(manifest.get("num_samples", -1)) != 1449
        or manifest.get("storage_policy") != "dense"
    ):
        problems.append(f"ablation cache identity mismatch {cache_dir}")
    for key in ("code_revision", "code_tree_sha256"):
        if manifest.get(key) != provenance[key]:
            problems.append(f"ablation cache {key} mismatch {cache_dir}")
    if manifest.get("code_dirty") is not False:
        problems.append(f"ablation cache came from a dirty worktree {cache_dir}")
    if manifest.get("runtime_profile") != runtime_identity:
        problems.append(f"ablation cache runtime profile mismatch {cache_dir}")
    if manifest.get("config") != expected_config.to_dict():
        problems.append(f"ablation cache config mismatch {cache_dir}")
    backend = manifest.get("backend", {})
    if (
        backend.get("backend") != "auraflow"
        or backend.get("frozen") is not True
        or backend.get("random_transformer") is not False
        or backend.get("prompt") != ""
        or backend.get("image_size") != 512
    ):
        problems.append(f"ablation cache backend mismatch {cache_dir}")
    if manifest.get("randomness", {}).get("path_noise") != "sample_id_sha256_seeded_v1":
        problems.append(f"ablation cache path-noise mismatch {cache_dir}")
    if manifest.get("randomness", {}).get("probe_basis") != "shared_fixed_seed_v1":
        problems.append(f"ablation cache probe-basis mismatch {cache_dir}")

    expected_response_dim = (
        4
        * expected_config.probe.num_directions
        * len(expected_config.probe.times)
        * (2 if expected_config.probe.antithetic_noise else 1)
    )
    shard_samples = 0
    shard_sample_ids: list[str] = []
    for shard in manifest.get("shards", []):
        shard_path = cache_dir / str(shard.get("path", ""))
        shard_count = int(shard.get("num_samples", 0))
        shard_samples += shard_count
        if not shard_path.is_file():
            problems.append(f"missing ablation cache shard {shard_path}")
            continue
        if shard_path.stat().st_size != int(shard.get("bytes", -1)):
            problems.append(f"ablation cache shard byte mismatch {shard_path}")
        if file_sha256(shard_path) != shard.get("sha256"):
            problems.append(f"ablation cache shard SHA-256 mismatch {shard_path}")
        try:
            features, targets, shard_manifest = load_features(shard_path)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            problems.append(f"invalid ablation cache shard {shard_path}: {error}")
            continue
        if features.state.shape != (shard_count, 256, 14):
            problems.append(f"ablation state shape mismatch {shard_path}")
        if features.response.shape != (shard_count, 256, expected_response_dim):
            problems.append(f"ablation response shape mismatch {shard_path}")
        if features.affinity.shape != (shard_count, 256, 256):
            problems.append(f"ablation affinity shape mismatch {shard_path}")
        if tuple(features.grid_size) != (16, 16):
            problems.append(f"ablation grid mismatch {shard_path}")
        segmentation = targets.get("segmentation")
        if segmentation is None or segmentation.shape != (shard_count, 512, 512):
            problems.append(f"ablation target shape mismatch {shard_path}")
        elif segmentation.dtype != torch.uint8:
            problems.append(f"ablation target dtype mismatch {shard_path}")
        sample_ids = shard_manifest.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != shard_count:
            problems.append(f"ablation shard sample IDs mismatch {shard_path}")
        else:
            shard_sample_ids.extend(str(sample_id) for sample_id in sample_ids)
    if shard_samples != 1449 or len(set(shard_sample_ids)) != 1449:
        problems.append(f"ablation shard coverage mismatch {cache_dir}")
    elif sample_ids_sha256(shard_sample_ids) != manifest.get("sample_ids_sha256"):
        problems.append(f"ablation cache sample-ID hash mismatch {cache_dir}")

    per_sample = report.get("per_sample")
    if not isinstance(per_sample, list) or len(per_sample) != 1449:
        problems.append(f"ablation report sample count mismatch {report_path}")
        per_sample = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for sample in per_sample:
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            problems.append(f"ablation report sample-ID mismatch {report_path}")
            continue
        for metric in ("boundary_average_precision", "pairwise_auroc"):
            try:
                value = float(sample["representations"]["response"][metric])
            except (KeyError, TypeError, ValueError):
                problems.append(f"missing ablation response metric {metric} {report_path}")
                continue
            if not math.isfinite(value):
                problems.append(f"non-finite ablation response metric {metric} {report_path}")
        indexed[sample_id] = sample
    if indexed and sample_ids_sha256(list(indexed)) != manifest.get("sample_ids_sha256"):
        problems.append(f"ablation report/cache sample-ID mismatch {report_path}")
    try:
        audited_cache_identity = cache_identity(cache_dir)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        audited_cache_identity = {}
        problems.append(f"invalid ablation cache identity {cache_dir}: {error}")
    return indexed, {
        "report_path": str(report_path),
        "cache_dir": str(cache_dir),
        "cache_identity": audited_cache_identity,
        "runtime_profile": runtime_identity,
        "runtime": manifest.get("runtime"),
    }, problems


def audit_extension_evidence(
    *,
    main_evidence_path: Path,
    imagenet_matrix_path: Path,
    imagenet_asset_audit_path: Path,
    imagenet_split_audit_path: Path,
    main_runtime_profile_path: Path,
    readout_runtime_profile_path: Path,
    base_voc_report_path: Path,
    base_config_path: Path,
    ablation_registry_path: Path,
    ablation_report_paths: Mapping[str, Path],
    ablation_runtime_profile_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Audit the conditional ImageNet-1k and 15-cache high-cost extension."""

    provenance = code_provenance()
    repository_root = Path(__file__).resolve().parents[2]
    problems: list[str] = []
    if provenance.get("code_dirty") is not False:
        problems.append("extension evidence audit requires a clean code worktree")
    main_evidence = _read_json(main_evidence_path) if main_evidence_path.is_file() else {}
    if main_evidence.get("verdict") != "main_tasks_supported_pending_causal_audits":
        problems.append("extension requires a complete positive main-task verdict")
    for key in ("code_revision", "code_tree_sha256"):
        if main_evidence.get(key) != provenance.get(key):
            problems.append(f"main evidence {key} mismatch")

    runtime_report, runtime_problems = _validate_runtime_profile(
        main_runtime_profile_path,
        provenance,
    )
    problems.extend(runtime_problems)
    expected_runtime_profile = runtime_report.get("identity")
    try:
        readout_runtime_profile = readout_runtime_profile_identity(
            readout_runtime_profile_path
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        readout_runtime_profile = {}
        problems.append(f"invalid readout runtime profile: {error}")
    for key in ("code_revision", "code_tree_sha256"):
        if readout_runtime_profile.get(key) != provenance.get(key):
            problems.append(f"readout runtime profile {key} mismatch")

    asset_audit = (
        _read_json(imagenet_asset_audit_path)
        if imagenet_asset_audit_path.is_file()
        else {}
    )
    if asset_audit.get("status") != "passed":
        problems.append("ImageNet-1k asset audit is missing or failed")
    split_counts = asset_audit.get("fieldscope_split_counts", {})
    if set(split_counts) != {"train", "val", "test"}:
        problems.append("ImageNet-1k asset audit split counts are missing")
        split_counts = {"train": -1, "val": -1, "test": -1}
    split_audit = (
        _read_json(imagenet_split_audit_path)
        if imagenet_split_audit_path.is_file()
        else {}
    )
    if split_audit.get("status") != "passed" or split_audit.get("dataset") != "imagenet":
        problems.append("ImageNet-1k split audit is missing or failed")
    for split, count in split_counts.items():
        if int(split_audit.get("splits", {}).get(split, {}).get("count", -2)) != int(count):
            problems.append(f"ImageNet-1k {split} split count mismatch")
    for source in (asset_audit, split_audit):
        for key in ("code_revision", "code_tree_sha256"):
            if source.get(key) != provenance.get(key):
                problems.append(f"ImageNet-1k audit {key} mismatch")
        if source.get("code_dirty") is not False:
            problems.append("ImageNet-1k audit came from a dirty worktree")

    observations, matrix_report, matrix_problems = _validate_matrix(
        "imagenet",
        imagenet_matrix_path,
        provenance,
        repository_root,
        expected_runtime_profile,
        task_contract=("classification", "top1", True, split_counts),
        readout_budget={"epochs": 90, "batch_size": 128},
    )
    problems.extend(matrix_problems)
    if imagenet_matrix_path.is_file():
        matrix_payload = _read_json(imagenet_matrix_path)
        if matrix_payload.get("readout_runtime_profile") != readout_runtime_profile:
            problems.append("ImageNet-1k matrix readout runtime profile mismatch")
    imagenet_candidates: dict[str, Any] = {}
    if not matrix_problems:
        for candidate in ("response", "full"):
            static = _paired_gain(
                observations[candidate],
                [observations[name] for name in sorted(_STATIC_CONTROLS)],
                maximize=True,
            )
            graph = _paired_gain(
                observations[candidate],
                [observations[name] for name in sorted(_GRAPH_CONTROLS[candidate])],
                maximize=True,
            )
            shuffled = _paired_gain(
                observations[candidate],
                [observations[_SHUFFLED_CONTROL[candidate]]],
                maximize=True,
            )
            imagenet_candidates[candidate] = {
                "static_and_hidden_controls": static,
                "graph_controls": graph,
                "shuffled_response_control": shuffled,
                "passed": all(
                    comparison["stable_positive_gain"]
                    for comparison in (static, graph, shuffled)
                ),
            }

    registry = _read_json(ablation_registry_path) if ablation_registry_path.is_file() else {}
    registered_names = set(HIGH_COST_ABLATION_VARIANTS)
    if (
        registry.get("status") != "passed"
        or int(registry.get("variant_count", -1)) != len(registered_names)
        or {item.get("name") for item in registry.get("variants", [])} != registered_names
    ):
        problems.append("high-cost ablation config registry mismatch")
    for key in ("code_revision", "code_tree_sha256"):
        if registry.get(key) != provenance.get(key):
            problems.append(f"high-cost ablation registry {key} mismatch")
    if registry.get("code_dirty") is not False:
        problems.append("high-cost ablation registry came from a dirty worktree")
    if (
        registry.get("base_config_sha256") != file_sha256(base_config_path)
        or Path(str(registry.get("base_config_path", ""))).resolve()
        != base_config_path.resolve()
    ):
        problems.append("high-cost ablation base config identity mismatch")
    if set(ablation_report_paths) != registered_names:
        problems.append("high-cost ablation report registry mismatch")
    if set(ablation_runtime_profile_paths) != registered_names:
        problems.append("high-cost ablation runtime registry mismatch")

    generated_configs: dict[str, Path] = {}
    for item in registry.get("variants", []):
        name = str(item.get("name", ""))
        config_path = Path(str(item.get("config_path", "")))
        generated_configs[name] = config_path
        if not config_path.is_file() or item.get("config_sha256") != file_sha256(config_path):
            problems.append(f"high-cost ablation config identity mismatch for {name}")
            continue
        field_name, expected_value = HIGH_COST_ABLATION_VARIANTS.get(name, ("", None))
        section, _, key = field_name.partition(".")
        config_mapping = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            not section
            or config_mapping.get(section, {}).get(key) != expected_value
            or item.get("changed_field") != field_name
            or item.get("value") != expected_value
        ):
            problems.append(f"high-cost ablation changed-field mismatch for {name}")
    base_samples, base_source, base_problems = _validate_dense_voc_ablation(
        report_path=base_voc_report_path,
        config_path=base_config_path,
        runtime_profile_path=main_runtime_profile_path,
        provenance=provenance,
        repository_root=repository_root,
    )
    problems.extend(base_problems)
    ablation_sources: dict[str, Any] = {}
    ablation_comparisons: dict[str, Any] = {}
    for name in sorted(registered_names):
        config_path = generated_configs.get(name)
        report_path = ablation_report_paths.get(name)
        profile_path = ablation_runtime_profile_paths.get(name)
        if config_path is None or report_path is None or profile_path is None:
            continue
        samples, source, variant_problems = _validate_dense_voc_ablation(
            report_path=report_path,
            config_path=config_path,
            runtime_profile_path=profile_path,
            provenance=provenance,
            repository_root=repository_root,
        )
        ablation_sources[name] = source
        problems.extend(variant_problems)
        if set(samples) != set(base_samples):
            problems.append(f"high-cost ablation sample IDs do not align for {name}")
            continue
        metric_reports: dict[str, Any] = {}
        for metric in ("boundary_average_precision", "pairwise_auroc"):
            differences = [
                float(samples[sample_id]["representations"]["response"][metric])
                - float(base_samples[sample_id]["representations"]["response"][metric])
                for sample_id in sorted(base_samples)
            ]
            summary = bootstrap_mean_interval(differences, seed=4121, resamples=2000)
            interval = summary["ci95"]
            direction = "mixed"
            if interval is not None and interval[0] > 0:
                direction = "improved"
            elif interval is not None and interval[1] < 0:
                direction = "degraded"
            metric_reports[metric] = {
                **summary,
                "num_images": len(differences),
                "direction": direction,
            }
        ablation_comparisons[name] = metric_reports

    scientific_supported = bool(
        imagenet_candidates
        and any(candidate["passed"] for candidate in imagenet_candidates.values())
    )
    if problems:
        status = "incomplete"
        verdict = "incomplete"
    elif scientific_supported:
        status = "passed"
        verdict = "supported"
    else:
        status = "failed"
        verdict = "mixed_or_negative"
    return {
        "schema_version": 1,
        "status": status,
        "verdict": verdict,
        "paper_evidence": verdict != "incomplete",
        "supports_scaling_extension": verdict == "supported",
        "problems": sorted(set(problems)),
        "main_evidence_path": str(main_evidence_path),
        "imagenet1k": {
            "asset_audit": str(imagenet_asset_audit_path),
            "split_audit": str(imagenet_split_audit_path),
            "matrix": matrix_report,
            "candidates": imagenet_candidates,
            "required_representations": sorted(_REPRESENTATIONS),
            "required_seeds": sorted(_SEEDS),
        },
        "high_cost_ablations": {
            "base": base_source,
            "sources": ablation_sources,
            "comparisons": ablation_comparisons,
            "required_variant_count": len(registered_names),
        },
        "runtime_profile": runtime_report,
        "readout_runtime_profile": readout_runtime_profile,
        **provenance,
    }


def audit_final_evidence(
    *,
    main_evidence_path: Path,
    causal_evidence_path: Path,
    extension_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Combine the main, causal, and conditionally required extension verdicts."""

    provenance = code_provenance()
    problems: list[str] = []
    main = _read_json(main_evidence_path) if main_evidence_path.is_file() else {}
    causal = _read_json(causal_evidence_path) if causal_evidence_path.is_file() else {}
    for name, payload in (("main", main), ("causal", causal)):
        for key in ("code_revision", "code_tree_sha256"):
            if payload.get(key) != provenance.get(key):
                problems.append(f"{name} evidence {key} mismatch")
        if payload.get("code_dirty") is not False:
            problems.append(f"{name} evidence came from a dirty worktree")
    main_verdict = main.get("verdict")
    causal_verdict = causal.get("verdict")
    extension: dict[str, Any] | None = None
    extension_verdict: str | None = None
    if main_verdict == "main_tasks_supported_pending_causal_audits":
        if extension_evidence_path is None or not extension_evidence_path.is_file():
            problems.append("positive main evidence requires extension evidence")
        else:
            extension = _read_json(extension_evidence_path)
            extension_verdict = extension.get("verdict")
            for key in ("code_revision", "code_tree_sha256"):
                if extension.get(key) != provenance.get(key):
                    problems.append(f"extension evidence {key} mismatch")
            if extension.get("code_dirty") is not False:
                problems.append("extension evidence came from a dirty worktree")
            if extension_verdict not in {"supported", "mixed_or_negative"}:
                problems.append("positive main evidence has incomplete extension evidence")
        if causal_verdict not in {
            "supports_core_hypothesis",
            "main_task_gain_not_causally_attributed",
        }:
            problems.append("positive main evidence has incomplete causal evidence")
    elif main_verdict == "limited_or_negative":
        if causal_verdict != "limited_or_negative":
            problems.append("negative main evidence did not preserve the causal verdict")
        if extension_evidence_path is not None:
            problems.append("negative main evidence must not consume conditional extension data")
    else:
        problems.append("main evidence is incomplete or has an unexpected verdict")

    if problems:
        status = "incomplete"
        verdict = "incomplete"
    elif main_verdict == "limited_or_negative":
        status = "failed"
        verdict = "limited_or_negative"
    elif causal_verdict == "main_task_gain_not_causally_attributed":
        status = "failed"
        verdict = "main_task_gain_not_causally_attributed"
    elif extension_verdict == "supported":
        status = "passed"
        verdict = "supports_core_hypothesis_with_scaling_extension"
    else:
        status = "passed"
        verdict = "supports_core_hypothesis_limited_scaling"
    return {
        "schema_version": 1,
        "status": status,
        "verdict": verdict,
        "paper_evidence": verdict != "incomplete",
        "supports_core_hypothesis": verdict.startswith("supports_core_hypothesis"),
        "supports_scaling_extension": (
            verdict == "supports_core_hypothesis_with_scaling_extension"
        ),
        "supports_unqualified_strong_claims": (
            verdict == "supports_core_hypothesis_with_scaling_extension"
        ),
        "problems": sorted(set(problems)),
        "main_evidence": str(main_evidence_path),
        "main_verdict": main_verdict,
        "causal_evidence": str(causal_evidence_path),
        "causal_verdict": causal_verdict,
        "extension_evidence": (
            str(extension_evidence_path) if extension_evidence_path is not None else None
        ),
        "extension_verdict": extension_verdict,
        **provenance,
    }
