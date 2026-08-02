"""Formal evidence audit for the pre-registered FieldScope hypothesis."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from fieldscope.cache import load_features
from fieldscope.cached_dataset import cached_control_contract_for_cache
from fieldscope.config import RunConfig, runtime_profile_identity
from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.experiments import (
    READOUT_TRAINING_COVERAGE_CONTRACT,
    cache_identity,
    code_provenance,
    file_sha256,
)
from fieldscope.readout_runtime_gate import (
    load_readout_runtime_gate_report,
    readout_runtime_profile_identity,
)
from fieldscope.runtime_gate import (
    MAX_MEMORY_FRACTION,
    MIN_SPEEDUP_FRACTION,
    NUM_SYNTHETIC_IMAGES,
    REGISTERED_PROFILES,
)
from fieldscope.statistics import (
    bootstrap_mean_interval,
    holm_adjusted_pvalues,
    paired_t_interval,
)

_SEEDS = {4121, 7319, 104729}
_REPRESENTATIONS = {
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
_TASKS = {
    "imagenet100": ("classification", "top1", True, {"train": 116455, "val": 12940, "test": 5000}),
    "voc2012": ("segmentation", "mean_iou", True, {"train": 1318, "val": 146, "test": 1449}),
    "ade20k": ("segmentation", "mean_iou", True, {"train": 18189, "val": 2021, "test": 2000}),
    "nyuv2": ("depth", "abs_rel", False, {"train": 715, "val": 80, "test": 654}),
}
_READOUT_BUDGETS = {
    "imagenet100": {"epochs": 90, "batch_size": 128},
    "voc2012": {"epochs": 80, "batch_size": 4},
    "ade20k": {"epochs": 80, "batch_size": 2},
    "nyuv2": {"epochs": 80, "batch_size": 4},
}
_LEARNING_RATE = 0.001
_WEIGHT_DECAY = 0.0001
_STATIC_CONTROLS = {
    "random_feature_local",
    "z0",
    "zt",
    "trajectory",
    "velocity",
    "mismatch",
    "endpoint",
    "state",
    "state_graph",
    "dit_hidden_local",
    "dit_hidden_attention",
}
_GRAPH_CONTROLS = {
    "response": {"response_nograph", "response_local"},
    "full": {"full_nograph", "full_local"},
}
_SHUFFLED_CONTROL = {
    "response": "response_shuffled",
    "full": "full_shuffled",
}
_UNSUPERVISED_CONTROLS = {
    "response_shuffled",
    "state",
    "z0",
    "velocity",
    "dit_hidden",
    "dit_attention",
}
_GRAPH_DIAGNOSTIC_REPRESENTATIONS = {
    "response",
    "response_shuffled",
    "state",
    "dit_attention",
    "mismatch",
    "velocity",
    "zt",
    "trajectory",
    "z0",
    "endpoint",
    "dit_hidden",
}
_GRAPH_DECISION_METRICS = ("boundary_average_precision", "pairwise_auroc")
_MIN_STRUCTURAL_ELIGIBLE_FRACTION = 0.95
_CAUSAL_VARIANTS = {
    "random_flow": {
        "probe_type": "structured",
        "prompt": "",
        "random_transformer": True,
    },
    "spatially_shuffled_probe": {
        "probe_type": "spatially_shuffled",
        "prompt": "",
        "random_transformer": False,
    },
    "neutral_prompt": {
        "probe_type": "structured",
        "prompt": "a neutral photograph",
        "random_transformer": False,
    },
    "unrelated_prompt": {
        "probe_type": "structured",
        "prompt": "an unrelated scene",
        "random_transformer": False,
    },
}
_CONDITION_CONTROLS = {
    "response_shuffled",
    "state",
    "z0",
    "velocity",
    "dit_hidden",
    "dit_attention",
}
_BASELINE_NAMES = {"z0", "zt", "velocity", "mismatch", "endpoint", "dit_hidden", "trajectory"}
_GRAPH_NAMES_BY_POLICY = {
    "dense": {"dit_attention", "dit_attention_adjacency"},
    "readout_sparse": {"dit_attention_adjacency"},
}


def _structural_metric_eligibility(
    per_sample: list[Mapping[str, Any]],
    metric: str,
    *,
    source: str,
) -> tuple[set[str], dict[str, Any], list[str]]:
    """Separate target-undefined metrics from representation-specific failures."""

    problems: list[str] = []
    eligible: set[str] = set()
    excluded: set[str] = set()
    mixed: set[str] = set()
    for sample in per_sample:
        sample_id = str(sample.get("sample_id", ""))
        representations = sample.get("representations")
        if not isinstance(representations, Mapping):
            problems.append(
                f"missing graph representations source={source} sample_id={sample_id}"
            )
            continue
        if set(representations) != _GRAPH_DIAGNOSTIC_REPRESENTATIONS:
            problems.append(
                "graph representation registry mismatch "
                f"source={source} sample_id={sample_id}"
            )
            continue
        finite: list[bool] = []
        try:
            for representation in sorted(_GRAPH_DIAGNOSTIC_REPRESENTATIONS):
                value = float(representations[representation][metric])
                finite.append(math.isfinite(value))
        except (KeyError, TypeError, ValueError):
            problems.append(
                f"missing graph metric source={source} metric={metric} "
                f"sample_id={sample_id}"
            )
            continue
        if all(finite):
            eligible.add(sample_id)
        elif any(finite):
            mixed.add(sample_id)
            problems.append(
                "mixed finite/non-finite graph metric "
                f"source={source} metric={metric} sample_id={sample_id}"
            )
        else:
            excluded.add(sample_id)
    total = len(per_sample)
    retained_fraction = len(eligible) / total if total else 0.0
    if retained_fraction < _MIN_STRUCTURAL_ELIGIBLE_FRACTION:
        problems.append(
            "graph metric retained fraction below registered minimum "
            f"source={source} metric={metric} retained={retained_fraction:.6f} "
            f"minimum={_MIN_STRUCTURAL_ELIGIBLE_FRACTION:.6f}"
        )
    audit = {
        "metric": metric,
        "total_samples": total,
        "eligible_samples": len(eligible),
        "structurally_undefined_samples": len(excluded),
        "mixed_finiteness_samples": len(mixed),
        "retained_fraction": retained_fraction,
        "minimum_retained_fraction": _MIN_STRUCTURAL_ELIGIBLE_FRACTION,
        "eligible_sample_ids_sha256": sample_ids_sha256(sorted(eligible)),
        "structurally_undefined_sample_ids_sha256": sample_ids_sha256(
            sorted(excluded)
        ),
        "mixed_finiteness_sample_ids_sha256": sample_ids_sha256(sorted(mixed)),
        "all_or_none_finiteness_passed": not mixed,
        "status": "passed" if not problems else "failed",
    }
    return eligible, audit, problems


def _validate_runtime_profile(
    runtime_profile_path: Path,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    if not runtime_profile_path.is_file():
        return {"path": str(runtime_profile_path)}, [f"missing {runtime_profile_path}"]
    payload = _read_json(runtime_profile_path)
    try:
        identity = runtime_profile_identity(runtime_profile_path)
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {"path": str(runtime_profile_path)}, [f"invalid runtime profile: {error}"]
    if payload.get("evidence_scope") != "label_free_runtime_equivalence_and_throughput_only":
        problems.append("runtime profile evidence scope mismatch")
    if payload.get("method_effectiveness_conclusion") is not None:
        problems.append("runtime profile contains a method-effectiveness conclusion")
    if payload.get("code_revision") != provenance["code_revision"]:
        problems.append("runtime profile revision mismatch")
    if payload.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append("runtime profile code tree mismatch")
    if payload.get("code_dirty") is not False:
        problems.append("runtime profile came from a dirty worktree")
    if payload.get("equivalence_rule") != "torch.equal for every cached tensor field":
        problems.append("runtime profile equivalence rule mismatch")
    expected_profiles = [
        {"image_batch_size": image, "probe_batch_size": probe}
        for image, probe in REGISTERED_PROFILES
    ]
    if payload.get("registered_profiles") != expected_profiles:
        problems.append("runtime profile registered candidates mismatch")
    if payload.get("num_synthetic_images") != NUM_SYNTHETIC_IMAGES:
        problems.append("runtime profile synthetic image count mismatch")
    if payload.get("synthetic_image_policy") != "fixed_seed_random_plus_coordinate_ramps_v1":
        problems.append("runtime profile synthetic image policy mismatch")
    if payload.get("minimum_speedup_fraction") != MIN_SPEEDUP_FRACTION:
        problems.append("runtime profile speedup threshold mismatch")
    if payload.get("maximum_cuda_reserved_fraction") != MAX_MEMORY_FRACTION:
        problems.append("runtime profile memory threshold mismatch")
    warmup = payload.get("warmup", {})
    if (
        warmup.get("profile") != {"image_batch_size": 2, "probe_batch_size": 8}
        or warmup.get("num_images") != 2
        or warmup.get("status") != "completed"
    ):
        problems.append("runtime profile warm-up contract mismatch")
    backend = payload.get("backend", {})
    if (
        backend.get("backend") != "auraflow"
        or backend.get("variant") != "fp16"
        or backend.get("device") != "cuda"
        or backend.get("dtype") != "bfloat16"
        or backend.get("image_size") != 512
        or backend.get("random_transformer") is not False
        or backend.get("frozen") is not True
    ):
        problems.append("runtime profile backend contract mismatch")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        problems.append("runtime profile candidate matrix mismatch")
    else:
        observed_profiles = [
            {
                "image_batch_size": candidate.get("image_batch_size"),
                "probe_batch_size": candidate.get("probe_batch_size"),
            }
            for candidate in candidates
        ]
        if observed_profiles != expected_profiles:
            problems.append("runtime profile candidate order mismatch")
        baseline_seconds = candidates[0].get("seconds_per_image")
        eligible: list[Mapping[str, Any]] = []
        if (
            not isinstance(baseline_seconds, (int, float))
            or not math.isfinite(baseline_seconds)
            or baseline_seconds <= 0
        ):
            problems.append("runtime profile baseline timing is invalid")
        else:
            baseline_feature_sha256 = candidates[0].get("feature_sha256")
            for index, candidate in enumerate(candidates):
                seconds = candidate.get("seconds_per_image")
                completed = candidate.get("status") == "completed"
                timing_valid = bool(
                    isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0
                )
                exact = candidate.get("equivalence", {}).get("exact") is True
                if exact and candidate.get("feature_sha256") != baseline_feature_sha256:
                    problems.append(
                        f"runtime profile exact candidate hash mismatch candidate={index}"
                    )
                fields = candidate.get("equivalence", {}).get("fields", {})
                if (
                    exact
                    and fields
                    and not all(field.get("exact") is True for field in fields.values())
                ):
                    problems.append(f"runtime profile exact field mismatch candidate={index}")
                fraction = candidate.get("cuda_peak_reserved_fraction")
                memory_safe = bool(
                    completed
                    and isinstance(fraction, (int, float))
                    and math.isfinite(fraction)
                    and fraction <= MAX_MEMORY_FRACTION
                )
                speedup = baseline_seconds / seconds - 1.0 if completed and timing_valid else None
                expected_eligible = bool(
                    completed
                    and timing_valid
                    and exact
                    and memory_safe
                    and (index == 0 or speedup >= MIN_SPEEDUP_FRACTION)
                )
                if candidate.get("memory_safe") is not memory_safe:
                    problems.append(f"runtime profile memory decision mismatch candidate={index}")
                recorded_speedup = candidate.get("speedup_fraction_vs_baseline")
                if speedup is None:
                    if recorded_speedup is not None:
                        problems.append(f"runtime profile speedup mismatch candidate={index}")
                elif not isinstance(recorded_speedup, (int, float)) or not math.isclose(
                    recorded_speedup,
                    speedup,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    problems.append(f"runtime profile speedup mismatch candidate={index}")
                if candidate.get("eligible") is not expected_eligible:
                    problems.append(f"runtime profile eligibility mismatch candidate={index}")
                if expected_eligible:
                    eligible.append(candidate)
        selected = payload.get("selected_profile", {})
        if eligible:
            expected_selected = min(
                eligible,
                key=lambda candidate: float(candidate["seconds_per_image"]),
            )
            for key in (
                "image_batch_size",
                "probe_batch_size",
                "seconds_per_image",
                "speedup_fraction_vs_baseline",
                "cuda_peak_reserved_fraction",
                "feature_sha256",
            ):
                if selected.get(key) != expected_selected.get(key):
                    problems.append(f"runtime profile selected candidate mismatch field={key}")
        else:
            problems.append("runtime profile has no eligible baseline")
    return {"path": str(runtime_profile_path), "identity": identity, "report": payload}, problems


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _integer_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _resolve_report_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repository_root / path


def _paired_gain(
    candidate: Mapping[int, float],
    controls: list[Mapping[int, float]],
    *,
    maximize: bool,
) -> dict[str, Any]:
    seeds = sorted(candidate)
    best_control = {
        seed: (max if maximize else min)(control[seed] for control in controls) for seed in seeds
    }
    oriented_candidate = (
        dict(candidate) if maximize else {seed: -value for seed, value in candidate.items()}
    )
    oriented_control = (
        best_control if maximize else {seed: -value for seed, value in best_control.items()}
    )
    summary = paired_t_interval(oriented_candidate, oriented_control)
    positive_seeds = sum(value > 0 for value in summary["differences"])
    interval = summary["ci95"]
    stable = bool(
        positive_seeds >= 2 and summary["mean"] > 0 and interval is not None and interval[0] > 0
    )
    return {
        **summary,
        "positive_seeds": positive_seeds,
        "minimum_positive_seeds": 2,
        "stable_positive_gain": stable,
        "best_control_by_seed": best_control,
    }


def _validate_cache_manifest(
    cache_dir: Path,
    *,
    expected_dataset: str,
    expected_split: str,
    expected_count: int,
    provenance: Mapping[str, Any],
    storage_policy: str = "readout_sparse",
    expected_probe_type: str = "structured",
    expected_prompt: str = "",
    expected_random_transformer: bool = False,
    expected_runtime_profile: Mapping[str, Any] | None = None,
) -> list[str]:
    problems: list[str] = []
    manifest_path = cache_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]
    manifest = _read_json(manifest_path)
    if manifest.get("complete") is not True:
        problems.append(f"incomplete cache {cache_dir}")
    if manifest.get("dataset") != expected_dataset or manifest.get("split") != expected_split:
        problems.append(f"cache identity mismatch {cache_dir}")
    if int(manifest.get("num_samples", -1)) != expected_count:
        problems.append(f"cache count mismatch {cache_dir}")
    if not isinstance(manifest.get("sample_ids_sha256"), str) or not manifest.get(
        "sample_ids_sha256"
    ):
        problems.append(f"cache sample-ID SHA-256 missing {cache_dir}")
    if manifest.get("storage_policy") != storage_policy:
        problems.append(f"cache storage policy mismatch {cache_dir}")
    if manifest.get("code_revision") != provenance["code_revision"]:
        problems.append(f"cache revision mismatch {cache_dir}")
    if manifest.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append(f"cache code tree mismatch {cache_dir}")
    if manifest.get("code_dirty") is not False:
        problems.append(f"cache was produced from a dirty worktree {cache_dir}")
    if manifest.get("runtime_profile") != expected_runtime_profile:
        problems.append(f"cache runtime profile mismatch {cache_dir}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        problems.append(f"cache has no shards {cache_dir}")
        shards = []
    shard_samples = 0
    shard_sample_ids: list[str] = []
    expected_target = {
        "imagenet": "classification",
        "imagenet100": "classification",
        "voc2012": "segmentation",
        "ade20k": "segmentation",
        "nyuv2": "depth",
    }.get(expected_dataset)
    for shard in shards:
        shard_samples += int(shard.get("num_samples", 0))
        shard_path = cache_dir / str(shard.get("path", ""))
        expected_bytes = int(shard.get("bytes", -1))
        if not shard_path.is_file() or shard_path.stat().st_size != expected_bytes:
            problems.append(f"cache shard missing or truncated {shard_path}")
        expected_sha256 = shard.get("sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            problems.append(f"cache shard SHA-256 missing {shard_path}")
        elif shard_path.is_file() and file_sha256(shard_path) != expected_sha256:
            problems.append(f"cache shard SHA-256 mismatch {shard_path}")
        if not shard_path.is_file():
            continue
        try:
            features, targets, shard_manifest = load_features(shard_path)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            problems.append(f"cache shard payload invalid {shard_path}: {error}")
            continue
        shard_count = int(shard.get("num_samples", 0))
        if features.state.shape[0] != shard_count:
            problems.append(f"cache shard tensor sample count mismatch {shard_path}")
        if shard_manifest.get("fingerprint") != shard.get("fingerprint"):
            problems.append(f"cache shard fingerprint mismatch {shard_path}")
        if shard_manifest.get("storage_policy") != storage_policy:
            problems.append(f"cache shard storage policy mismatch {shard_path}")
        metadata = shard_manifest.get("metadata", {})
        if metadata.get("backend") != manifest.get("backend"):
            problems.append(f"cache shard backend metadata mismatch {shard_path}")
        if _json_compatible(metadata.get("probe")) != manifest.get("config", {}).get("probe"):
            problems.append(f"cache shard probe metadata mismatch {shard_path}")
        if tuple(features.grid_size) != (16, 16):
            problems.append(f"cache shard grid mismatch {shard_path}")
        if features.state.shape[-1] != 14 or features.response.shape[-1] != 192:
            problems.append(f"cache shard feature dimensions mismatch {shard_path}")
        if features.state.dtype != torch.bfloat16 or features.response.dtype != torch.bfloat16:
            problems.append(f"cache shard feature dtype mismatch {shard_path}")
        if set(features.baselines) != _BASELINE_NAMES:
            problems.append(f"cache shard baselines mismatch {shard_path}")
        elif any(value.dtype != torch.bfloat16 for value in features.baselines.values()):
            problems.append(f"cache shard baseline dtype mismatch {shard_path}")
        if set(features.graphs) != _GRAPH_NAMES_BY_POLICY[storage_policy]:
            problems.append(f"cache shard graph fields mismatch {shard_path}")
        if features.baselines.get("dit_hidden", torch.empty(0)).shape[-1:] != (768,):
            problems.append(f"cache shard DiT hidden dimension mismatch {shard_path}")
        if expected_target is not None and set(targets) != {expected_target}:
            problems.append(f"cache shard target fields mismatch {shard_path}")
        elif expected_target is not None and targets[expected_target].shape[0] != shard_count:
            problems.append(f"cache shard target sample count mismatch {shard_path}")
        elif expected_target is not None:
            expected_dtype = {
                "classification": torch.int32,
                "segmentation": torch.uint8,
                "depth": torch.float32,
            }[expected_target]
            if targets[expected_target].dtype != expected_dtype:
                problems.append(f"cache shard target dtype mismatch {shard_path}")
            expected_shape = {
                "classification": (shard_count,),
                "segmentation": (shard_count, 512, 512),
                "depth": (shard_count, 1, 512, 512),
            }[expected_target]
            if tuple(targets[expected_target].shape) != expected_shape:
                problems.append(f"cache shard target shape mismatch {shard_path}")
        sample_ids = shard_manifest.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != int(shard.get("num_samples", 0)):
            problems.append(f"cache shard sample IDs mismatch {shard_path}")
        else:
            shard_sample_ids.extend(str(sample_id) for sample_id in sample_ids)
    if shard_samples != expected_count:
        problems.append(f"cache shard sample count mismatch {cache_dir}")
    if len(set(shard_sample_ids)) != len(shard_sample_ids):
        problems.append(f"cache shard sample IDs are duplicated {cache_dir}")
    elif shard_sample_ids and sample_ids_sha256(shard_sample_ids) != manifest.get(
        "sample_ids_sha256"
    ):
        problems.append(f"cache shard sample-ID SHA-256 mismatch {cache_dir}")
    backend = manifest.get("backend", {})
    if backend.get("backend") != "auraflow" or backend.get("variant") != "fp16":
        problems.append(f"cache backend identity mismatch {cache_dir}")
    if backend.get("frozen") is not True:
        problems.append(f"backend is not frozen {cache_dir}")
    if backend.get("random_transformer") is not expected_random_transformer:
        problems.append(f"cache random-transformer mismatch {cache_dir}")
    if expected_random_transformer and backend.get("random_transformer_seed") != 104729:
        problems.append(f"cache random-transformer seed mismatch {cache_dir}")
    if backend.get("prompt") != expected_prompt:
        problems.append(f"cache prompt mismatch {cache_dir}")
    if (
        backend.get("device") != "cuda"
        or backend.get("dtype") != "bfloat16"
        or backend.get("image_size") != 512
        or Path(str(backend.get("model_id", ""))).name != "AuraFlow-v0.3"
        or backend.get("native_time") != "1=noise, 0=image"
        or backend.get("public_time") != "clean_time: 0=noise, 1=image"
        or backend.get("velocity_conversion") != "public_velocity=-native_transformer_output"
    ):
        problems.append(f"cache backend runtime contract mismatch {cache_dir}")
    config = manifest.get("config", {})
    backend_config = config.get("backend", {})
    probe = config.get("probe", {})
    runtime = config.get("runtime", {})
    tokenizer = config.get("tokenizer", {})
    selected_runtime = (
        expected_runtime_profile.get("selected_profile", {})
        if expected_runtime_profile is not None
        else {"image_batch_size": 2, "probe_batch_size": 8}
    )
    if (
        backend_config.get("name") != "auraflow"
        or Path(str(backend_config.get("model_path", ""))).name != "AuraFlow-v0.3"
        or backend_config.get("variant") != "fp16"
        or backend_config.get("device") != "cuda"
        or backend_config.get("dtype") != "bfloat16"
        or backend_config.get("image_size") != 512
        or backend_config.get("prompt") != expected_prompt
        or backend_config.get("max_sequence_length") != 256
        or backend_config.get("local_files_only") is not True
        or backend_config.get("offload_text_encoder") is not True
        or backend_config.get("random_transformer") is not expected_random_transformer
        or (expected_random_transformer and backend_config.get("random_transformer_seed") != 104729)
    ):
        problems.append(f"cache backend config mismatch {cache_dir}")
    if probe.get("times") != [0.2, 0.5, 0.8]:
        problems.append(f"cache probe times mismatch {cache_dir}")
    if probe.get("num_directions") != 8:
        problems.append(f"cache probe directions mismatch {cache_dir}")
    if (
        probe.get("difference") != "central"
        or probe.get("probe_type") != expected_probe_type
        or probe.get("eta") != 0.03
        or probe.get("graph_grid") != [16, 16]
        or probe.get("topk") != 16
        or probe.get("local_radius") != 1
        or probe.get("probe_batch_size") != selected_runtime.get("probe_batch_size")
        or probe.get("antithetic_noise") is not True
        or probe.get("hidden_baseline_dim") != 768
        or probe.get("seed") != 4121
    ):
        problems.append(f"cache probe contract mismatch {cache_dir}")
    if (
        tokenizer.get("hidden_dim") != 256
        or tokenizer.get("input_dim") != 768
        or tokenizer.get("num_layers") != 3
        or tokenizer.get("dropout") != 0.1
    ):
        problems.append(f"cache tokenizer contract mismatch {cache_dir}")
    if (
        runtime.get("batch_size") != selected_runtime.get("image_batch_size")
        or runtime.get("cache_shard_size") != 64
        or runtime.get("deterministic") is not True
    ):
        problems.append(f"cache runtime contract mismatch {cache_dir}")
    randomness = manifest.get("randomness", {})
    if randomness.get("path_noise") != "sample_id_sha256_seeded_v1":
        problems.append(f"cache path noise mismatch {cache_dir}")
    if randomness.get("probe_basis") != "shared_fixed_seed_v1":
        problems.append(f"cache probe basis mismatch {cache_dir}")
    return problems


def _validate_matrix(
    dataset: str,
    path: Path,
    provenance: Mapping[str, Any],
    repository_root: Path,
    expected_runtime_profile: Mapping[str, Any] | None = None,
    *,
    task_contract: tuple[str, str, bool, dict[str, int]] | None = None,
    readout_budget: dict[str, int] | None = None,
) -> tuple[dict[str, dict[int, float]], dict[str, Any], list[str]]:
    task, metric, maximize, expected_counts = task_contract or _TASKS[dataset]
    problems: list[str] = []
    if not path.is_file():
        return {}, {"dataset": dataset, "path": str(path)}, [f"missing {path}"]
    report = _read_json(path)
    if report.get("status") != "passed":
        problems.append(f"matrix is not passed {path}")
    if report.get("code_revision") != provenance["code_revision"]:
        problems.append(f"matrix revision mismatch {path}")
    if report.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append(f"matrix code tree mismatch {path}")
    if report.get("code_dirty") is not False:
        problems.append(f"matrix was produced from a dirty worktree {path}")
    if report.get("task") != task or report.get("metric") != metric:
        problems.append(f"matrix task/metric mismatch {path}")
    if set(report.get("seeds", [])) != _SEEDS:
        problems.append(f"matrix seeds mismatch {path}")
    if set(report.get("representations", [])) != _REPRESENTATIONS:
        problems.append(f"matrix representations mismatch {path}")
    expected_budget = readout_budget or _READOUT_BUDGETS[dataset]
    if int(report.get("epochs", -1)) != expected_budget["epochs"]:
        problems.append(f"matrix epoch budget mismatch {path}")
    if int(report.get("batch_size", -1)) != expected_budget["batch_size"]:
        problems.append(f"matrix batch size mismatch {path}")
    if float(report.get("learning_rate", float("nan"))) != _LEARNING_RATE:
        problems.append(f"matrix learning rate mismatch {path}")
    if float(report.get("weight_decay", float("nan"))) != _WEIGHT_DECAY:
        problems.append(f"matrix weight decay mismatch {path}")

    observations: dict[str, dict[int, float]] = {}
    parameter_counts: set[int] = set()
    seen: set[tuple[str, int]] = set()
    cache_dirs = {
        "train": _resolve_report_path(str(report.get("train_cache_dir", "")), repository_root),
        "validation": _resolve_report_path(str(report.get("val_cache_dir", "")), repository_root),
        "test": _resolve_report_path(str(report.get("test_cache_dir", "")), repository_root),
    }
    expected_control_contracts: dict[tuple[str, int, str], dict[str, Any]] = {}
    completed_optimizer_steps = 0
    completed_training_sample_exposures = 0

    def expected_control_contract(
        representation: str,
        seed: int,
        split: str,
    ) -> dict[str, Any]:
        key = (representation, seed, split)
        if key in expected_control_contracts:
            return expected_control_contracts[key]
        contract = cached_control_contract_for_cache(cache_dirs[split], representation, seed)
        expected_control_contracts[key] = contract
        return contract

    for run in report.get("runs", []):
        representation = str(run.get("representation"))
        seed = int(run.get("seed", -1))
        key = (representation, seed)
        if key in seen:
            problems.append(f"duplicate matrix run {dataset}/{representation}/{seed}")
            continue
        seen.add(key)
        try:
            value = float(run["test_metric"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"missing test metric {dataset}/{representation}/{seed}")
            continue
        if not math.isfinite(value):
            problems.append(f"non-finite test metric {dataset}/{representation}/{seed}")
        observations.setdefault(representation, {})[seed] = value
        training_checkpoint: Path | None = None
        training_checkpoint_sha256: str | None = None
        for report_kind in ("training_report", "test_report"):
            source = _resolve_report_path(str(run.get(report_kind, "")), repository_root)
            if not source.is_file():
                problems.append(f"missing {report_kind} {source}")
                continue
            payload = _read_json(source)
            if payload.get("status") != "passed":
                problems.append(f"{report_kind} is not passed {source}")
            if payload.get("code_revision") != provenance["code_revision"]:
                problems.append(f"{report_kind} revision mismatch {source}")
            if payload.get("code_tree_sha256") != provenance["code_tree_sha256"]:
                problems.append(f"{report_kind} code tree mismatch {source}")
            if payload.get("code_dirty") is not False:
                problems.append(f"{report_kind} was produced from a dirty worktree {source}")
            if payload.get("task") != task or payload.get("representation") != representation:
                problems.append(f"{report_kind} identity mismatch {source}")
            if int(payload.get("seed", -1)) != seed:
                problems.append(f"{report_kind} seed mismatch {source}")
            if report_kind == "training_report":
                if payload.get("config") != report.get("config"):
                    problems.append(f"training config mismatch {source}")
                if int(payload.get("epochs", -1)) != expected_budget["epochs"]:
                    problems.append(f"training epoch budget mismatch {source}")
                if int(payload.get("batch_size", -1)) != expected_budget["batch_size"]:
                    problems.append(f"training batch size mismatch {source}")
                if float(payload.get("learning_rate", float("nan"))) != _LEARNING_RATE:
                    problems.append(f"training learning rate mismatch {source}")
                if float(payload.get("weight_decay", float("nan"))) != _WEIGHT_DECAY:
                    problems.append(f"training weight decay mismatch {source}")
                if payload.get("train_cache") != report.get("train_cache"):
                    problems.append(f"training train-cache identity mismatch {source}")
                if payload.get("validation_cache") != report.get("validation_cache"):
                    problems.append(f"training validation-cache identity mismatch {source}")
                if payload.get("train_control_contract") != expected_control_contract(
                    representation, seed, "train"
                ):
                    problems.append(f"training control contract mismatch {source}")
                if payload.get("validation_control_contract") != expected_control_contract(
                    representation, seed, "validation"
                ):
                    problems.append(f"validation control contract mismatch {source}")
                train_samples = int(payload.get("train_samples", -1))
                payload_train_cache = payload.get("train_cache")
                cache_train_samples = (
                    int(payload_train_cache.get("num_samples", -2))
                    if isinstance(payload_train_cache, Mapping)
                    else -2
                )
                if train_samples < 1 or train_samples != cache_train_samples:
                    problems.append(f"training sample-count contract mismatch {source}")
                expected_steps_per_epoch = (
                    math.ceil(train_samples / expected_budget["batch_size"])
                    if train_samples > 0
                    else -1
                )
                if (
                    payload.get("training_coverage_contract")
                    != READOUT_TRAINING_COVERAGE_CONTRACT
                ):
                    problems.append(f"training coverage contract mismatch {source}")
                if (
                    int(payload.get("optimizer_steps_per_epoch", -1))
                    != expected_steps_per_epoch
                ):
                    problems.append(f"training per-epoch optimizer-step mismatch {source}")
                history = payload.get("history")
                if not isinstance(history, list) or len(history) != expected_budget["epochs"]:
                    problems.append(f"training history epoch coverage mismatch {source}")
                    history = []
                history_optimizer_steps = 0
                history_sample_exposures = 0
                for epoch_index, entry in enumerate(history, start=1):
                    if not isinstance(entry, Mapping):
                        problems.append(f"training history entry is invalid {source}")
                        continue
                    if int(entry.get("epoch", -1)) != epoch_index:
                        problems.append(f"training history epoch sequence mismatch {source}")
                    if int(entry.get("train_samples", -1)) != train_samples:
                        problems.append(f"training epoch sample coverage mismatch {source}")
                    if int(entry.get("optimizer_steps", -1)) != expected_steps_per_epoch:
                        problems.append(f"training epoch optimizer-step mismatch {source}")
                    history_optimizer_steps += int(entry.get("optimizer_steps", 0))
                    history_sample_exposures += int(entry.get("train_samples", 0))
                expected_optimizer_steps = (
                    expected_steps_per_epoch * expected_budget["epochs"]
                )
                expected_sample_exposures = train_samples * expected_budget["epochs"]
                if (
                    history_optimizer_steps != expected_optimizer_steps
                    or int(payload.get("completed_optimizer_steps", -1))
                    != expected_optimizer_steps
                ):
                    problems.append(f"training completed optimizer-step mismatch {source}")
                if (
                    history_sample_exposures != expected_sample_exposures
                    or int(payload.get("completed_training_sample_exposures", -1))
                    != expected_sample_exposures
                ):
                    problems.append(f"training completed sample-exposure mismatch {source}")
                completed_optimizer_steps += max(0, expected_optimizer_steps)
                completed_training_sample_exposures += max(0, expected_sample_exposures)
                if int(run.get("completed_optimizer_steps", -1)) != expected_optimizer_steps:
                    problems.append(f"matrix run optimizer-step mismatch {source}")
                if (
                    int(run.get("completed_training_sample_exposures", -1))
                    != expected_sample_exposures
                ):
                    problems.append(f"matrix run sample-exposure mismatch {source}")
                last_checkpoint = _resolve_report_path(
                    str(payload.get("last_checkpoint", "")),
                    repository_root,
                )
                if not last_checkpoint.is_file():
                    problems.append(f"missing last checkpoint {last_checkpoint}")
                else:
                    last_checkpoint_sha256 = file_sha256(last_checkpoint)
                    if payload.get("last_checkpoint_sha256") != last_checkpoint_sha256:
                        problems.append(f"last checkpoint SHA-256 mismatch {last_checkpoint}")
                    try:
                        last_payload = torch.load(
                            last_checkpoint,
                            map_location="cpu",
                            weights_only=False,
                        )
                    except Exception as error:  # noqa: BLE001 - artifact audit boundary
                        problems.append(f"invalid last checkpoint {last_checkpoint}: {error}")
                    else:
                        if not isinstance(last_payload, Mapping):
                            problems.append(f"invalid last checkpoint payload {last_checkpoint}")
                            last_payload = {}
                        checkpoint_contract = {
                            "epoch": expected_budget["epochs"],
                            "target_epochs": expected_budget["epochs"],
                            "batch_size": expected_budget["batch_size"],
                            "training_coverage_contract": READOUT_TRAINING_COVERAGE_CONTRACT,
                            "optimizer_steps_per_epoch": expected_steps_per_epoch,
                            "completed_optimizer_steps": expected_optimizer_steps,
                            "completed_training_sample_exposures": expected_sample_exposures,
                            "history": history,
                            "train_cache": payload.get("train_cache"),
                            "validation_cache": payload.get("validation_cache"),
                            "code_revision": provenance["code_revision"],
                            "code_tree_sha256": provenance["code_tree_sha256"],
                        }
                        for name, expected in checkpoint_contract.items():
                            if last_payload.get(name) != expected:
                                problems.append(
                                    f"last checkpoint {name} contract mismatch {last_checkpoint}"
                                )
                        optimizer_payload = last_payload.get("optimizer")
                        optimizer_states = (
                            optimizer_payload.get("state", {})
                            if isinstance(optimizer_payload, Mapping)
                            else {}
                        )
                        try:
                            optimizer_state_steps = {
                                int(state["step"].item())
                                if isinstance(state.get("step"), torch.Tensor)
                                else int(state["step"])
                                for state in optimizer_states.values()
                                if isinstance(state, Mapping) and "step" in state
                            }
                        except (TypeError, ValueError):
                            optimizer_state_steps = set()
                        if optimizer_state_steps != {expected_optimizer_steps}:
                            problems.append(
                                "last checkpoint AdamW step-state mismatch "
                                f"{last_checkpoint}"
                            )
                try:
                    parameter_counts.add(int(payload["trainable_parameters"]))
                except (KeyError, TypeError, ValueError):
                    problems.append(f"missing trainable parameter count {source}")
                checkpoint = _resolve_report_path(
                    str(payload.get("best_checkpoint", "")),
                    repository_root,
                )
                if not checkpoint.is_file():
                    problems.append(f"missing best checkpoint {checkpoint}")
                else:
                    checkpoint_sha256 = file_sha256(checkpoint)
                    training_checkpoint = checkpoint.resolve()
                    training_checkpoint_sha256 = checkpoint_sha256
                    if payload.get("best_checkpoint_sha256") != checkpoint_sha256:
                        problems.append(f"best checkpoint SHA-256 mismatch {checkpoint}")
                    run_checkpoint = _resolve_report_path(
                        str(run.get("best_checkpoint", "")),
                        repository_root,
                    )
                    if run_checkpoint.resolve() != training_checkpoint:
                        problems.append(f"matrix best-checkpoint path mismatch {checkpoint}")
                    if run.get("best_checkpoint_sha256") != checkpoint_sha256:
                        problems.append(f"matrix checkpoint SHA-256 mismatch {checkpoint}")
            else:
                if payload.get("config") != report.get("config"):
                    problems.append(f"test config mismatch {source}")
                if int(payload.get("batch_size", -1)) != expected_budget["batch_size"]:
                    problems.append(f"test batch size mismatch {source}")
                if payload.get("test_cache") != report.get("test_cache"):
                    problems.append(f"test cache identity mismatch {source}")
                if payload.get("test_control_contract") != expected_control_contract(
                    representation, seed, "test"
                ):
                    problems.append(f"test control contract mismatch {source}")
                checkpoint = _resolve_report_path(
                    str(payload.get("checkpoint", "")),
                    repository_root,
                )
                if not checkpoint.is_file():
                    problems.append(f"missing test checkpoint {checkpoint}")
                else:
                    checkpoint_sha256 = file_sha256(checkpoint)
                    if payload.get("checkpoint_sha256") != checkpoint_sha256:
                        problems.append(f"test checkpoint SHA-256 mismatch {checkpoint}")
                    if (
                        training_checkpoint is not None
                        and checkpoint.resolve() != training_checkpoint
                    ):
                        problems.append(f"test checkpoint path mismatch {checkpoint}")
                    if (
                        training_checkpoint_sha256 is not None
                        and checkpoint_sha256 != training_checkpoint_sha256
                    ):
                        problems.append(f"test checkpoint differs from training {checkpoint}")
                try:
                    report_metric = float(payload["evaluation"]["metrics"][metric])
                except (KeyError, TypeError, ValueError):
                    problems.append(f"missing metric in test report {source}")
                else:
                    if not math.isfinite(report_metric) or report_metric != value:
                        problems.append(f"test metric mismatch {source}")
    expected_runs = {
        (representation, seed) for representation in _REPRESENTATIONS for seed in _SEEDS
    }
    missing_runs = sorted(expected_runs - seen)
    if missing_runs:
        problems.append(f"matrix missing {len(missing_runs)} runs {path}")
    unexpected_runs = sorted(seen - expected_runs)
    if unexpected_runs:
        problems.append(f"matrix has {len(unexpected_runs)} unexpected runs {path}")
    if len(parameter_counts) != 1:
        problems.append(f"readout parameter counts are not matched {path}")
    if int(report.get("completed_optimizer_steps", -1)) != completed_optimizer_steps:
        problems.append(f"matrix completed optimizer-step total mismatch {path}")
    if (
        int(report.get("completed_training_sample_exposures", -1))
        != completed_training_sample_exposures
    ):
        problems.append(f"matrix completed sample-exposure total mismatch {path}")
    for split, path_key, identity_key in (
        ("train", "train_cache_dir", "train_cache"),
        ("val", "val_cache_dir", "validation_cache"),
        ("test", "test_cache_dir", "test_cache"),
    ):
        cache_dir = _resolve_report_path(str(report.get(path_key, "")), repository_root)
        problems.extend(
            _validate_cache_manifest(
                cache_dir,
                expected_dataset=dataset,
                expected_split=split,
                expected_count=expected_counts[split],
                provenance=provenance,
                expected_runtime_profile=expected_runtime_profile,
            )
        )
        try:
            actual_identity = cache_identity(cache_dir)
        except (FileNotFoundError, TypeError, ValueError) as error:
            problems.append(str(error))
        else:
            if report.get(identity_key) != actual_identity:
                problems.append(f"matrix cache identity mismatch {dataset}/{split}")
    task_report = {
        "dataset": dataset,
        "path": str(path),
        "task": task,
        "metric": metric,
        "maximize": maximize,
        "trainable_parameter_count": next(iter(parameter_counts), None),
        "run_count": len(seen),
        "required_optimizer_steps": (
            math.ceil(expected_counts["train"] / expected_budget["batch_size"])
            * expected_budget["epochs"]
            * len(_REPRESENTATIONS)
            * len(_SEEDS)
        ),
        "reported_completed_optimizer_steps": report.get("completed_optimizer_steps"),
        "required_training_sample_exposures": (
            expected_counts["train"]
            * expected_budget["epochs"]
            * len(_REPRESENTATIONS)
            * len(_SEEDS)
        ),
        "reported_completed_training_sample_exposures": report.get(
            "completed_training_sample_exposures"
        ),
    }
    return observations, task_report, problems


def _supervised_candidate_checks(
    matrices: Mapping[str, dict[str, dict[int, float]]],
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate in ("response", "full"):
        task_checks: dict[str, Any] = {}
        for dataset, observations in matrices.items():
            maximize = _TASKS[dataset][2]
            candidate_values = observations[candidate]
            static = _paired_gain(
                candidate_values,
                [observations[name] for name in sorted(_STATIC_CONTROLS)],
                maximize=maximize,
            )
            graph = _paired_gain(
                candidate_values,
                [observations[name] for name in sorted(_GRAPH_CONTROLS[candidate])],
                maximize=maximize,
            )
            shuffled = _paired_gain(
                candidate_values,
                [observations[_SHUFFLED_CONTROL[candidate]]],
                maximize=maximize,
            )
            passed = all(
                comparison["stable_positive_gain"] for comparison in (static, graph, shuffled)
            )
            task_checks[dataset] = {
                "passed": passed,
                "static_and_hidden_controls": static,
                "graph_controls": graph,
                "shuffled_control": shuffled,
            }
        dense_passes = sum(task_checks[name]["passed"] for name in ("voc2012", "ade20k", "nyuv2"))
        candidates[candidate] = {
            "classification_passed": task_checks["imagenet100"]["passed"],
            "dense_tasks_passed": dense_passes,
            "minimum_dense_tasks": 2,
            "supports_cross_task_hypothesis": bool(
                task_checks["imagenet100"]["passed"] and dense_passes >= 2
            ),
            "tasks": task_checks,
        }
    pvalues: dict[str, float] = {}
    for candidate, candidate_report in candidates.items():
        for dataset, task_report in candidate_report["tasks"].items():
            for control_group in (
                "static_and_hidden_controls",
                "graph_controls",
                "shuffled_control",
            ):
                name = f"{candidate}/{dataset}/{control_group}"
                pvalues[name] = task_report[control_group]["paired_sign_flip_pvalue"]
    adjusted = holm_adjusted_pvalues(pvalues)
    for name, value in adjusted.items():
        candidate, dataset, control_group = name.split("/")
        comparison = candidates[candidate]["tasks"][dataset][control_group]
        comparison["holm_adjusted_sign_flip_pvalue"] = value
        comparison["holm_reject_alpha_0_05"] = value < 0.05
    return candidates


def _unsupervised_checks(
    path: Path,
    provenance: Mapping[str, Any],
    expected_runtime_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    if not path.is_file():
        return {"path": str(path), "passed": False}, [f"missing {path}"]
    report = _read_json(path)
    if report.get("status") != "passed":
        problems.append(f"VOC unsupervised report is not passed {path}")
    if report.get("code_revision") != provenance["code_revision"]:
        problems.append(f"VOC unsupervised revision mismatch {path}")
    if report.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append(f"VOC unsupervised code tree mismatch {path}")
    if report.get("code_dirty") is not False:
        problems.append(f"VOC unsupervised report came from a dirty worktree {path}")
    cache_dir = _resolve_report_path(
        str(report.get("cache_dir", "")),
        Path(__file__).resolve().parents[2],
    )
    problems.extend(
        _validate_cache_manifest(
            cache_dir,
            expected_dataset="voc2012",
            expected_split="test",
            expected_count=1449,
            provenance=provenance,
            storage_policy="dense",
            expected_runtime_profile=expected_runtime_profile,
        )
    )
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        return {"path": str(path), "passed": False}, problems + [
            "VOC per-sample metrics are missing"
        ]
    if int(report.get("num_samples", -1)) != 1449 or len(per_sample) != 1449:
        problems.append("VOC unsupervised sample count mismatch")
    sample_ids = [str(sample.get("sample_id", "")) for sample in per_sample]
    if len(set(sample_ids)) != len(sample_ids) or any(not value for value in sample_ids):
        problems.append("VOC unsupervised sample IDs are missing or duplicated")
    else:
        manifest = _read_json(cache_dir / "dataset_manifest.json")
        if sample_ids_sha256(sample_ids) != manifest.get("sample_ids_sha256"):
            problems.append("VOC unsupervised sample-ID hash mismatch")
    indexed = {
        str(sample.get("sample_id", "")): sample
        for sample in per_sample
        if sample.get("sample_id")
    }
    comparisons: dict[str, Any] = {}
    metric_eligibility: dict[str, Any] = {}
    for metric in _GRAPH_DECISION_METRICS:
        eligible, eligibility_audit, eligibility_problems = (
            _structural_metric_eligibility(
                per_sample,
                metric,
                source="empty_prompt",
            )
        )
        metric_eligibility[metric] = eligibility_audit
        problems.extend(eligibility_problems)
        if not eligible:
            continue
        for control in sorted(_UNSUPERVISED_CONTROLS):
            differences: list[float] = []
            for sample_id in sorted(eligible):
                sample = indexed[sample_id]
                representations = sample.get("representations", {})
                try:
                    response_value = float(representations["response"][metric])
                    control_value = float(representations[control][metric])
                except (KeyError, TypeError, ValueError):
                    problems.append(f"missing VOC {metric} for {control}")
                    break
                if not math.isfinite(response_value) or not math.isfinite(control_value):
                    problems.append(
                        "eligible VOC metric is non-finite "
                        f"metric={metric} control={control} sample_id={sample_id}"
                    )
                    break
                differences.append(response_value - control_value)
            if len(differences) != len(eligible):
                continue
            summary = bootstrap_mean_interval(differences, seed=4121, resamples=2000)
            interval = summary["ci95"]
            passed = bool(summary["mean"] > 0 and interval is not None and interval[0] > 0)
            comparisons[f"response-minus-{control}/{metric}"] = {
                **summary,
                "num_images": len(differences),
                "sample_ids_sha256": eligibility_audit[
                    "eligible_sample_ids_sha256"
                ],
                "paired_bootstrap_ci_excludes_zero": passed,
                "passed": passed,
            }
    passed = bool(comparisons and all(value["passed"] for value in comparisons.values()))
    return {
        "path": str(path),
        "passed": passed,
        "comparisons": comparisons,
        "num_images": len(per_sample),
        "metric_eligibility": metric_eligibility,
    }, problems


def _load_unsupervised_per_sample(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    expected_probe_type: str,
    expected_prompt: str,
    expected_random_transformer: bool,
    expected_runtime_profile: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Any],
    dict[str, set[str]],
    list[str],
]:
    problems: list[str] = []
    if not path.is_file():
        return {}, {"path": str(path)}, {}, [f"missing {path}"]
    report = _read_json(path)
    if report.get("status") != "passed":
        problems.append(f"causal report is not passed {path}")
    if report.get("code_revision") != provenance["code_revision"]:
        problems.append(f"causal report revision mismatch {path}")
    if report.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append(f"causal report code tree mismatch {path}")
    if report.get("code_dirty") is not False:
        problems.append(f"causal report came from a dirty worktree {path}")
    cache_dir = _resolve_report_path(
        str(report.get("cache_dir", "")),
        Path(__file__).resolve().parents[2],
    )
    problems.extend(
        _validate_cache_manifest(
            cache_dir,
            expected_dataset="voc2012",
            expected_split="test",
            expected_count=1449,
            provenance=provenance,
            storage_policy="dense",
            expected_probe_type=expected_probe_type,
            expected_prompt=expected_prompt,
            expected_random_transformer=expected_random_transformer,
            expected_runtime_profile=expected_runtime_profile,
        )
    )
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, list) or len(per_sample) != 1449:
        problems.append(f"causal report sample count mismatch {path}")
        per_sample = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for sample in per_sample:
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            problems.append(f"causal report sample-ID mismatch {path}")
            continue
        indexed[sample_id] = sample
    if indexed:
        manifest = _read_json(cache_dir / "dataset_manifest.json")
        if sample_ids_sha256(list(indexed)) != manifest.get("sample_ids_sha256"):
            problems.append(f"causal report sample-ID hash mismatch {path}")
    eligibility_sets: dict[str, set[str]] = {}
    eligibility_audits: dict[str, Any] = {}
    for metric in _GRAPH_DECISION_METRICS:
        eligible, audit, eligibility_problems = _structural_metric_eligibility(
            per_sample,
            metric,
            source=str(path),
        )
        eligibility_sets[metric] = eligible
        eligibility_audits[metric] = audit
        problems.extend(eligibility_problems)
    return (
        indexed,
        {
            "path": str(path),
            "cache_dir": str(cache_dir),
            "metric_eligibility": eligibility_audits,
        },
        eligibility_sets,
        problems,
    )


def audit_causal_evidence(
    *,
    main_evidence_path: Path,
    causal_report_paths: Mapping[str, Path],
    runtime_profile_path: Path | None = None,
) -> dict[str, Any]:
    """Combine main-task evidence with pre-registered causal/condition audits."""

    provenance = code_provenance()
    problems: list[str] = []
    runtime_profile: dict[str, Any] = {}
    expected_runtime_profile: Mapping[str, Any] | None = None
    if runtime_profile_path is not None:
        runtime_profile, runtime_problems = _validate_runtime_profile(
            runtime_profile_path,
            provenance,
        )
        problems.extend(runtime_problems)
        expected_runtime_profile = runtime_profile.get("identity")
    if provenance.get("code_dirty") is not False:
        problems.append("causal evidence audit requires a clean code worktree")
    if not main_evidence_path.is_file():
        problems.append(f"missing {main_evidence_path}")
        main_evidence: dict[str, Any] = {}
    else:
        main_evidence = _read_json(main_evidence_path)
        if main_evidence.get("verdict") not in {
            "main_tasks_supported_pending_causal_audits",
            "limited_or_negative",
        }:
            problems.append("main evidence is neither complete positive nor complete negative")
        if main_evidence.get("code_revision") != provenance["code_revision"]:
            problems.append("main evidence revision mismatch")
        if main_evidence.get("code_tree_sha256") != provenance["code_tree_sha256"]:
            problems.append("main evidence code tree mismatch")
        if main_evidence.get("runtime_profile") != runtime_profile:
            problems.append("main evidence runtime profile mismatch")
    if set(causal_report_paths) != set(_CAUSAL_VARIANTS):
        problems.append("causal_report_paths do not match the registered variants")

    loaded: dict[str, dict[str, Mapping[str, Any]]] = {}
    eligibility: dict[str, dict[str, set[str]]] = {}
    sources: dict[str, Any] = {}
    main_unsupervised_path = _resolve_report_path(
        str(main_evidence.get("unsupervised", {}).get("path", "")),
        Path(__file__).resolve().parents[2],
    )
    main_indexed, main_source, main_eligibility, main_problems = (
        _load_unsupervised_per_sample(
            main_unsupervised_path,
            provenance,
            expected_probe_type="structured",
            expected_prompt="",
            expected_random_transformer=False,
            expected_runtime_profile=expected_runtime_profile,
        )
    )
    loaded["empty_prompt"] = main_indexed
    eligibility["empty_prompt"] = main_eligibility
    sources["empty_prompt"] = main_source
    problems.extend(main_problems)
    for variant, contract in _CAUSAL_VARIANTS.items():
        path = causal_report_paths.get(variant)
        if path is None:
            continue
        indexed, source, variant_eligibility, variant_problems = (
            _load_unsupervised_per_sample(
                path,
                provenance,
                expected_probe_type=contract["probe_type"],
                expected_prompt=contract["prompt"],
                expected_random_transformer=contract["random_transformer"],
                expected_runtime_profile=expected_runtime_profile,
            )
        )
        loaded[variant] = indexed
        eligibility[variant] = variant_eligibility
        sources[variant] = source
        problems.extend(variant_problems)

    comparisons: dict[str, Any] = {}
    if not problems:
        sample_ids = set(loaded["empty_prompt"])
        for variant, samples in loaded.items():
            if set(samples) != sample_ids:
                problems.append(f"causal sample IDs do not align for {variant}")
        for metric in _GRAPH_DECISION_METRICS:
            eligible_ids = eligibility["empty_prompt"][metric]
            eligibility_mismatch = False
            for variant in sorted(loaded):
                if eligibility[variant][metric] != eligible_ids:
                    problems.append(
                        "causal structural eligibility does not align "
                        f"source={variant} metric={metric}"
                    )
                    eligibility_mismatch = True
            if eligibility_mismatch or not eligible_ids:
                continue
            eligible_hash = sample_ids_sha256(sorted(eligible_ids))
            for variant in ("random_flow", "spatially_shuffled_probe"):
                differences: list[float] = []
                for sample_id in sorted(eligible_ids):
                    try:
                        main_value = float(
                            loaded["empty_prompt"][sample_id]["representations"]["response"][metric]
                        )
                        control_value = float(
                            loaded[variant][sample_id]["representations"]["response"][metric]
                        )
                    except (KeyError, TypeError, ValueError):
                        problems.append(f"missing causal metric {variant}/{metric}")
                        break
                    if not math.isfinite(main_value) or not math.isfinite(control_value):
                        problems.append(
                            "eligible causal metric is non-finite "
                            f"source={variant} metric={metric} sample_id={sample_id}"
                        )
                        break
                    differences.append(main_value - control_value)
                if len(differences) != len(eligible_ids):
                    continue
                summary = bootstrap_mean_interval(differences, seed=4121, resamples=2000)
                interval = summary["ci95"]
                passed = bool(summary["mean"] > 0 and interval is not None and interval[0] > 0)
                comparisons[f"pretrained-structured-minus-{variant}/{metric}"] = {
                    **summary,
                    "num_images": len(differences),
                    "sample_ids_sha256": eligible_hash,
                    "passed": passed,
                }
            for condition in ("neutral_prompt", "unrelated_prompt"):
                for control in sorted(_CONDITION_CONTROLS):
                    differences = []
                    for sample_id in sorted(eligible_ids):
                        try:
                            response_value = float(
                                loaded[condition][sample_id]["representations"]["response"][metric]
                            )
                            control_value = float(
                                loaded[condition][sample_id]["representations"][control][metric]
                            )
                        except (KeyError, TypeError, ValueError):
                            problems.append(
                                f"missing condition metric {condition}/{control}/{metric}"
                            )
                            break
                        if not math.isfinite(response_value) or not math.isfinite(control_value):
                            problems.append(
                                "eligible condition metric is non-finite "
                                f"condition={condition} control={control} metric={metric} "
                                f"sample_id={sample_id}"
                            )
                            break
                        differences.append(response_value - control_value)
                    if len(differences) != len(eligible_ids):
                        continue
                    summary = bootstrap_mean_interval(differences, seed=4121, resamples=2000)
                    interval = summary["ci95"]
                    passed = bool(summary["mean"] > 0 and interval is not None and interval[0] > 0)
                    comparisons[f"{condition}/response-minus-{control}/{metric}"] = {
                        **summary,
                        "num_images": len(differences),
                        "sample_ids_sha256": eligible_hash,
                        "passed": passed,
                    }
    main_verdict = main_evidence.get("verdict")
    if problems:
        status = "incomplete"
        verdict = "incomplete"
    elif main_verdict == "limited_or_negative":
        status = "failed"
        verdict = "limited_or_negative"
    elif comparisons and all(value["passed"] for value in comparisons.values()):
        status = "passed"
        verdict = "supports_core_hypothesis"
    else:
        status = "failed"
        verdict = "main_task_gain_not_causally_attributed"
    return {
        "status": status,
        "verdict": verdict,
        **provenance,
        "paper_evidence": verdict != "incomplete",
        "supports_strong_claims": verdict == "supports_core_hypothesis",
        "problems": sorted(set(problems)),
        "main_evidence": str(main_evidence_path),
        "main_evidence_verdict": main_verdict,
        "sources": sources,
        "runtime_profile": runtime_profile,
        "comparisons": comparisons,
        "decision_rule": (
            "paired-image bootstrap CI95 lower bound > 0 for pretrained structured "
            "response versus random Flow and spatially shuffled probes, and for response "
            "versus registered controls under both fixed text conditions"
        ),
    }


def audit_full_evidence(
    *,
    matrix_paths: Mapping[str, Path],
    voc_unsupervised_path: Path,
    backbone_asset_path: Path,
    split_audit_paths: Mapping[str, Path],
    runtime_profile_path: Path | None = None,
    readout_runtime_profile_path: Path | None = None,
) -> dict[str, Any]:
    """Audit complete formal evidence without converting smoke tests into claims."""

    provenance = code_provenance()
    repository_root = Path(__file__).resolve().parents[2]
    problems: list[str] = []
    runtime_profile: dict[str, Any] = {}
    expected_runtime_profile: Mapping[str, Any] | None = None
    if runtime_profile_path is not None:
        runtime_profile, runtime_problems = _validate_runtime_profile(
            runtime_profile_path,
            provenance,
        )
        problems.extend(runtime_problems)
        expected_runtime_profile = runtime_profile.get("identity")
    readout_runtime_profile: dict[str, Any] = {}
    expected_readout_runtime_profile: Mapping[str, Any] | None = None
    if readout_runtime_profile_path is None:
        problems.append("formal evidence audit requires a readout runtime profile")
    else:
        try:
            imagenet_matrix_path = matrix_paths.get("imagenet100")
            if imagenet_matrix_path is None:
                raise ValueError("missing imagenet100 matrix path")
            profile_report = load_readout_runtime_gate_report(
                RunConfig.from_mapping(_read_json(imagenet_matrix_path)["config"]),
                readout_runtime_profile_path,
            )
            identity = readout_runtime_profile_identity(readout_runtime_profile_path)
            readout_runtime_profile = {
                "path": str(readout_runtime_profile_path),
                "identity": identity,
                "report": profile_report,
            }
            expected_readout_runtime_profile = identity
        except (KeyError, OSError, TypeError, ValueError) as error:
            problems.append(f"invalid readout runtime profile: {error}")
    if provenance.get("code_dirty") is not False:
        problems.append("formal evidence audit requires a clean code worktree")
    if set(matrix_paths) != set(_TASKS):
        problems.append("matrix_paths must contain imagenet100, voc2012, ade20k, and nyuv2")
    if set(split_audit_paths) != set(_TASKS):
        problems.append("split_audit_paths must contain imagenet100, voc2012, ade20k, and nyuv2")
    backbone_asset: dict[str, Any] = {"path": str(backbone_asset_path)}
    if not backbone_asset_path.is_file():
        problems.append(f"missing {backbone_asset_path}")
    else:
        backbone_asset = _read_json(backbone_asset_path)
        if backbone_asset.get("status") != "passed":
            problems.append("backbone asset audit is not passed")
        if backbone_asset.get("code_revision") != provenance["code_revision"]:
            problems.append("backbone asset revision mismatch")
        if backbone_asset.get("code_tree_sha256") != provenance["code_tree_sha256"]:
            problems.append("backbone asset code tree mismatch")
        if backbone_asset.get("code_dirty") is not False:
            problems.append("backbone asset audit came from a dirty worktree")
        if backbone_asset.get("random_transformer") is not False:
            problems.append("backbone asset audit is not pretrained")

    split_audits: dict[str, Any] = {}
    expected_split_hashes: dict[str, dict[str, str]] = {}
    for dataset in _TASKS:
        split_path = split_audit_paths.get(dataset)
        if split_path is None:
            continue
        if not split_path.is_file():
            problems.append(f"missing {split_path}")
            continue
        split_report = _read_json(split_path)
        split_audits[dataset] = {"path": str(split_path), "report": split_report}
        if split_report.get("status") != "passed":
            problems.append(f"split audit is not passed {split_path}")
        if split_report.get("dataset") != dataset:
            problems.append(f"split audit dataset mismatch {split_path}")
        if split_report.get("code_revision") != provenance["code_revision"]:
            problems.append(f"split audit revision mismatch {split_path}")
        if split_report.get("code_tree_sha256") != provenance["code_tree_sha256"]:
            problems.append(f"split audit code tree mismatch {split_path}")
        if split_report.get("code_dirty") is not False:
            problems.append(f"split audit came from a dirty worktree {split_path}")
        expected_split_hashes[dataset] = {
            split: str(split_report.get("splits", {}).get(split, {}).get("sample_ids_sha256", ""))
            for split in ("train", "val", "test")
        }
    matrices: dict[str, dict[str, dict[int, float]]] = {}
    task_reports: dict[str, Any] = {}
    for dataset in _TASKS:
        path = matrix_paths.get(dataset)
        if path is None:
            continue
        observations, task_report, task_problems = _validate_matrix(
            dataset,
            path,
            provenance,
            repository_root,
            expected_runtime_profile,
        )
        matrices[dataset] = observations
        task_reports[dataset] = task_report
        problems.extend(task_problems)
        if path.is_file() and dataset in expected_split_hashes:
            matrix = _read_json(path)
            if matrix.get("readout_runtime_profile") != expected_readout_runtime_profile:
                problems.append(f"matrix readout runtime profile mismatch {dataset}")
            for split, cache_key in (
                ("train", "train_cache"),
                ("val", "validation_cache"),
                ("test", "test_cache"),
            ):
                actual_hash = matrix.get(cache_key, {}).get("sample_ids_sha256")
                expected_hash = expected_split_hashes[dataset][split]
                if not expected_hash or actual_hash != expected_hash:
                    problems.append(f"cache sample-ID hash mismatch {dataset}/{split}")
    unsupervised, unsupervised_problems = _unsupervised_checks(
        voc_unsupervised_path,
        provenance,
        expected_runtime_profile,
    )
    problems.extend(unsupervised_problems)
    supervised: dict[str, Any] = {}
    if not problems:
        supervised = _supervised_candidate_checks(matrices)
    if problems:
        verdict = "incomplete"
        status = "incomplete"
    else:
        supported_candidates = [
            name for name, result in supervised.items() if result["supports_cross_task_hypothesis"]
        ]
        if unsupervised["passed"] and supported_candidates:
            verdict = "main_tasks_supported_pending_causal_audits"
            status = "passed"
        else:
            verdict = "limited_or_negative"
            status = "failed"
    return {
        "status": status,
        "verdict": verdict,
        **provenance,
        "paper_evidence": verdict != "incomplete",
        "supports_strong_claims": False,
        "requires_causal_audits": verdict == "main_tasks_supported_pending_causal_audits",
        "problems": sorted(set(problems)),
        "required_seeds": sorted(_SEEDS),
        "required_representations": sorted(_REPRESENTATIONS),
        "training_execution": {
            "coverage_contract": READOUT_TRAINING_COVERAGE_CONTRACT,
            "required_optimizer_steps": sum(
                int(task.get("required_optimizer_steps", 0))
                for task in task_reports.values()
            ),
            "reported_completed_optimizer_steps": sum(
                _integer_or(task.get("reported_completed_optimizer_steps"), 0)
                for task in task_reports.values()
            ),
            "required_training_sample_exposures": sum(
                int(task.get("required_training_sample_exposures", 0))
                for task in task_reports.values()
            ),
            "reported_completed_training_sample_exposures": sum(
                _integer_or(task.get("reported_completed_training_sample_exposures"), 0)
                for task in task_reports.values()
            ),
        },
        "tasks": task_reports,
        "backbone_asset": backbone_asset,
        "split_audits": split_audits,
        "runtime_profile": runtime_profile,
        "readout_runtime_profile": readout_runtime_profile,
        "unsupervised": unsupervised,
        "supervised": supervised,
        "decision_rule": {
            "unsupervised": (
                "paired-image bootstrap CI95 lower bound > 0 for every "
                "registered control and both metrics"
            ),
            "supervised_task": (
                "candidate beats strongest per-seed static/hidden, graph, and "
                "shuffled controls with >=2 positive seeds and paired-t CI95 "
                "lower bound > 0"
            ),
            "cross_task": (
                "the same candidate passes ImageNet-100 and at least two of VOC/ADE20K/NYUv2"
            ),
        },
    }
