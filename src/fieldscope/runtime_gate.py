"""Pre-registered, label-free AuraFlow runtime-equivalence gate."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from fieldscope.backends import build_backend
from fieldscope.config import RunConfig, runtime_method_contract_sha256
from fieldscope.contracts import FieldFeatures
from fieldscope.experiments import code_provenance, sample_noise_seed, set_experiment_seed
from fieldscope.feature_ops import stack_features
from fieldscope.response import FieldResponseExtractor

SCHEMA_VERSION = 1
REGISTERED_PROFILES = ((2, 8), (2, 16), (4, 32), (8, 64))
NUM_SYNTHETIC_IMAGES = 8
MIN_SPEEDUP_FRACTION = 0.05
MAX_MEMORY_FRACTION = 0.70


def load_runtime_gate_report(config: RunConfig, path: Path) -> dict[str, Any]:
    """Reuse a same-revision gate without changing its content-addressed identity."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Runtime gate report must be a JSON object")
    provenance = code_provenance()
    problems = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version")
    if payload.get("status") != "passed":
        problems.append("status")
    if payload.get("code_revision") != provenance["code_revision"]:
        problems.append("code_revision")
    if payload.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append("code_tree_sha256")
    if payload.get("code_dirty") is not False or provenance.get("code_dirty") is not False:
        problems.append("code_dirty")
    if payload.get("method_runtime_contract_sha256") != runtime_method_contract_sha256(
        config
    ):
        problems.append("method_runtime_contract_sha256")
    selected = payload.get("selected_profile", {})
    selected_shape = (
        selected.get("image_batch_size"),
        selected.get("probe_batch_size"),
    )
    if selected_shape not in REGISTERED_PROFILES:
        problems.append("selected_profile")
    if problems:
        raise ValueError(
            "Existing runtime gate report is stale or invalid: " + ", ".join(problems)
        )
    return payload


def _synthetic_images(size: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(90317)
    random = torch.rand(
        (NUM_SYNTHETIC_IMAGES, 3, size, size),
        generator=generator,
        dtype=torch.float32,
    )
    axis = torch.linspace(0.0, 1.0, size, dtype=torch.float32)
    horizontal = axis.reshape(1, 1, 1, size)
    vertical = axis.reshape(1, 1, size, 1)
    pattern = torch.cat(
        [
            horizontal.expand(1, 1, size, size),
            vertical.expand(1, 1, size, size),
            ((horizontal + vertical) * 0.5).expand(1, 1, size, size),
        ],
        dim=1,
    )
    return (0.85 * random + 0.15 * pattern).clamp(0.0, 1.0)


def _tensor_items(features: FieldFeatures) -> list[tuple[str, torch.Tensor]]:
    return [
        ("state", features.state),
        ("response", features.response),
        ("affinity", features.affinity),
        ("adjacency", features.adjacency),
        *[(f"baselines/{name}", value) for name, value in sorted(features.baselines.items())],
        *[(f"graphs/{name}", value) for name, value in sorted(features.graphs.items())],
    ]


def _feature_sha256(features: FieldFeatures) -> str:
    digest = hashlib.sha256()
    for name, tensor in _tensor_items(features):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _compare_features(reference: FieldFeatures, candidate: FieldFeatures) -> dict[str, Any]:
    reference_items = dict(_tensor_items(reference))
    candidate_items = dict(_tensor_items(candidate))
    fields: dict[str, Any] = {}
    exact = reference_items.keys() == candidate_items.keys()
    for name in sorted(reference_items.keys() | candidate_items.keys()):
        left = reference_items.get(name)
        right = candidate_items.get(name)
        if left is None or right is None:
            fields[name] = {"exact": False, "reason": "missing_field"}
            exact = False
            continue
        same_shape = left.shape == right.shape
        same_dtype = left.dtype == right.dtype
        equal = same_shape and same_dtype and torch.equal(left, right)
        maximum_absolute_error = None
        if same_shape and left.numel() and right.numel():
            maximum_absolute_error = float(
                (left.float() - right.float()).abs().max().item()
            )
        fields[name] = {
            "exact": equal,
            "same_shape": same_shape,
            "same_dtype": same_dtype,
            "maximum_absolute_error": maximum_absolute_error,
        }
        exact = exact and equal
    return {"exact": exact, "fields": fields}


def _run_profile(
    config: RunConfig,
    backend: Any,
    images: torch.Tensor,
    *,
    image_batch_size: int,
    probe_batch_size: int,
) -> tuple[dict[str, Any], FieldFeatures | None]:
    profile_config = replace(
        config,
        probe=replace(config.probe, probe_batch_size=probe_batch_size),
        runtime=replace(config.runtime, batch_size=image_batch_size),
    )
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(backend.device)
        torch.cuda.synchronize(backend.device)
    extractor = FieldResponseExtractor(backend, profile_config.probe)
    batches: list[FieldFeatures] = []
    started = time.perf_counter()
    try:
        for start in range(0, len(images), image_batch_size):
            stop = min(start + image_batch_size, len(images))
            sample_ids = [f"runtime-gate-{index:04d}" for index in range(start, stop)]
            batches.append(
                extractor.extract(
                    images[start:stop],
                    noise_seeds=[
                        sample_noise_seed(sample_id, config.probe.seed)
                        for sample_id in sample_ids
                    ],
                ).detached_cpu()
            )
        if backend.device.type == "cuda":
            torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - started
        features = stack_features(batches)
        peak_allocated = (
            torch.cuda.max_memory_allocated(backend.device)
            if backend.device.type == "cuda"
            else 0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(backend.device)
            if backend.device.type == "cuda"
            else 0
        )
        total_memory = (
            torch.cuda.get_device_properties(backend.device).total_memory
            if backend.device.type == "cuda"
            else 0
        )
        return (
            {
                "status": "completed",
                "image_batch_size": image_batch_size,
                "probe_batch_size": probe_batch_size,
                "num_images": len(images),
                "elapsed_seconds": elapsed,
                "seconds_per_image": elapsed / len(images),
                "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "cuda_total_memory_bytes": total_memory,
                "cuda_peak_reserved_fraction": (
                    peak_reserved / total_memory if total_memory else None
                ),
                "feature_sha256": _feature_sha256(features),
            },
            features,
        )
    except torch.OutOfMemoryError as error:
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
        return (
            {
                "status": "out_of_memory",
                "image_batch_size": image_batch_size,
                "probe_batch_size": probe_batch_size,
                "error": str(error),
            },
            None,
        )


def run_runtime_gate(config: RunConfig) -> dict[str, Any]:
    """Select only an exact, memory-safe, label-free runtime batching profile."""

    if config.backend.name != "auraflow" or config.backend.device != "cuda":
        raise ValueError("Runtime gate requires the formal CUDA AuraFlow config")
    if (config.runtime.batch_size, config.probe.probe_batch_size) != REGISTERED_PROFILES[0]:
        raise ValueError("Runtime gate baseline must be image batch 2 and probe batch 8")
    set_experiment_seed(config.probe.seed, config.runtime.deterministic)
    provenance = code_provenance()
    backend_started = time.perf_counter()
    backend = build_backend(config.backend)
    backend_initialization_seconds = time.perf_counter() - backend_started
    images = _synthetic_images(config.backend.image_size)
    warmup, _ = _run_profile(
        config,
        backend,
        images[:2],
        image_batch_size=REGISTERED_PROFILES[0][0],
        probe_batch_size=REGISTERED_PROFILES[0][1],
    )
    if warmup["status"] != "completed":
        raise RuntimeError("Registered runtime baseline warm-up did not complete")
    results: list[dict[str, Any]] = []
    reference: FieldFeatures | None = None
    for image_batch_size, probe_batch_size in REGISTERED_PROFILES:
        result, features = _run_profile(
            config,
            backend,
            images,
            image_batch_size=image_batch_size,
            probe_batch_size=probe_batch_size,
        )
        if reference is None and features is not None:
            reference = features
            result["equivalence"] = {"exact": True, "fields": {}}
        elif reference is not None and features is not None:
            result["equivalence"] = _compare_features(reference, features)
        else:
            result["equivalence"] = {"exact": False, "fields": {}}
        results.append(result)
    if reference is None or results[0]["status"] != "completed":
        raise RuntimeError("Registered runtime baseline did not complete")
    baseline_seconds = float(results[0]["seconds_per_image"])
    for result in results:
        completed = result["status"] == "completed"
        speedup = (
            baseline_seconds / float(result["seconds_per_image"]) - 1.0
            if completed
            else None
        )
        memory_safe = bool(
            completed
            and result.get("cuda_peak_reserved_fraction") is not None
            and result["cuda_peak_reserved_fraction"] <= MAX_MEMORY_FRACTION
        )
        exact = bool(result.get("equivalence", {}).get("exact"))
        eligible = bool(
            completed
            and exact
            and memory_safe
            and (
                result is results[0]
                or (speedup is not None and speedup >= MIN_SPEEDUP_FRACTION)
            )
        )
        result["speedup_fraction_vs_baseline"] = speedup
        result["memory_safe"] = memory_safe
        result["eligible"] = eligible
    eligible = [result for result in results if result["eligible"]]
    selected = min(eligible, key=lambda item: float(item["seconds_per_image"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        **provenance,
        "evidence_scope": "label_free_runtime_equivalence_and_throughput_only",
        "method_effectiveness_conclusion": None,
        "method_runtime_contract_sha256": runtime_method_contract_sha256(config),
        "registered_profiles": [
            {"image_batch_size": image, "probe_batch_size": probe}
            for image, probe in REGISTERED_PROFILES
        ],
        "num_synthetic_images": NUM_SYNTHETIC_IMAGES,
        "synthetic_image_policy": "fixed_seed_random_plus_coordinate_ramps_v1",
        "minimum_speedup_fraction": MIN_SPEEDUP_FRACTION,
        "maximum_cuda_reserved_fraction": MAX_MEMORY_FRACTION,
        "equivalence_rule": "torch.equal for every cached tensor field",
        "backend_initialization_seconds": backend_initialization_seconds,
        "warmup": {
            "profile": {
                "image_batch_size": REGISTERED_PROFILES[0][0],
                "probe_batch_size": REGISTERED_PROFILES[0][1],
            },
            "num_images": 2,
            "status": warmup["status"],
        },
        "backend": backend.describe(),
        "candidates": results,
        "selected_profile": {
            "image_batch_size": selected["image_batch_size"],
            "probe_batch_size": selected["probe_batch_size"],
            "seconds_per_image": selected["seconds_per_image"],
            "speedup_fraction_vs_baseline": selected["speedup_fraction_vs_baseline"],
            "cuda_peak_reserved_fraction": selected["cuda_peak_reserved_fraction"],
            "feature_sha256": selected["feature_sha256"],
        },
    }
