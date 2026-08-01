"""Exactness and resource gate for parallel cached-readout seed execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from fieldscope.config import RunConfig
from fieldscope.experiments import code_provenance

SCHEMA_VERSION = 2
REGISTERED_SEED_WORKERS = (1, 2, 3)
REGISTERED_SEEDS = (4121, 7319, 104729)
GATE_TASK = "classification"
GATE_REPRESENTATION = "full"
GATE_EPOCHS = 20
GATE_BATCH_SIZE = 128
MIN_SPEEDUP_FRACTION = 0.05
MAX_CUDA_RESERVED_FRACTION = 0.70
MIN_FREE_RAM_RESERVE_BYTES = 64 * 1024**3
EQUIVALENCE_RULE = (
    "strict validated synchronous serial versus fast serial, then fast serial versus "
    "parallel candidates, using torch.equal/array_equal/exact scalar equality for "
    "model, optimizer, scheduler, RNG, timing-stripped history, best selection, and "
    "held-out metric"
)
REGISTERED_FORMAL_WORKLOADS = (
    {
        "dataset": "imagenet100",
        "task": "classification",
        "batch_size": 128,
        "output_channels": 100,
    },
    {
        "dataset": "voc2012",
        "task": "segmentation",
        "batch_size": 4,
        "output_channels": 21,
    },
    {
        "dataset": "ade20k",
        "task": "segmentation",
        "batch_size": 2,
        "output_channels": 150,
    },
    {
        "dataset": "nyuv2",
        "task": "depth",
        "batch_size": 4,
        "output_channels": 1,
    },
)


def _activation_proxy_units(
    config: RunConfig,
    *,
    task: str,
    batch_size: int,
    output_channels: int,
) -> int:
    """Return a conservative relative activation envelope for one worker."""

    patches = config.probe.graph_grid[0] * config.probe.graph_grid[1]
    hidden_dim = config.tokenizer.hidden_dim
    input_dim = config.tokenizer.input_dim
    layers = config.tokenizer.num_layers
    # ``full`` is the registered worst-case representation: two fixed input
    # sketches, full adjacency, and all graph-message activations. The dense
    # head multiplier intentionally overcounts its three convolutions and
    # backward buffers so the proxy only serves as a conservative ordering.
    tokenizer_per_patch = 2 * input_dim + (4 * layers + 8) * hidden_dim + 2 * patches
    units = batch_size * patches * tokenizer_per_patch
    if task in {"segmentation", "depth", "normals"}:
        units += batch_size * patches * (32 * hidden_dim + output_channels)
    else:
        units += batch_size * (hidden_dim + output_channels)
    return int(units)


def formal_readout_workload_envelope(config: RunConfig) -> dict[str, Any]:
    workloads = [
        {
            **workload,
            "activation_proxy_units": _activation_proxy_units(
                config,
                task=str(workload["task"]),
                batch_size=int(workload["batch_size"]),
                output_channels=int(workload["output_channels"]),
            ),
        }
        for workload in REGISTERED_FORMAL_WORKLOADS
    ]
    gate_units = _activation_proxy_units(
        config,
        task=GATE_TASK,
        batch_size=GATE_BATCH_SIZE,
        output_channels=config.tokenizer.num_classes,
    )
    maximum = max(workloads, key=lambda workload: int(workload["activation_proxy_units"]))
    if gate_units < int(maximum["activation_proxy_units"]):
        raise ValueError("Readout runtime gate workload does not cover formal workloads")
    return {
        "schema_version": 1,
        "representation": GATE_REPRESENTATION,
        "grid_size": list(config.probe.graph_grid),
        "tokenizer_hidden_dim": config.tokenizer.hidden_dim,
        "tokenizer_input_dim": config.tokenizer.input_dim,
        "tokenizer_num_layers": config.tokenizer.num_layers,
        "gate": {
            "task": GATE_TASK,
            "batch_size": GATE_BATCH_SIZE,
            "output_channels": config.tokenizer.num_classes,
            "activation_proxy_units": gate_units,
        },
        "formal_workloads": workloads,
        "maximum_formal_activation_proxy_units": int(maximum["activation_proxy_units"]),
        "gate_to_maximum_formal_ratio": gate_units / int(maximum["activation_proxy_units"]),
        "coverage_rule": (
            "full representation, graph grid, tokenizer, conservative dense-head "
            "activation proxy, and registered formal task batch/class envelopes"
        ),
    }


def readout_execution_contract(config: RunConfig) -> dict[str, Any]:
    return {
        "backend": {
            "device": config.backend.device,
            "dtype": config.backend.dtype,
        },
        "tokenizer": {
            "hidden_dim": config.tokenizer.hidden_dim,
            "input_dim": config.tokenizer.input_dim,
            "num_layers": config.tokenizer.num_layers,
            "dropout": config.tokenizer.dropout,
        },
        "runtime": {
            "deterministic": config.runtime.deterministic,
            "readout_memory_cache_gib": config.runtime.readout_memory_cache_gib,
        },
    }


def readout_execution_contract_sha256(config: RunConfig) -> str:
    return hashlib.sha256(
        json.dumps(
            readout_execution_contract(config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def readout_runtime_profile_identity(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    raw = profile_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    selected = payload.get("selected_profile", {})
    seed_workers = int(selected.get("seed_workers", -1))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Readout runtime profile schema version mismatch")
    if payload.get("status") != "passed":
        raise ValueError("Readout runtime profile is not passed")
    if payload.get("code_dirty") is not False:
        raise ValueError("Readout runtime profile must come from a clean worktree")
    if payload.get("evidence_scope") != ("readout_seed_parallel_exactness_and_throughput_only"):
        raise ValueError("Readout runtime profile evidence scope mismatch")
    if payload.get("method_effectiveness_conclusion") is not None:
        raise ValueError("Readout runtime profile contains an effectiveness conclusion")
    if payload.get("equivalence_rule") != EQUIVALENCE_RULE:
        raise ValueError("Readout runtime profile equivalence rule mismatch")
    envelope = payload.get("formal_workload_envelope", {})
    if envelope.get("schema_version") != 1:
        raise ValueError("Readout runtime formal workload envelope schema mismatch")
    if envelope.get("representation") != GATE_REPRESENTATION:
        raise ValueError("Readout runtime formal workload envelope representation mismatch")
    if tuple(
        (
            workload.get("dataset"),
            workload.get("task"),
            workload.get("batch_size"),
            workload.get("output_channels"),
        )
        for workload in envelope.get("formal_workloads", [])
    ) != tuple(
        (
            workload["dataset"],
            workload["task"],
            workload["batch_size"],
            workload["output_channels"],
        )
        for workload in REGISTERED_FORMAL_WORKLOADS
    ):
        raise ValueError("Readout runtime formal workload registry mismatch")
    if tuple(payload.get("registered_seed_workers", [])) != REGISTERED_SEED_WORKERS:
        raise ValueError("Readout runtime profile worker registry mismatch")
    if tuple(payload.get("registered_seeds", [])) != REGISTERED_SEEDS:
        raise ValueError("Readout runtime profile seed registry mismatch")
    if payload.get("task") != GATE_TASK:
        raise ValueError("Readout runtime profile task mismatch")
    if payload.get("representation") != GATE_REPRESENTATION:
        raise ValueError("Readout runtime profile representation mismatch")
    if int(payload.get("epochs", -1)) != GATE_EPOCHS:
        raise ValueError("Readout runtime profile epoch budget mismatch")
    if int(payload.get("batch_size", -1)) != GATE_BATCH_SIZE:
        raise ValueError("Readout runtime profile batch size mismatch")
    if float(payload.get("minimum_speedup_fraction", float("nan"))) != (MIN_SPEEDUP_FRACTION):
        raise ValueError("Readout runtime profile speed threshold mismatch")
    if float(payload.get("minimum_fast_path_speedup_fraction", float("nan"))) != (
        MIN_SPEEDUP_FRACTION
    ):
        raise ValueError("Readout runtime fast-path speed threshold mismatch")
    if float(payload.get("maximum_cuda_reserved_fraction", float("nan"))) != (
        MAX_CUDA_RESERVED_FRACTION
    ):
        raise ValueError("Readout runtime profile CUDA threshold mismatch")
    if int(payload.get("minimum_free_ram_reserve_bytes", -1)) != (MIN_FREE_RAM_RESERVE_BYTES):
        raise ValueError("Readout runtime profile RAM threshold mismatch")
    if seed_workers not in REGISTERED_SEED_WORKERS:
        raise ValueError("Readout runtime profile selected unregistered workers")
    strict_reference = payload.get("strict_reference")
    if not isinstance(strict_reference, Mapping):
        raise ValueError("Readout runtime strict reference is missing")
    if strict_reference.get("status") != "completed":
        raise ValueError("Readout runtime strict reference did not complete")
    if int(strict_reference.get("seed_workers", -1)) != 1:
        raise ValueError("Readout runtime strict reference worker count mismatch")
    strict_elapsed = float(strict_reference.get("elapsed_seconds", float("nan")))
    if not math.isfinite(strict_elapsed) or strict_elapsed <= 0:
        raise ValueError("Readout runtime strict reference elapsed time is invalid")
    strict_speedup = float(
        strict_reference.get("fast_serial_speedup_fraction_vs_strict", float("nan"))
    )
    if not math.isfinite(strict_speedup):
        raise ValueError("Readout runtime strict reference speedup is invalid")
    strict_equivalence = strict_reference.get("equivalence_to_fast_serial")
    if not isinstance(strict_equivalence, Mapping):
        raise ValueError("Readout runtime strict reference equivalence is missing")
    strict_run_details = strict_equivalence.get("runs")
    if not isinstance(strict_run_details, Mapping):
        raise ValueError("Readout runtime strict reference run registry is missing")
    expected_labels = {f"{GATE_REPRESENTATION}/seed-{seed}" for seed in REGISTERED_SEEDS}
    if set(strict_run_details) != expected_labels:
        raise ValueError("Readout runtime strict reference run registry mismatch")
    for detail in strict_run_details.values():
        if not isinstance(detail, Mapping):
            raise ValueError("Readout runtime strict reference detail mismatch")
        if not (
            detail.get("checkpoint_semantics_exact") is True
            and detail.get("held_out_metric_exact") is True
            and detail.get("exact") is True
        ):
            raise ValueError("Readout runtime strict reference is not exact")
    if strict_equivalence.get("exact") is not True:
        raise ValueError("Readout runtime strict reference is not exact")
    candidate_list = payload.get("candidates", [])
    candidates = {int(candidate.get("seed_workers", -1)): candidate for candidate in candidate_list}
    if len(candidates) != len(candidate_list) or set(candidates) != set(REGISTERED_SEED_WORKERS):
        raise ValueError("Readout runtime profile candidate registry mismatch")
    serial_elapsed: float | None = None
    for workers, candidate in candidates.items():
        status = candidate.get("status")
        if status not in {"completed", "failed"}:
            raise ValueError("Readout runtime profile candidate status mismatch")
        if status == "failed" and candidate.get("eligible") is not False:
            raise ValueError("Failed readout runtime candidate cannot be eligible")
        if workers == 1 and status != "completed":
            raise ValueError("Serial readout runtime candidate must complete")
        if status == "failed":
            continue
        equivalence = candidate.get("equivalence", {})
        run_details = equivalence.get("runs", {})
        if workers == 1:
            if equivalence != {"exact": True, "runs": {}}:
                raise ValueError("Serial readout runtime equivalence summary mismatch")
        else:
            expected_labels = {f"{GATE_REPRESENTATION}/seed-{seed}" for seed in REGISTERED_SEEDS}
            if set(run_details) != expected_labels:
                raise ValueError("Readout runtime equivalence run registry mismatch")
            detail_exact = True
            for detail in run_details.values():
                expected_exact = bool(
                    detail.get("checkpoint_semantics_exact") is True
                    and detail.get("held_out_metric_exact") is True
                )
                if detail.get("exact") is not expected_exact:
                    raise ValueError("Readout runtime equivalence detail mismatch")
                detail_exact = detail_exact and expected_exact
            if equivalence.get("exact") is not detail_exact:
                raise ValueError("Readout runtime equivalence summary mismatch")
        candidate_elapsed = float(candidate.get("elapsed_seconds", float("nan")))
        candidate_speedup = float(candidate.get("speedup_fraction_vs_serial", float("nan")))
        candidate_cuda = candidate.get("cuda_reserved_fraction_upper_bound")
        per_seed_peaks = candidate.get("per_seed_cuda_peak_reserved_bytes", [])
        memory_measurement_complete = bool(
            len(per_seed_peaks) == len(REGISTERED_SEEDS)
            and all(int(value) > 0 for value in per_seed_peaks)
        )
        available_ram = int(candidate.get("available_ram_bytes", -1))
        required_ram = int(candidate.get("required_ram_with_reserve_bytes", -1))
        candidate_memory_safe = bool(
            candidate_cuda is not None
            and memory_measurement_complete
            and math.isfinite(float(candidate_cuda))
            and float(candidate_cuda) <= MAX_CUDA_RESERVED_FRACTION
            and available_ram >= required_ram
            and required_ram >= MIN_FREE_RAM_RESERVE_BYTES
        )
        if not math.isfinite(candidate_elapsed) or candidate_elapsed <= 0:
            raise ValueError("Readout runtime candidate elapsed time is invalid")
        if not math.isfinite(candidate_speedup):
            raise ValueError("Readout runtime candidate speedup is invalid")
        if workers == 1:
            serial_elapsed = candidate_elapsed
            if candidate_speedup != 0.0:
                raise ValueError("Serial readout runtime candidate speedup must be zero")
        if candidate.get("memory_safe") is not candidate_memory_safe:
            raise ValueError("Readout runtime candidate memory eligibility mismatch")
        if candidate.get("memory_measurement_complete") is not memory_measurement_complete:
            raise ValueError("Readout runtime candidate memory measurement mismatch")
        candidate_eligible = bool(
            candidate.get("equivalence", {}).get("exact") is True
            and candidate_memory_safe
            and (workers == 1 or candidate_speedup >= MIN_SPEEDUP_FRACTION)
        )
        if candidate.get("eligible") is not candidate_eligible:
            raise ValueError("Readout runtime candidate eligibility mismatch")
    if serial_elapsed is None:
        raise ValueError("Serial readout runtime candidate elapsed time is missing")
    expected_strict_speedup = strict_elapsed / serial_elapsed - 1.0
    if strict_speedup != expected_strict_speedup:
        raise ValueError("Readout runtime strict reference speedup summary mismatch")
    if strict_speedup < MIN_SPEEDUP_FRACTION:
        raise ValueError("Readout runtime fast path is not sufficiently faster than strict")
    for workers, candidate in candidates.items():
        if candidate.get("status") != "completed" or workers == 1:
            continue
        elapsed = float(candidate["elapsed_seconds"])
        expected_speedup = serial_elapsed / elapsed - 1.0
        if float(candidate["speedup_fraction_vs_serial"]) != expected_speedup:
            raise ValueError("Readout runtime candidate speedup summary mismatch")
    selected_candidate = candidates[seed_workers]
    if not selected_candidate.get("eligible"):
        raise ValueError("Selected readout runtime profile is not eligible")
    if selected_candidate.get("status") != "completed":
        raise ValueError("Selected readout runtime profile did not complete")
    if selected_candidate.get("equivalence", {}).get("exact") is not True:
        raise ValueError("Selected readout runtime profile is not exact")
    if selected_candidate.get("memory_safe") is not True:
        raise ValueError("Selected readout runtime profile is not memory safe")
    elapsed = float(selected_candidate.get("elapsed_seconds", float("nan")))
    speedup = float(selected_candidate.get("speedup_fraction_vs_serial", float("nan")))
    cuda_fraction = float(
        selected_candidate.get(
            "cuda_reserved_fraction_upper_bound",
            float("nan"),
        )
    )
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Selected readout runtime profile elapsed time is invalid")
    if not math.isfinite(speedup) or (seed_workers != 1 and speedup < MIN_SPEEDUP_FRACTION):
        raise ValueError("Selected readout runtime profile speedup is invalid")
    if not math.isfinite(cuda_fraction) or cuda_fraction > MAX_CUDA_RESERVED_FRACTION:
        raise ValueError("Selected readout runtime profile CUDA bound is invalid")
    selected_profile = payload.get("selected_profile", {})
    for key, expected in (
        ("elapsed_seconds", elapsed),
        ("speedup_fraction_vs_serial", speedup),
        ("cuda_reserved_fraction_upper_bound", cuda_fraction),
    ):
        if float(selected_profile.get(key, float("nan"))) != expected:
            raise ValueError("Readout runtime selected profile summary mismatch")
    eligible_candidates = [candidate for candidate in candidates.values() if candidate["eligible"]]
    fastest = min(
        eligible_candidates,
        key=lambda candidate: float(candidate["elapsed_seconds"]),
    )
    if int(fastest["seed_workers"]) != seed_workers:
        raise ValueError("Readout runtime profile did not select the fastest eligible candidate")
    return {
        "path": str(profile_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": payload["schema_version"],
        "code_revision": payload.get("code_revision"),
        "code_tree_sha256": payload.get("code_tree_sha256"),
        "code_dirty": payload.get("code_dirty"),
        "readout_execution_contract_sha256": payload.get("readout_execution_contract_sha256"),
        "formal_workload_envelope": envelope,
        "selected_profile": {"seed_workers": seed_workers},
    }


def load_readout_runtime_gate_report(
    config: RunConfig,
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = readout_runtime_profile_identity(path)
    provenance = code_provenance()
    problems = []
    if identity.get("code_revision") != provenance["code_revision"]:
        problems.append("code_revision")
    if identity.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        problems.append("code_tree_sha256")
    if identity.get("code_dirty") is not False or provenance.get("code_dirty") is not False:
        problems.append("code_dirty")
    if identity.get("readout_execution_contract_sha256") != readout_execution_contract_sha256(
        config
    ):
        problems.append("readout_execution_contract_sha256")
    if identity.get("formal_workload_envelope") != formal_readout_workload_envelope(config):
        problems.append("formal_workload_envelope")
    for candidate in payload.get("candidates", []):
        if candidate.get("status") != "completed":
            continue
        workers = int(candidate["seed_workers"])
        if int(candidate.get("available_ram_bytes", -1)) != int(
            payload.get("available_ram_bytes", -2)
        ):
            problems.append(f"available_ram_bytes[{workers}]")
        expected_cache_budget = int(config.runtime.readout_memory_cache_gib * 1024**3 * workers)
        expected_required_ram = expected_cache_budget + MIN_FREE_RAM_RESERVE_BYTES
        if int(candidate.get("worker_cache_budget_bytes", -1)) != expected_cache_budget:
            problems.append(f"worker_cache_budget_bytes[{workers}]")
        if int(candidate.get("required_ram_with_reserve_bytes", -1)) != expected_required_ram:
            problems.append(f"required_ram_with_reserve_bytes[{workers}]")
        concurrent_peak = int(candidate.get("concurrent_cuda_peak_reserved_upper_bound_bytes", -1))
        cuda_total = int(candidate.get("cuda_total_memory_bytes", -1))
        expected_fraction = concurrent_peak / cuda_total if cuda_total > 0 else None
        if candidate.get("cuda_reserved_fraction_upper_bound") != expected_fraction:
            problems.append(f"cuda_reserved_fraction_upper_bound[{workers}]")
    if problems:
        raise ValueError(
            "Existing readout runtime gate report is stale or invalid: " + ", ".join(problems)
        )
    return payload


def _available_ram_bytes() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return page_size * available_pages


def _semantic_history(history: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ignored = {"train_seconds", "train_samples_per_second", "validation_seconds"}
    return [{key: value for key, value in epoch.items() if key not in ignored} for epoch in history]


def _semantic_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    semantic = dict(payload)
    semantic["history"] = _semantic_history(list(payload["history"]))
    return semantic


def _exact_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, Mapping):
        return (
            isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_exact_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)):
        return (
            isinstance(second, type(first))
            and len(first) == len(second)
            and all(_exact_equal(left, right) for left, right in zip(first, second, strict=True))
        )
    return bool(first == second)


def _run_map(matrix: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {(str(run["representation"]), int(run["seed"])): run for run in matrix["runs"]}


def _compare_candidate(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    reference_matrix = json.loads(
        (reference_root / "matrix_report.json").read_text(encoding="utf-8")
    )
    candidate_matrix = json.loads(
        (candidate_root / "matrix_report.json").read_text(encoding="utf-8")
    )
    reference_runs = _run_map(reference_matrix)
    candidate_runs = _run_map(candidate_matrix)
    details: dict[str, Any] = {}
    exact = reference_runs.keys() == candidate_runs.keys()
    for key in sorted(reference_runs.keys() | candidate_runs.keys()):
        label = f"{key[0]}/seed-{key[1]}"
        if key not in reference_runs or key not in candidate_runs:
            details[label] = {"exact": False, "reason": "missing_run"}
            exact = False
            continue
        reference_run = reference_runs[key]
        candidate_run = candidate_runs[key]
        checkpoints_exact = True
        for checkpoint_name in ("best", "last"):
            reference_path = (
                reference_root
                / key[0]
                / f"seed-{key[1]}"
                / f"{GATE_TASK}_{key[0]}_seed{key[1]}_{checkpoint_name}.pt"
            )
            candidate_path = (
                candidate_root
                / key[0]
                / f"seed-{key[1]}"
                / f"{GATE_TASK}_{key[0]}_seed{key[1]}_{checkpoint_name}.pt"
            )
            reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
            candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
            checkpoints_exact = checkpoints_exact and _exact_equal(
                _semantic_checkpoint(reference_payload),
                _semantic_checkpoint(candidate_payload),
            )
        metric_exact = bool(reference_run["test_metric"] == candidate_run["test_metric"])
        details[label] = {
            "exact": checkpoints_exact and metric_exact,
            "checkpoint_semantics_exact": checkpoints_exact,
            "held_out_metric_exact": metric_exact,
        }
        exact = exact and details[label]["exact"]
    return {"exact": exact, "runs": details}


def _candidate_memory(
    output_root: Path,
    workers: int,
    config: RunConfig,
) -> dict[str, Any]:
    peaks = []
    for report_path in output_root.glob("*/seed-*/*_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        value = report.get("runtime", {}).get("cuda_peak_reserved_bytes")
        if value is not None:
            peaks.append(int(value))
    concurrent_peak = sum(sorted(peaks, reverse=True)[:workers])
    total_cuda = (
        torch.cuda.get_device_properties(config.backend.device).total_memory
        if torch.cuda.is_available()
        else 0
    )
    return {
        "per_seed_cuda_peak_reserved_bytes": sorted(peaks, reverse=True),
        "memory_measurement_complete": len(peaks) == len(REGISTERED_SEEDS),
        "concurrent_cuda_peak_reserved_upper_bound_bytes": concurrent_peak,
        "cuda_total_memory_bytes": total_cuda,
        "cuda_reserved_fraction_upper_bound": (
            concurrent_peak / total_cuda if total_cuda else None
        ),
    }


def _run_candidate(
    *,
    config_path: Path,
    train_cache_dir: Path,
    val_cache_dir: Path,
    test_cache_dir: Path,
    output_root: Path,
    workers: int,
    strict_reference: bool = False,
) -> tuple[float, Path]:
    command = [
        sys.executable,
        "-m",
        "fieldscope.cli",
        "run-readout-matrix",
        "--config",
        str(config_path),
        "--train-cache-dir",
        str(train_cache_dir),
        "--val-cache-dir",
        str(val_cache_dir),
        "--test-cache-dir",
        str(test_cache_dir),
        "--output-dir",
        str(output_root),
        "--task",
        GATE_TASK,
        "--epochs",
        str(GATE_EPOCHS),
        "--batch-size",
        str(GATE_BATCH_SIZE),
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--reference",
        GATE_REPRESENTATION,
        "--representation",
        GATE_REPRESENTATION,
        "--seed-workers",
        str(workers),
    ]
    if strict_reference:
        if workers != 1:
            raise ValueError("Strict readout reference must use one worker")
        command.extend(("--validate-model-features", "--strict-host-sync"))
    for seed in REGISTERED_SEEDS:
        command.extend(("--seed", str(seed)))
    label = "strict-reference" if strict_reference else f"workers-{workers}"
    log_path = output_root.parent / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"Readout runtime candidate workers={workers} failed:\n{tail}")
    return elapsed, log_path


def run_readout_runtime_gate(
    config: RunConfig,
    *,
    config_path: Path,
    train_cache_dir: Path,
    val_cache_dir: Path,
    test_cache_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if config.backend.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("Readout runtime gate requires CUDA")
    provenance = code_provenance()
    if provenance.get("code_dirty") is not False:
        raise ValueError("Readout runtime gate requires a clean worktree")
    gate_root = output_path.parent / f".{output_path.stem}.runs-{uuid4().hex}"
    gate_root.mkdir(parents=True, exist_ok=False)
    available_ram = _available_ram_bytes()
    candidates = []
    strict_root = (gate_root / "strict-reference").resolve()
    strict_elapsed, strict_log = _run_candidate(
        config_path=config_path.resolve(),
        train_cache_dir=train_cache_dir.resolve(),
        val_cache_dir=val_cache_dir.resolve(),
        test_cache_dir=test_cache_dir.resolve(),
        output_root=strict_root,
        workers=1,
        strict_reference=True,
    )
    reference_root: Path | None = None
    reference_seconds: float | None = None
    for workers in REGISTERED_SEED_WORKERS:
        candidate_root = (gate_root / f"workers-{workers}").resolve()
        try:
            elapsed, log_path = _run_candidate(
                config_path=config_path.resolve(),
                train_cache_dir=train_cache_dir.resolve(),
                val_cache_dir=val_cache_dir.resolve(),
                test_cache_dir=test_cache_dir.resolve(),
                output_root=candidate_root,
                workers=workers,
            )
        except RuntimeError as error:
            if workers == 1:
                raise
            candidates.append(
                {
                    "seed_workers": workers,
                    "status": "failed",
                    "error": str(error),
                    "equivalence": {"exact": False, "runs": {}},
                    "memory_safe": False,
                    "eligible": False,
                }
            )
            continue
        if reference_root is None:
            reference_root = candidate_root
            reference_seconds = elapsed
            equivalence = {"exact": True, "runs": {}}
            strict_equivalence = _compare_candidate(strict_root, candidate_root)
            if strict_equivalence["exact"] is not True:
                raise RuntimeError("Fast readout execution does not match the strict reference")
            fast_path_speedup = strict_elapsed / reference_seconds - 1.0
            if fast_path_speedup < MIN_SPEEDUP_FRACTION:
                raise RuntimeError("Fast readout execution is not sufficiently faster than strict")
        else:
            equivalence = _compare_candidate(reference_root, candidate_root)
        memory = _candidate_memory(candidate_root, workers, config)
        cache_budget = int(config.runtime.readout_memory_cache_gib * 1024**3 * workers)
        required_ram = cache_budget + MIN_FREE_RAM_RESERVE_BYTES
        speedup = reference_seconds / elapsed - 1.0 if reference_seconds is not None else 0.0
        cuda_fraction = memory["cuda_reserved_fraction_upper_bound"]
        memory_safe = bool(
            cuda_fraction is not None
            and memory["memory_measurement_complete"]
            and math.isfinite(float(cuda_fraction))
            and float(cuda_fraction) <= MAX_CUDA_RESERVED_FRACTION
            and available_ram >= required_ram
        )
        eligible = bool(
            equivalence["exact"]
            and memory_safe
            and (workers == 1 or speedup >= MIN_SPEEDUP_FRACTION)
        )
        candidates.append(
            {
                "seed_workers": workers,
                "status": "completed",
                "elapsed_seconds": elapsed,
                "speedup_fraction_vs_serial": speedup,
                "equivalence": equivalence,
                **memory,
                "available_ram_bytes": available_ram,
                "worker_cache_budget_bytes": cache_budget,
                "required_ram_with_reserve_bytes": required_ram,
                "memory_safe": memory_safe,
                "eligible": eligible,
                "matrix_report": str(candidate_root / "matrix_report.json"),
                "log": str(log_path.resolve()),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("No exact, resource-safe readout runtime profile is eligible")
    selected = min(eligible, key=lambda candidate: float(candidate["elapsed_seconds"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        **provenance,
        "evidence_scope": "readout_seed_parallel_exactness_and_throughput_only",
        "method_effectiveness_conclusion": None,
        "strict_reference": {
            "status": "completed",
            "seed_workers": 1,
            "elapsed_seconds": strict_elapsed,
            "fast_serial_speedup_fraction_vs_strict": (
                strict_elapsed / reference_seconds - 1.0
            ),
            "equivalence_to_fast_serial": strict_equivalence,
            "matrix_report": str(strict_root / "matrix_report.json"),
            "log": str(strict_log.resolve()),
        },
        "readout_execution_contract_sha256": readout_execution_contract_sha256(config),
        "formal_workload_envelope": formal_readout_workload_envelope(config),
        "registered_seed_workers": list(REGISTERED_SEED_WORKERS),
        "registered_seeds": list(REGISTERED_SEEDS),
        "representation": GATE_REPRESENTATION,
        "task": GATE_TASK,
        "epochs": GATE_EPOCHS,
        "batch_size": GATE_BATCH_SIZE,
        "minimum_speedup_fraction": MIN_SPEEDUP_FRACTION,
        "minimum_fast_path_speedup_fraction": MIN_SPEEDUP_FRACTION,
        "maximum_cuda_reserved_fraction": MAX_CUDA_RESERVED_FRACTION,
        "minimum_free_ram_reserve_bytes": MIN_FREE_RAM_RESERVE_BYTES,
        "available_ram_bytes": available_ram,
        "equivalence_rule": EQUIVALENCE_RULE,
        "candidates": candidates,
        "selected_profile": {
            "seed_workers": selected["seed_workers"],
            "elapsed_seconds": selected["elapsed_seconds"],
            "speedup_fraction_vs_serial": selected["speedup_fraction_vs_serial"],
            "cuda_reserved_fraction_upper_bound": selected["cuda_reserved_fraction_upper_bound"],
        },
    }
