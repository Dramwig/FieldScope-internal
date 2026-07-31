import json
from pathlib import Path

import pytest

from fieldscope.config import RunConfig
from fieldscope.readout_runtime_gate import (
    EQUIVALENCE_RULE,
    GATE_REPRESENTATION,
    MAX_CUDA_RESERVED_FRACTION,
    MIN_FREE_RAM_RESERVE_BYTES,
    MIN_SPEEDUP_FRACTION,
    _exact_equal,
    _semantic_checkpoint,
    formal_readout_workload_envelope,
    load_readout_runtime_gate_report,
    readout_execution_contract_sha256,
    readout_runtime_profile_identity,
    run_readout_runtime_gate,
)


def _config(*, hidden_dim: int = 256) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "backend": {
                "name": "auraflow",
                "model_path": "/models/AuraFlow-v0.3",
                "device": "cuda",
                "dtype": "bfloat16",
            },
            "tokenizer": {
                "hidden_dim": hidden_dim,
                "input_dim": 768,
                "num_layers": 3,
                "dropout": 0.1,
                "num_classes": 100,
                "segmentation_classes": 21,
            },
            "runtime": {
                "deterministic": True,
                "readout_memory_cache_gib": 1,
            },
        }
    )


def _profile(config: RunConfig) -> dict:
    provenance = {
        "code_revision": "revision",
        "code_tree_sha256": "tree",
        "code_dirty": False,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        **provenance,
        "evidence_scope": "readout_seed_parallel_exactness_and_throughput_only",
        "method_effectiveness_conclusion": None,
        "equivalence_rule": EQUIVALENCE_RULE,
        "readout_execution_contract_sha256": readout_execution_contract_sha256(config),
        "formal_workload_envelope": formal_readout_workload_envelope(config),
        "registered_seed_workers": [1, 2, 3],
        "registered_seeds": [4121, 7319, 104729],
        "task": "classification",
        "representation": GATE_REPRESENTATION,
        "epochs": 20,
        "batch_size": 128,
        "minimum_speedup_fraction": MIN_SPEEDUP_FRACTION,
        "maximum_cuda_reserved_fraction": MAX_CUDA_RESERVED_FRACTION,
        "minimum_free_ram_reserve_bytes": MIN_FREE_RAM_RESERVE_BYTES,
        "available_ram_bytes": 128 * 1024**3,
        "candidates": [
            {
                "seed_workers": 1,
                "status": "completed",
                "elapsed_seconds": 10.0,
                "speedup_fraction_vs_serial": 0.0,
                "equivalence": {"exact": True, "runs": {}},
                "cuda_reserved_fraction_upper_bound": 0.1,
                "available_ram_bytes": 128 * 1024**3,
                "worker_cache_budget_bytes": 1 * 1024**3,
                "required_ram_with_reserve_bytes": 65 * 1024**3,
                "concurrent_cuda_peak_reserved_upper_bound_bytes": 1,
                "cuda_total_memory_bytes": 10,
                "per_seed_cuda_peak_reserved_bytes": [1, 1, 1],
                "memory_measurement_complete": True,
                "memory_safe": True,
                "eligible": True,
            },
            {
                "seed_workers": 2,
                "status": "failed",
                "equivalence": {"exact": False, "runs": {}},
                "memory_safe": False,
                "eligible": False,
            },
            {
                "seed_workers": 3,
                "status": "failed",
                "equivalence": {"exact": False, "runs": {}},
                "memory_safe": False,
                "eligible": False,
            },
        ],
        "selected_profile": {
            "seed_workers": 1,
            "elapsed_seconds": 10.0,
            "speedup_fraction_vs_serial": 0.0,
            "cuda_reserved_fraction_upper_bound": 0.1,
        },
    }


def _write_profile(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exact_run_details() -> dict[str, dict[str, bool]]:
    return {
        f"full/seed-{seed}": {
            "exact": True,
            "checkpoint_semantics_exact": True,
            "held_out_metric_exact": True,
        }
        for seed in (4121, 7319, 104729)
    }


def test_readout_runtime_profile_rejects_unregistered_or_ineligible_selection(
    tmp_path: Path,
) -> None:
    config = _config()
    path = tmp_path / "readout.json"
    payload = _profile(config)
    payload["selected_profile"]["seed_workers"] = 4
    _write_profile(path, payload)
    with pytest.raises(ValueError, match="unregistered"):
        readout_runtime_profile_identity(path)

    payload = _profile(config)
    payload["selected_profile"]["seed_workers"] = 2
    _write_profile(path, payload)
    with pytest.raises(ValueError, match="not eligible"):
        readout_runtime_profile_identity(path)


def test_readout_runtime_profile_rejects_tampered_candidate_eligibility(
    tmp_path: Path,
) -> None:
    config = _config()
    path = tmp_path / "readout.json"
    payload = _profile(config)
    candidate = payload["candidates"][1]
    candidate.update(
        {
            "status": "completed",
            "elapsed_seconds": 9.9,
            "speedup_fraction_vs_serial": 10.0 / 9.9 - 1.0,
            "equivalence": {"exact": True, "runs": _exact_run_details()},
            "cuda_reserved_fraction_upper_bound": 0.2,
            "available_ram_bytes": 128 * 1024**3,
            "worker_cache_budget_bytes": 2 * 1024**3,
            "required_ram_with_reserve_bytes": 66 * 1024**3,
            "concurrent_cuda_peak_reserved_upper_bound_bytes": 2,
            "cuda_total_memory_bytes": 10,
            "per_seed_cuda_peak_reserved_bytes": [2, 2, 2],
            "memory_measurement_complete": True,
            "memory_safe": True,
            "eligible": True,
        }
    )
    _write_profile(path, payload)
    with pytest.raises(ValueError, match="eligibility mismatch"):
        readout_runtime_profile_identity(path)


def test_readout_runtime_profile_requires_fastest_eligible_candidate(
    tmp_path: Path,
) -> None:
    config = _config()
    path = tmp_path / "readout.json"
    payload = _profile(config)
    candidate = payload["candidates"][1]
    candidate.update(
        {
            "status": "completed",
            "elapsed_seconds": 8.0,
            "speedup_fraction_vs_serial": 0.25,
            "equivalence": {"exact": True, "runs": _exact_run_details()},
            "cuda_reserved_fraction_upper_bound": 0.2,
            "available_ram_bytes": 128 * 1024**3,
            "worker_cache_budget_bytes": 2 * 1024**3,
            "required_ram_with_reserve_bytes": 66 * 1024**3,
            "concurrent_cuda_peak_reserved_upper_bound_bytes": 2,
            "cuda_total_memory_bytes": 10,
            "per_seed_cuda_peak_reserved_bytes": [2, 2, 2],
            "memory_measurement_complete": True,
            "memory_safe": True,
            "eligible": True,
        }
    )
    _write_profile(path, payload)
    with pytest.raises(ValueError, match="fastest eligible"):
        readout_runtime_profile_identity(path)


def test_existing_readout_gate_requires_same_clean_revision_and_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    provenance = {
        "code_revision": "revision",
        "code_tree_sha256": "tree",
        "code_dirty": False,
    }
    monkeypatch.setattr(
        "fieldscope.readout_runtime_gate.code_provenance",
        lambda: provenance,
    )
    path = tmp_path / "readout.json"
    _write_profile(path, _profile(config))
    assert load_readout_runtime_gate_report(config, path)["status"] == "passed"

    payload = _profile(config)
    payload["code_revision"] = "stale"
    _write_profile(path, payload)
    with pytest.raises(ValueError, match="code_revision"):
        load_readout_runtime_gate_report(config, path)

    _write_profile(path, _profile(config))
    with pytest.raises(ValueError, match="readout_execution_contract_sha256"):
        load_readout_runtime_gate_report(_config(hidden_dim=128), path)


def test_checkpoint_semantics_ignore_only_wall_clock_fields() -> None:
    first = {
        "model": {"weight": 1},
        "optimizer": {"step": 2},
        "scheduler": {"last_epoch": 3},
        "rng_state": {"torch": [4]},
        "epoch": 4,
        "history": [
            {
                "epoch": 1,
                "loss": 0.5,
                "train_seconds": 10.0,
                "train_samples_per_second": 2.0,
                "validation_seconds": 3.0,
            }
        ],
        "best_epoch": 1,
        "best_primary_metric": 0.7,
    }
    repeated = json.loads(json.dumps(first))
    repeated["history"][0].update(
        {
            "train_seconds": 20.0,
            "train_samples_per_second": 1.0,
            "validation_seconds": 6.0,
        }
    )
    assert _exact_equal(_semantic_checkpoint(first), _semantic_checkpoint(repeated))
    repeated["history"][0]["loss"] = 0.6
    assert not _exact_equal(_semantic_checkpoint(first), _semantic_checkpoint(repeated))
    repeated = json.loads(json.dumps(first))
    repeated["config"] = {"changed": True}
    assert not _exact_equal(_semantic_checkpoint(first), _semantic_checkpoint(repeated))


def test_full_classification_gate_covers_registered_formal_workloads() -> None:
    envelope = formal_readout_workload_envelope(_config())
    assert envelope["representation"] == "full"
    assert envelope["gate"]["batch_size"] == 128
    assert envelope["gate_to_maximum_formal_ratio"] >= 1.0
    assert [workload["dataset"] for workload in envelope["formal_workloads"]] == [
        "imagenet100",
        "voc2012",
        "ade20k",
        "nyuv2",
    ]


def test_candidate_failures_fall_back_to_exact_serial_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    provenance = {
        "code_revision": "revision",
        "code_tree_sha256": "tree",
        "code_dirty": False,
    }
    monkeypatch.setattr(
        "fieldscope.readout_runtime_gate.code_provenance",
        lambda: provenance,
    )
    monkeypatch.setattr("fieldscope.readout_runtime_gate.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "fieldscope.readout_runtime_gate._available_ram_bytes",
        lambda: 128 * 1024**3,
    )

    def fake_candidate(**kwargs):
        workers = kwargs["workers"]
        if workers > 1:
            raise RuntimeError(f"workers={workers} OOM")
        return 10.0, tmp_path / "workers-1.log"

    monkeypatch.setattr("fieldscope.readout_runtime_gate._run_candidate", fake_candidate)
    monkeypatch.setattr(
        "fieldscope.readout_runtime_gate._candidate_memory",
        lambda *_args: {
            "per_seed_cuda_peak_reserved_bytes": [1, 1, 1],
            "memory_measurement_complete": True,
            "concurrent_cuda_peak_reserved_upper_bound_bytes": 1,
            "cuda_total_memory_bytes": 10,
            "cuda_reserved_fraction_upper_bound": 0.1,
        },
    )
    report = run_readout_runtime_gate(
        config,
        config_path=tmp_path / "config.yaml",
        train_cache_dir=tmp_path / "train",
        val_cache_dir=tmp_path / "val",
        test_cache_dir=tmp_path / "test",
        output_path=tmp_path / "readout.json",
    )
    assert report["selected_profile"]["seed_workers"] == 1
    assert [candidate["status"] for candidate in report["candidates"]] == [
        "completed",
        "failed",
        "failed",
    ]
