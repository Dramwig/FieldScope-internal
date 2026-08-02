import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fieldscope.cached_dataset import cached_control_contract_for_cache
from fieldscope.config import RunConfig
from fieldscope.evidence import audit_causal_evidence, audit_full_evidence
from fieldscope.experiments import READOUT_TRAINING_COVERAGE_CONTRACT
from fieldscope.readout_runtime_gate import (
    EQUIVALENCE_RULE,
    GATE_REPRESENTATION,
    MAX_CUDA_RESERVED_FRACTION,
    MIN_FREE_RAM_RESERVE_BYTES,
    MIN_SPEEDUP_FRACTION,
    SCHEMA_VERSION,
    formal_readout_workload_envelope,
    readout_execution_contract_sha256,
    readout_runtime_profile_identity,
)

SEEDS = [4121, 7319, 104729]
REPRESENTATIONS = [
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
]
TASKS = {
    "imagenet100": ("classification", "top1", True, (116455, 12940, 5000)),
    "voc2012": ("segmentation", "mean_iou", True, (1318, 146, 1449)),
    "ade20k": ("segmentation", "mean_iou", True, (18189, 2021, 2000)),
    "nyuv2": ("depth", "abs_rel", False, (715, 80, 654)),
}
READOUT_BUDGETS = {
    "imagenet100": (90, 128),
    "voc2012": (80, 4),
    "ade20k": (80, 2),
    "nyuv2": (80, 4),
}
UNSUPERVISED_CONTROLS = [
    "response_shuffled",
    "state",
    "z0",
    "velocity",
    "dit_hidden",
    "dit_attention",
]


def _sample_ids(dataset: str, split: str, count: int) -> list[str]:
    prefix = "voc" if dataset == "voc2012" and split == "test" else f"{dataset}-{split}"
    return [f"{prefix}-{index:08d}" for index in range(count)]


def _sample_id_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


class _TensorStub:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _mock_load_features(path: Path):
    manifest = json.loads((path.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    count = int(shard["num_samples"])
    baselines = {
        name: _TensorStub((count, 256, 768 if name == "dit_hidden" else 12), torch.bfloat16)
        for name in ("z0", "zt", "velocity", "mismatch", "endpoint", "dit_hidden", "trajectory")
    }
    graph_names = (
        ("dit_attention", "dit_attention_adjacency")
        if manifest["storage_policy"] == "dense"
        else ("dit_attention_adjacency",)
    )
    features = SimpleNamespace(
        grid_size=(16, 16),
        state=_TensorStub((count, 256, 14), torch.bfloat16),
        response=_TensorStub((count, 256, 192), torch.bfloat16),
        baselines=baselines,
        graphs={name: _TensorStub((count, 256, 256), torch.float32) for name in graph_names},
    )
    target_name = {
        "imagenet100": "classification",
        "voc2012": "segmentation",
        "ade20k": "segmentation",
        "nyuv2": "depth",
    }[manifest["dataset"]]
    target_dtype = {
        "classification": torch.int32,
        "segmentation": torch.uint8,
        "depth": torch.float32,
    }[target_name]
    target_shape = {
        "classification": (count,),
        "segmentation": (count, 512, 512),
        "depth": (count, 1, 512, 512),
    }[target_name]
    targets = {target_name: _TensorStub(target_shape, target_dtype)}
    shard_manifest = {
        "fingerprint": shard["fingerprint"],
        "storage_policy": manifest["storage_policy"],
        "metadata": {
            "backend": manifest["backend"],
            "probe": manifest["config"]["probe"],
        },
        "sample_ids": _sample_ids(manifest["dataset"], manifest["split"], count),
    }
    return features, targets, shard_manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cache(
    root: Path,
    dataset: str,
    split: str,
    count: int,
    provenance: dict[str, Any],
    *,
    storage_policy: str = "readout_sparse",
    probe_type: str = "structured",
    prompt: str = "",
    random_transformer: bool = False,
) -> Path:
    suffix = hashlib.sha256(f"{probe_type}|{prompt}|{random_transformer}".encode()).hexdigest()[:8]
    cache = root / f"{dataset}-{split}-{storage_policy}-{suffix}"
    cache.mkdir(parents=True)
    shard = cache / "shard-00000.pt"
    shard.write_bytes(b"cache")
    shard_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    sample_id_hash = _sample_id_hash(_sample_ids(dataset, split, count))
    _write_json(
        cache / "dataset_manifest.json",
        {
            "complete": True,
            "dataset": dataset,
            "split": split,
            "num_samples": count,
            "sample_ids_sha256": sample_id_hash,
            "storage_policy": storage_policy,
            **provenance,
            "backend": {
                "backend": "auraflow",
                "model_id": "/formal/checkpoints/AuraFlow-v0.3",
                "variant": "fp16",
                "device": "cuda",
                "dtype": "bfloat16",
                "image_size": 512,
                "prompt": prompt,
                "native_time": "1=noise, 0=image",
                "public_time": "clean_time: 0=noise, 1=image",
                "velocity_conversion": "public_velocity=-native_transformer_output",
                "frozen": True,
                "random_transformer": random_transformer,
                "random_transformer_seed": 104729 if random_transformer else None,
            },
            "config": {
                "backend": {
                    "name": "auraflow",
                    "model_path": "/formal/checkpoints/AuraFlow-v0.3",
                    "variant": "fp16",
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "image_size": 512,
                    "prompt": prompt,
                    "max_sequence_length": 256,
                    "local_files_only": True,
                    "offload_text_encoder": True,
                    "random_transformer": random_transformer,
                    "random_transformer_seed": 104729,
                },
                "probe": {
                    "times": [0.2, 0.5, 0.8],
                    "num_directions": 8,
                    "difference": "central",
                    "probe_type": probe_type,
                    "eta": 0.03,
                    "graph_grid": [16, 16],
                    "topk": 16,
                    "local_radius": 1,
                    "probe_batch_size": 8,
                    "antithetic_noise": True,
                    "hidden_baseline_dim": 768,
                    "seed": 4121,
                },
                "runtime": {
                    "batch_size": 2,
                    "cache_shard_size": 64,
                    "deterministic": True,
                },
                "tokenizer": {
                    "hidden_dim": 256,
                    "input_dim": 768,
                    "num_layers": 3,
                    "dropout": 0.1,
                },
            },
            "randomness": {
                "path_noise": "sample_id_sha256_seeded_v1",
                "probe_basis": "shared_fixed_seed_v1",
            },
            "shards": [
                {
                    "path": shard.name,
                    "num_samples": count,
                    "fingerprint": "formal-fingerprint",
                    "bytes": shard.stat().st_size,
                    "sha256": shard_sha256,
                }
            ],
        },
    )
    return cache


def _value(
    dataset: str,
    representation: str,
    seed_index: int,
    *,
    passing: bool,
) -> float:
    maximize = TASKS[dataset][2]
    if maximize:
        controls = 0.40 + seed_index * 0.001
        graph_controls = 0.50 + seed_index * 0.001
        candidates = {"response": 0.70, "full": 0.75}
    else:
        controls = 0.80 + seed_index * 0.001
        graph_controls = 0.60 + seed_index * 0.001
        candidates = {"response": 0.40, "full": 0.35}
    if representation in candidates:
        value = candidates[representation] + (
            seed_index * 0.001 if maximize else -seed_index * 0.001
        )
        if not passing and dataset in {"ade20k", "nyuv2"}:
            value = controls - 0.01 if maximize else controls + 0.01
        return value
    if representation in {
        "response_nograph",
        "response_local",
        "full_nograph",
        "full_local",
    }:
        return graph_controls
    if representation in {"response_shuffled", "full_shuffled"}:
        return 0.45 + seed_index * 0.001 if maximize else 0.70 + seed_index * 0.001
    return controls


def _formal_config_mapping() -> dict[str, Any]:
    return {
        "backend": {
            "name": "auraflow",
            "model_path": "/formal/AuraFlow-v0.3",
            "variant": "fp16",
            "device": "cuda",
            "dtype": "bfloat16",
            "image_size": 512,
        },
        "probe": {
            "times": [0.2, 0.5, 0.8],
            "num_directions": 8,
            "graph_grid": [16, 16],
            "topk": 16,
            "probe_batch_size": 8,
            "seed": 4121,
        },
        "tokenizer": {
            "hidden_dim": 256,
            "input_dim": 768,
            "num_layers": 3,
            "dropout": 0.1,
            "num_classes": 100,
            "segmentation_classes": 21,
        },
        "runtime": {
            "batch_size": 2,
            "cache_shard_size": 64,
            "readout_memory_cache_gib": 160,
            "num_workers": 0,
            "deterministic": True,
        },
    }


def _readout_profile(
    root: Path,
    provenance: dict[str, Any],
    config_mapping: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    config = RunConfig.from_mapping(config_mapping)
    path = root / "readout_runtime_profile.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        **provenance,
        "evidence_scope": "readout_seed_parallel_exactness_and_throughput_only",
        "method_effectiveness_conclusion": None,
        "equivalence_rule": EQUIVALENCE_RULE,
        "readout_execution_contract_sha256": readout_execution_contract_sha256(config),
        "formal_workload_envelope": formal_readout_workload_envelope(config),
        "registered_seed_workers": [1, 2, 3],
        "registered_seeds": SEEDS,
        "task": "classification",
        "representation": GATE_REPRESENTATION,
        "epochs": 20,
        "batch_size": 128,
        "minimum_speedup_fraction": MIN_SPEEDUP_FRACTION,
        "minimum_fast_path_speedup_fraction": MIN_SPEEDUP_FRACTION,
        "maximum_cuda_reserved_fraction": MAX_CUDA_RESERVED_FRACTION,
        "minimum_free_ram_reserve_bytes": MIN_FREE_RAM_RESERVE_BYTES,
        "available_ram_bytes": 700 * 1024**3,
        "strict_reference": {
            "status": "completed",
            "seed_workers": 1,
            "elapsed_seconds": 2.0,
            "fast_serial_speedup_fraction_vs_strict": 1.0,
            "equivalence_to_fast_serial": {
                "exact": True,
                "runs": {
                    f"full/seed-{seed}": {
                        "exact": True,
                        "checkpoint_semantics_exact": True,
                        "held_out_metric_exact": True,
                    }
                    for seed in SEEDS
                },
            },
            "matrix_report": "strict-reference/matrix_report.json",
            "log": "strict-reference.log",
        },
        "candidates": [
            {
                "seed_workers": 1,
                "status": "completed",
                "eligible": True,
                "equivalence": {"exact": True, "runs": {}},
                "memory_safe": True,
                "memory_measurement_complete": True,
                "elapsed_seconds": 1.0,
                "speedup_fraction_vs_serial": 0.0,
                "cuda_reserved_fraction_upper_bound": 0.1,
                "available_ram_bytes": 700 * 1024**3,
                "worker_cache_budget_bytes": 160 * 1024**3,
                "required_ram_with_reserve_bytes": 224 * 1024**3,
                "concurrent_cuda_peak_reserved_upper_bound_bytes": 1,
                "cuda_total_memory_bytes": 10,
                "per_seed_cuda_peak_reserved_bytes": [1, 1, 1],
            },
            {
                "seed_workers": 2,
                "status": "failed",
                "eligible": False,
                "equivalence": {"exact": False, "runs": {}},
            },
            {
                "seed_workers": 3,
                "status": "failed",
                "eligible": False,
                "equivalence": {"exact": False, "runs": {}},
            },
        ],
        "selected_profile": {
            "seed_workers": 1,
            "elapsed_seconds": 1.0,
            "speedup_fraction_vs_serial": 0.0,
            "cuda_reserved_fraction_upper_bound": 0.1,
        },
    }
    _write_json(path, payload)
    return path, readout_runtime_profile_identity(path)


def _matrix(
    root: Path,
    dataset: str,
    provenance: dict[str, Any],
    readout_runtime_profile: dict[str, Any],
    *,
    passing: bool,
) -> Path:
    task, metric, _maximize, counts = TASKS[dataset]
    epochs, batch_size = READOUT_BUDGETS[dataset]
    cache_dirs = {
        split: _cache(root, dataset, split, count, provenance)
        for split, count in zip(("train", "val", "test"), counts, strict=True)
    }
    runs = []
    matrix_root = root / "matrices" / dataset
    config = _formal_config_mapping()
    cache_identities = {
        split: {
            "path": str(cache.resolve()),
            "manifest_sha256": hashlib.sha256(
                (cache / "dataset_manifest.json").read_bytes()
            ).hexdigest(),
            "extraction_signature": None,
            "dataset": dataset,
            "split": split,
            "num_samples": count,
            "sample_ids_sha256": json.loads(
                (cache / "dataset_manifest.json").read_text(encoding="utf-8")
            )["sample_ids_sha256"],
            "storage_policy": "readout_sparse",
            "code_revision": provenance["code_revision"],
            "code_tree_sha256": provenance["code_tree_sha256"],
        }
        for (split, count), cache in zip(
            zip(("train", "val", "test"), counts, strict=True),
            cache_dirs.values(),
            strict=True,
        )
    }
    for representation in REPRESENTATIONS:
        for seed_index, seed in enumerate(SEEDS):
            control_contracts = {
                split: cached_control_contract_for_cache(cache_dirs[split], representation, seed)
                for split in ("train", "val", "test")
            }
            run_root = matrix_root / representation / f"seed-{seed}"
            checkpoint = run_root / "best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            train_samples = counts[0]
            optimizer_steps_per_epoch = math.ceil(train_samples / batch_size)
            completed_optimizer_steps = optimizer_steps_per_epoch * epochs
            completed_training_sample_exposures = train_samples * epochs
            history = [
                {
                    "epoch": epoch,
                    "train_samples": train_samples,
                    "optimizer_steps": optimizer_steps_per_epoch,
                }
                for epoch in range(1, epochs + 1)
            ]
            last_checkpoint = run_root / "last.pt"
            torch.save(
                {
                    "epoch": epochs,
                    "target_epochs": epochs,
                    "batch_size": batch_size,
                    "training_coverage_contract": READOUT_TRAINING_COVERAGE_CONTRACT,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "completed_optimizer_steps": completed_optimizer_steps,
                    "completed_training_sample_exposures": (
                        completed_training_sample_exposures
                    ),
                    "history": history,
                    "train_cache": cache_identities["train"],
                    "validation_cache": cache_identities["val"],
                    "code_revision": provenance["code_revision"],
                    "code_tree_sha256": provenance["code_tree_sha256"],
                    "optimizer": {
                        "state": {0: {"step": torch.tensor(completed_optimizer_steps)}}
                    },
                },
                last_checkpoint,
            )
            last_checkpoint_sha256 = hashlib.sha256(last_checkpoint.read_bytes()).hexdigest()
            value = _value(
                dataset,
                representation,
                seed_index,
                passing=passing,
            )
            training = run_root / "training.json"
            test = run_root / "test.json"
            _write_json(
                training,
                {
                    "status": "passed",
                    **provenance,
                    "task": task,
                    "representation": representation,
                    "seed": seed,
                    "config": config,
                    "epochs": epochs,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "batch_size": batch_size,
                    "train_cache": cache_identities["train"],
                    "validation_cache": cache_identities["val"],
                    "train_control_contract": control_contracts["train"],
                    "validation_control_contract": control_contracts["val"],
                    "train_samples": train_samples,
                    "training_coverage_contract": READOUT_TRAINING_COVERAGE_CONTRACT,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "completed_optimizer_steps": completed_optimizer_steps,
                    "completed_training_sample_exposures": (
                        completed_training_sample_exposures
                    ),
                    "trainable_parameters": 123456,
                    "best_checkpoint": str(checkpoint),
                    "best_checkpoint_sha256": checkpoint_sha256,
                    "last_checkpoint": str(last_checkpoint),
                    "last_checkpoint_sha256": last_checkpoint_sha256,
                    "history": history,
                },
            )
            _write_json(
                test,
                {
                    "status": "passed",
                    **provenance,
                    "task": task,
                    "representation": representation,
                    "seed": seed,
                    "config": config,
                    "batch_size": batch_size,
                    "test_cache": cache_identities["test"],
                    "test_control_contract": control_contracts["test"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "evaluation": {"metrics": {metric: value}},
                },
            )
            runs.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "training_report": str(training),
                    "test_report": str(test),
                    "best_checkpoint": str(checkpoint),
                    "best_checkpoint_sha256": checkpoint_sha256,
                    "test_metric": value,
                    "completed_optimizer_steps": completed_optimizer_steps,
                    "completed_training_sample_exposures": (
                        completed_training_sample_exposures
                    ),
                }
            )
    matrix = matrix_root / "matrix_report.json"
    _write_json(
        matrix,
        {
            "status": "passed",
            **provenance,
            "task": task,
            "metric": metric,
            "representations": REPRESENTATIONS,
            "seeds": SEEDS,
            "epochs": epochs,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": batch_size,
            "config": config,
            "train_cache_dir": str(cache_dirs["train"]),
            "val_cache_dir": str(cache_dirs["val"]),
            "test_cache_dir": str(cache_dirs["test"]),
            "train_cache": cache_identities["train"],
            "validation_cache": cache_identities["val"],
            "test_cache": cache_identities["test"],
            "readout_runtime_profile": readout_runtime_profile,
            "completed_optimizer_steps": sum(
                int(run["completed_optimizer_steps"]) for run in runs
            ),
            "completed_training_sample_exposures": sum(
                int(run["completed_training_sample_exposures"]) for run in runs
            ),
            "runs": runs,
        },
    )
    return matrix


def _unsupervised(
    root: Path,
    provenance: dict[str, Any],
    *,
    passing: bool,
) -> Path:
    cache = _cache(
        root,
        "voc2012",
        "test",
        1449,
        provenance,
        storage_policy="dense",
    )
    per_sample = []
    for index in range(1449):
        representations = {
            "response": {
                "boundary_average_precision": 0.8,
                "pairwise_auroc": 0.85,
            }
        }
        for control in UNSUPERVISED_CONTROLS:
            value = 0.5
            if not passing and control == "dit_hidden":
                value = 0.9
            representations[control] = {
                "boundary_average_precision": value,
                "pairwise_auroc": value,
            }
        per_sample.append({"sample_id": f"voc-{index:08d}", "representations": representations})
    path = root / "voc_unsupervised.json"
    _write_json(
        path,
        {
            "status": "passed",
            **provenance,
            "cache_dir": str(cache),
            "num_samples": 1449,
            "per_sample": per_sample,
        },
    )
    return path


def _evidence_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passing: bool,
) -> tuple[
    dict[str, Path],
    Path,
    Path,
    dict[str, Path],
    dict[str, Any],
    Path,
]:
    provenance = {
        "code_revision": "formal-revision",
        "code_dirty": False,
        "code_tree_sha256": "formal-tree",
    }
    monkeypatch.setattr("fieldscope.evidence.code_provenance", lambda: provenance)
    monkeypatch.setattr(
        "fieldscope.readout_runtime_gate.code_provenance",
        lambda: provenance,
    )
    monkeypatch.setattr("fieldscope.evidence.load_features", _mock_load_features)
    monkeypatch.setattr(
        "fieldscope.evidence.bootstrap_mean_interval",
        lambda values, **_kwargs: {
            "mean": mean(values),
            "ci95": [min(values), max(values)],
            "resamples": 2000,
        },
    )
    readout_profile, readout_profile_identity = _readout_profile(
        tmp_path,
        provenance,
        _formal_config_mapping(),
    )
    matrices = {
        dataset: _matrix(
            tmp_path,
            dataset,
            provenance,
            readout_profile_identity,
            passing=passing,
        )
        for dataset in TASKS
    }
    unsupervised = _unsupervised(tmp_path, provenance, passing=passing)
    backbone_asset = tmp_path / "backbone_asset.json"
    _write_json(
        backbone_asset,
        {
            "status": "passed",
            **provenance,
            "random_transformer": False,
        },
    )
    split_audits = {}
    for dataset, (_task, _metric, _maximize, counts) in TASKS.items():
        split_path = tmp_path / f"{dataset}_split_audit.json"
        matrix_report = json.loads(matrices[dataset].read_text(encoding="utf-8"))
        cache_keys = {
            "train": "train_cache",
            "val": "validation_cache",
            "test": "test_cache",
        }
        _write_json(
            split_path,
            {
                "status": "passed",
                **provenance,
                "dataset": dataset,
                "splits": {
                    split: {
                        "count": count,
                        "sample_ids_sha256": matrix_report[cache_keys[split]]["sample_ids_sha256"],
                    }
                    for split, count in zip(
                        ("train", "val", "test"),
                        counts,
                        strict=True,
                    )
                },
            },
        )
        split_audits[dataset] = split_path
    return (
        matrices,
        unsupervised,
        backbone_asset,
        split_audits,
        provenance,
        readout_profile,
    )


def test_full_evidence_supports_only_complete_cross_task_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=True,
    )
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "passed"
    assert report["verdict"] == "main_tasks_supported_pending_causal_audits"
    assert report["requires_causal_audits"] is True
    assert report["unsupervised"]["passed"] is True
    assert report["supervised"]["full"]["supports_cross_task_hypothesis"] is True
    assert report["training_execution"]["required_optimizer_steps"] == 51_013_200
    assert report["training_execution"]["reported_completed_optimizer_steps"] == 51_013_200
    assert report["training_execution"]["required_training_sample_exposures"] == 725_922_600
    assert (
        report["training_execution"]["reported_completed_training_sample_exposures"]
        == 725_922_600
    )
    comparison = report["supervised"]["full"]["tasks"]["imagenet100"]["graph_controls"]
    assert comparison["ci95"][0] > 0
    assert comparison["paired_sign_flip_pvalue"] == 0.25
    assert comparison["holm_reject_alpha_0_05"] is False


def test_full_evidence_reports_limited_or_negative_without_cross_task_gain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=False,
    )
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "failed"
    assert report["verdict"] == "limited_or_negative"
    assert report["supports_strong_claims"] is False


def test_full_evidence_is_incomplete_for_stale_or_nonfinite_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=True,
    )
    matrix = json.loads(matrices["imagenet100"].read_text(encoding="utf-8"))
    matrix["runs"][0]["test_metric"] = float("nan")
    matrices["imagenet100"].write_text(json.dumps(matrix), encoding="utf-8")
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "incomplete"
    assert report["verdict"] == "incomplete"
    assert any("non-finite test metric" in problem for problem in report["problems"])


def test_full_evidence_rejects_unmatched_readout_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=True,
    )
    matrix = json.loads(matrices["imagenet100"].read_text(encoding="utf-8"))
    training_path = Path(matrix["runs"][0]["training_report"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["trainable_parameters"] += 1
    training_path.write_text(json.dumps(training), encoding="utf-8")
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "incomplete"
    assert report["verdict"] == "incomplete"
    assert any(
        "readout parameter counts are not matched" in problem
        for problem in report["problems"]
    )


def test_full_evidence_rejects_incomplete_optimizer_step_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=True,
    )
    matrix = json.loads(matrices["imagenet100"].read_text(encoding="utf-8"))
    training_path = Path(matrix["runs"][0]["training_report"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["history"][-1]["optimizer_steps"] -= 1
    training["completed_optimizer_steps"] -= 1
    last_checkpoint = Path(training["last_checkpoint"])
    last_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    last_payload["optimizer"]["state"][0]["step"] -= 1
    torch.save(last_payload, last_checkpoint)
    training["last_checkpoint_sha256"] = hashlib.sha256(last_checkpoint.read_bytes()).hexdigest()
    training_path.write_text(json.dumps(training), encoding="utf-8")
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "incomplete"
    assert report["verdict"] == "incomplete"
    assert any("training epoch optimizer-step mismatch" in item for item in report["problems"])
    assert any("last checkpoint AdamW step-state mismatch" in item for item in report["problems"])


def test_full_evidence_rejects_tampered_response_shuffle_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=True,
    )
    matrix = json.loads(matrices["imagenet100"].read_text(encoding="utf-8"))
    shuffled = next(run for run in matrix["runs"] if run["representation"] == "response_shuffled")
    training_path = Path(shuffled["training_report"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["train_control_contract"]["response_shuffle"]["donor_permutation_sha256"] = "0" * 64
    training_path.write_text(json.dumps(training), encoding="utf-8")
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "incomplete"
    assert report["verdict"] == "incomplete"
    assert any("training control contract mismatch" in problem for problem in report["problems"])


def test_full_evidence_requires_readout_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, _ = _evidence_inputs(
        tmp_path,
        monkeypatch,
        passing=True,
    )
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
    )
    assert report["status"] == "incomplete"
    assert "formal evidence audit requires a readout runtime profile" in report["problems"]


def test_full_evidence_rejects_stale_readout_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path, monkeypatch, passing=True
    )
    payload = json.loads(readout_profile.read_text(encoding="utf-8"))
    payload["code_revision"] = "stale"
    _write_json(readout_profile, payload)
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "incomplete"
    assert any("invalid readout runtime profile" in problem for problem in report["problems"])


def test_full_evidence_rejects_matrix_readout_profile_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices, unsupervised, backbone_asset, split_audits, _, readout_profile = _evidence_inputs(
        tmp_path, monkeypatch, passing=True
    )
    payload = json.loads(readout_profile.read_text(encoding="utf-8"))
    payload["candidates"][0]["elapsed_seconds"] = 2.0
    payload["selected_profile"]["elapsed_seconds"] = 2.0
    _write_json(readout_profile, payload)
    report = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    assert report["status"] == "incomplete"
    assert any(
        "matrix readout runtime profile mismatch" in problem for problem in report["problems"]
    )


def _causal_report(
    root: Path,
    provenance: dict[str, Any],
    name: str,
    *,
    response_value: float,
    probe_type: str = "structured",
    prompt: str = "",
    random_transformer: bool = False,
) -> Path:
    cache = _cache(
        root / name,
        "voc2012",
        "test",
        1449,
        provenance,
        storage_policy="dense",
        probe_type=probe_type,
        prompt=prompt,
        random_transformer=random_transformer,
    )
    per_sample = []
    for index in range(1449):
        representations = {
            "response": {
                "boundary_average_precision": response_value,
                "pairwise_auroc": response_value,
            }
        }
        for control in (
            "response_shuffled",
            "state",
            "z0",
            "velocity",
            "dit_hidden",
            "dit_attention",
        ):
            representations[control] = {
                "boundary_average_precision": 0.50,
                "pairwise_auroc": 0.50,
            }
        per_sample.append(
            {
                "sample_id": f"voc-{index:08d}",
                "representations": representations,
            }
        )
    path = root / f"{name}.json"
    _write_json(
        path,
        {
            "status": "passed",
            **provenance,
            "cache_dir": str(cache),
            "num_samples": 1449,
            "per_sample": per_sample,
        },
    )
    return path


def test_causal_evidence_is_the_only_strong_claim_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        matrices,
        unsupervised,
        backbone_asset,
        split_audits,
        provenance,
        readout_profile,
    ) = _evidence_inputs(tmp_path, monkeypatch, passing=True)
    main = audit_full_evidence(
        matrix_paths=matrices,
        voc_unsupervised_path=unsupervised,
        backbone_asset_path=backbone_asset,
        split_audit_paths=split_audits,
        readout_runtime_profile_path=readout_profile,
    )
    main_path = tmp_path / "main_evidence.json"
    _write_json(main_path, main)
    causal_reports = {
        "random_flow": _causal_report(
            tmp_path,
            provenance,
            "random_flow",
            response_value=0.40,
            random_transformer=True,
        ),
        "spatially_shuffled_probe": _causal_report(
            tmp_path,
            provenance,
            "spatially_shuffled_probe",
            response_value=0.45,
            probe_type="spatially_shuffled",
        ),
        "neutral_prompt": _causal_report(
            tmp_path,
            provenance,
            "neutral_prompt",
            response_value=0.80,
            prompt="a neutral photograph",
        ),
        "unrelated_prompt": _causal_report(
            tmp_path,
            provenance,
            "unrelated_prompt",
            response_value=0.75,
            prompt="an unrelated scene",
        ),
    }
    report = audit_causal_evidence(
        main_evidence_path=main_path,
        causal_report_paths=causal_reports,
    )
    assert report["status"] == "passed"
    assert report["verdict"] == "supports_core_hypothesis"
    assert report["supports_strong_claims"] is True
    random_comparison = report["comparisons"][
        "pretrained-structured-minus-random_flow/boundary_average_precision"
    ]
    assert random_comparison["ci95"][0] > 0


def test_causal_evidence_rejects_failed_random_flow_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        matrices,
        unsupervised,
        backbone_asset,
        split_audits,
        provenance,
        readout_profile,
    ) = _evidence_inputs(tmp_path, monkeypatch, passing=True)
    main_path = tmp_path / "main_evidence.json"
    _write_json(
        main_path,
        audit_full_evidence(
            matrix_paths=matrices,
            voc_unsupervised_path=unsupervised,
            backbone_asset_path=backbone_asset,
            split_audit_paths=split_audits,
            readout_runtime_profile_path=readout_profile,
        ),
    )
    causal_reports = {
        "random_flow": _causal_report(
            tmp_path,
            provenance,
            "random_flow",
            response_value=0.90,
            random_transformer=True,
        ),
        "spatially_shuffled_probe": _causal_report(
            tmp_path,
            provenance,
            "spatially_shuffled_probe",
            response_value=0.45,
            probe_type="spatially_shuffled",
        ),
        "neutral_prompt": _causal_report(
            tmp_path,
            provenance,
            "neutral_prompt",
            response_value=0.80,
            prompt="a neutral photograph",
        ),
        "unrelated_prompt": _causal_report(
            tmp_path,
            provenance,
            "unrelated_prompt",
            response_value=0.75,
            prompt="an unrelated scene",
        ),
    }
    report = audit_causal_evidence(
        main_evidence_path=main_path,
        causal_report_paths=causal_reports,
    )
    assert report["status"] == "failed"
    assert report["verdict"] == "main_task_gain_not_causally_attributed"
    assert report["supports_strong_claims"] is False


def test_causal_evidence_finishes_registered_controls_after_negative_main_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        matrices,
        unsupervised,
        backbone_asset,
        split_audits,
        provenance,
        readout_profile,
    ) = _evidence_inputs(tmp_path, monkeypatch, passing=False)
    main_path = tmp_path / "main_evidence.json"
    _write_json(
        main_path,
        audit_full_evidence(
            matrix_paths=matrices,
            voc_unsupervised_path=unsupervised,
            backbone_asset_path=backbone_asset,
            split_audit_paths=split_audits,
            readout_runtime_profile_path=readout_profile,
        ),
    )
    causal_reports = {
        "random_flow": _causal_report(
            tmp_path,
            provenance,
            "random_flow",
            response_value=0.40,
            random_transformer=True,
        ),
        "spatially_shuffled_probe": _causal_report(
            tmp_path,
            provenance,
            "spatially_shuffled_probe",
            response_value=0.45,
            probe_type="spatially_shuffled",
        ),
        "neutral_prompt": _causal_report(
            tmp_path,
            provenance,
            "neutral_prompt",
            response_value=0.80,
            prompt="a neutral photograph",
        ),
        "unrelated_prompt": _causal_report(
            tmp_path,
            provenance,
            "unrelated_prompt",
            response_value=0.75,
            prompt="an unrelated scene",
        ),
    }
    report = audit_causal_evidence(
        main_evidence_path=main_path,
        causal_report_paths=causal_reports,
    )
    assert report["status"] == "failed"
    assert report["verdict"] == "limited_or_negative"
    assert report["main_evidence_verdict"] == "limited_or_negative"
    assert report["comparisons"]
    assert report["supports_strong_claims"] is False


def test_causal_evidence_marks_nonfinite_metrics_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        matrices,
        unsupervised,
        backbone_asset,
        split_audits,
        provenance,
        readout_profile,
    ) = _evidence_inputs(tmp_path, monkeypatch, passing=True)
    main_path = tmp_path / "main_evidence.json"
    _write_json(
        main_path,
        audit_full_evidence(
            matrix_paths=matrices,
            voc_unsupervised_path=unsupervised,
            backbone_asset_path=backbone_asset,
            split_audit_paths=split_audits,
            readout_runtime_profile_path=readout_profile,
        ),
    )
    causal_reports = {
        "random_flow": _causal_report(
            tmp_path,
            provenance,
            "random_flow",
            response_value=0.40,
            random_transformer=True,
        ),
        "spatially_shuffled_probe": _causal_report(
            tmp_path,
            provenance,
            "spatially_shuffled_probe",
            response_value=0.45,
            probe_type="spatially_shuffled",
        ),
        "neutral_prompt": _causal_report(
            tmp_path,
            provenance,
            "neutral_prompt",
            response_value=float("nan"),
            prompt="a neutral photograph",
        ),
        "unrelated_prompt": _causal_report(
            tmp_path,
            provenance,
            "unrelated_prompt",
            response_value=0.75,
            prompt="an unrelated scene",
        ),
    }
    report = audit_causal_evidence(
        main_evidence_path=main_path,
        causal_report_paths=causal_reports,
    )
    assert report["status"] == "incomplete"
    assert report["verdict"] == "incomplete"
    assert any("non-finite condition metric" in problem for problem in report["problems"])
