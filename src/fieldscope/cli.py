"""Command-line entry points for validation, extraction, and diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fieldscope.assets import audit_backbone_assets
from fieldscope.backends import build_backend
from fieldscope.cache import load_features, save_features
from fieldscope.config import RunConfig, load_config
from fieldscope.dataset_audit import audit_dataset_splits
from fieldscope.datasets import SyntheticShapesDataset
from fieldscope.diagnostics import graph_diagnostics
from fieldscope.evidence import audit_causal_evidence, audit_full_evidence
from fieldscope.experiments import (
    atomic_json_dump,
    diagnose_segmentation_cache,
    evaluate_checkpoint,
    extract_dataset_cache,
    run_readout_matrix_seed_parallel,
    set_experiment_seed,
    train_cached_readout,
)
from fieldscope.extension import (
    audit_extension_evidence,
    audit_final_evidence,
    audit_imagenet1k_asset,
    build_high_cost_ablation_configs,
)
from fieldscope.gates import audit_signal_gate
from fieldscope.losses import multitask_loss
from fieldscope.model import FieldScopeModel
from fieldscope.readout_runtime_gate import (
    load_readout_runtime_gate_report,
    readout_runtime_profile_identity,
    run_readout_runtime_gate,
)
from fieldscope.resource_planning import plan_cache_budget
from fieldscope.response import FieldResponseExtractor
from fieldscope.runtime_gate import load_runtime_gate_report, run_runtime_gate
from fieldscope.statistics import summarize_run_reports


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    atomic_json_dump(path, payload)


def _set_seed(seed: int, deterministic: bool) -> None:
    set_experiment_seed(seed, deterministic)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def doctor(config: RunConfig) -> dict[str, Any]:
    model_path = Path(config.backend.model_path) if config.backend.model_path else None
    output_parent = Path(config.runtime.output_dir).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_parent)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "pyyaml": _package_version("PyYAML"),
        "diffusers": _package_version("diffusers"),
        "transformers": _package_version("transformers"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "backend": config.backend.name,
        "model_path": str(model_path) if model_path else None,
        "model_path_exists": model_path.exists() if model_path else None,
        "output_parent": str(output_parent),
        "output_free_bytes": disk.free,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "FIELDSCOPE_ROOT",
                "FIELDSCOPE_DATASETS_ROOT",
                "FIELDSCOPE_CHECKPOINTS_ROOT",
                "HF_HOME",
            )
        },
    }
    if config.backend.name == "auraflow":
        report["ready"] = bool(
            torch.cuda.is_available()
            and report["diffusers"]
            and report["transformers"]
            and report["model_path_exists"]
        )
    else:
        report["ready"] = True
    return report


def _synthetic_batch(config: RunConfig) -> dict[str, torch.Tensor | list[str]]:
    dataset = SyntheticShapesDataset(
        length=max(config.runtime.batch_size, 4),
        image_size=config.backend.image_size,
        seed=config.probe.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
    )
    return next(iter(loader))


def run_smoke(config: RunConfig, steps: int) -> dict[str, Any]:
    total_started = time.perf_counter()
    _set_seed(config.probe.seed, config.runtime.deterministic)
    backend_started = time.perf_counter()
    backend = build_backend(config.backend)
    backend_seconds = time.perf_counter() - backend_started
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    extractor = FieldResponseExtractor(backend, config.probe)
    batch = _synthetic_batch(config)
    images = batch["image"]
    assert isinstance(images, torch.Tensor)
    extraction_started = time.perf_counter()
    features = extractor.extract(images)
    extraction_seconds = time.perf_counter() - extraction_started
    features_for_model = features.to(backend.device, dtype=torch.float32)
    model = FieldScopeModel(
        state_dim=features.state.shape[-1],
        response_dim=features.response.shape[-1],
        config=config.tokenizer,
    ).to(backend.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = {
        name: tensor.to(backend.device)
        for name, tensor in batch.items()
        if name in {"classification", "segmentation", "depth", "normals"}
        and isinstance(tensor, torch.Tensor)
    }
    losses: list[float] = []
    components: dict[str, float] = {}
    training_started = time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            features_for_model,
            output_size=(config.backend.image_size, config.backend.image_size),
        )
        total, pieces = multitask_loss(predictions, targets)
        total.backward()
        optimizer.step()
        losses.append(float(total.detach().item()))
        components = {name: float(value.detach().item()) for name, value in pieces.items()}
    training_seconds = time.perf_counter() - training_started

    cache_path = Path(config.runtime.output_dir) / "smoke_features.pt"
    manifest = save_features(
        cache_path,
        features,
        targets={name: value.cpu() for name, value in targets.items()},
        sample_ids=list(batch["sample_id"]),
    )
    reloaded, _, _ = load_features(cache_path)
    runtime_stats = getattr(backend, "runtime_stats", lambda: {})()
    return {
        "status": "passed",
        "backend": backend.describe(),
        "feature_shapes": {
            "state": list(features.state.shape),
            "response": list(features.response.shape),
            "affinity": list(features.affinity.shape),
            "baselines": {name: list(value.shape) for name, value in features.baselines.items()},
            "graphs": {name: list(value.shape) for name, value in features.graphs.items()},
        },
        "diagnostics": graph_diagnostics(features),
        "train_steps": steps,
        "losses": losses,
        "loss_components": components,
        "cache": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "cache_fingerprint": manifest["fingerprint"],
        "cache_reload_equal": bool(torch.equal(features.state.cpu(), reloaded.state)),
        "runtime": {
            "backend_initialization_seconds": backend_seconds,
            "feature_extraction_seconds": extraction_seconds,
            "readout_training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(backend.device)
                if backend.device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(backend.device)
                if backend.device.type == "cuda"
                else None
            ),
            **runtime_stats,
        },
    }


def _load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def extract_images(config: RunConfig, image_paths: list[Path], output: Path) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("At least one image is required")
    images = torch.stack(
        [_load_image(path, config.backend.image_size) for path in image_paths], dim=0
    )
    backend = build_backend(config.backend)
    features = FieldResponseExtractor(backend, config.probe).extract(images)
    manifest = save_features(
        output,
        features,
        sample_ids=[str(path.resolve()) for path in image_paths],
    )
    return {"status": "passed", "output": str(output), "manifest": manifest}


def extract_dataset(
    config: RunConfig,
    *,
    dataset_name: str,
    dataset_root: Path,
    split: str,
    output_dir: Path,
    limit: int | None,
    offset: int = 0,
    resume: bool = False,
    class_names: list[str] | None = None,
    storage_policy: str = "dense",
) -> dict[str, Any]:
    return extract_dataset_cache(
        config,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        split=split,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        resume=resume,
        class_names=class_names,
        storage_policy=storage_policy,
    )


def train_cache(
    config: RunConfig,
    *,
    cache_dir: Path,
    task: str,
    representation: str,
    epochs: int,
    learning_rate: float,
    val_cache_dir: Path | None = None,
    output_dir: Path | None = None,
    weight_decay: float = 1e-4,
    seed: int | None = None,
    resume_checkpoint: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    return train_cached_readout(
        config,
        train_cache_dir=cache_dir,
        val_cache_dir=val_cache_dir or cache_dir,
        output_dir=output_dir or Path(config.runtime.output_dir),
        task=task,
        representation=representation,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=config.probe.seed if seed is None else seed,
        resume_checkpoint=resume_checkpoint,
        batch_size=batch_size,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fieldscope")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime and configured assets")
    doctor_parser.add_argument("--config", required=True, type=Path)
    doctor_parser.add_argument("--output", type=Path)

    smoke_parser = subparsers.add_parser("smoke", help="Run end-to-end synthetic smoke test")
    smoke_parser.add_argument("--config", required=True, type=Path)
    smoke_parser.add_argument("--steps", type=int, default=2)

    runtime_parser = subparsers.add_parser(
        "runtime-gate",
        help="Select an exact, label-free AuraFlow extraction batching profile",
    )
    runtime_parser.add_argument("--config", required=True, type=Path)
    runtime_parser.add_argument("--output", required=True, type=Path)

    extract_parser = subparsers.add_parser("extract", help="Extract features for local images")
    extract_parser.add_argument("--config", required=True, type=Path)
    extract_parser.add_argument("--image", required=True, type=Path, action="append")
    extract_parser.add_argument("--output", required=True, type=Path)

    dataset_parser = subparsers.add_parser(
        "extract-dataset", help="Precompute sharded features for a real dataset"
    )
    dataset_parser.add_argument("--config", required=True, type=Path)
    dataset_parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "cifar10",
            "voc2012",
            "imagenet",
            "imagenet100",
            "ade20k",
            "nyuv2",
        ],
    )
    dataset_parser.add_argument("--root", required=True, type=Path)
    dataset_parser.add_argument("--split", required=True)
    dataset_parser.add_argument("--output", required=True, type=Path)
    dataset_parser.add_argument("--limit", type=int)
    dataset_parser.add_argument("--offset", type=int, default=0)
    dataset_parser.add_argument("--resume", action="store_true")
    dataset_parser.add_argument(
        "--storage-policy",
        choices=["dense", "readout_sparse"],
        default="dense",
    )
    dataset_parser.add_argument(
        "--classes-file",
        type=Path,
        help="One ImageNet synset per line; order defines contiguous labels",
    )

    train_parser = subparsers.add_parser(
        "train-cache", help="Train a lightweight readout from cached features"
    )
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--cache-dir", required=True, type=Path)
    train_parser.add_argument("--val-cache-dir", type=Path)
    train_parser.add_argument("--output-dir", type=Path)
    train_parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "segmentation", "depth", "normals"],
    )
    train_parser.add_argument(
        "--representation",
        default="full",
        choices=[
            "random_feature_local",
            "full",
            "full_local",
            "full_nograph",
            "state",
            "state_nograph",
            "response",
            "response_local",
            "response_nograph",
            "state_graph",
            "dit_hidden_local",
            "dit_hidden_attention",
            "response_shuffled",
            "full_shuffled",
            "z0",
            "zt",
            "trajectory",
            "velocity",
            "mismatch",
            "endpoint",
        ],
    )
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--num-classes", type=int)
    train_parser.add_argument("--segmentation-classes", type=int)

    evaluate_parser = subparsers.add_parser(
        "evaluate-cache", help="Evaluate a saved readout on a cached split"
    )
    evaluate_parser.add_argument("--config", required=True, type=Path)
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--cache-dir", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser.add_argument("--batch-size", type=int)

    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose a feature cache")
    diagnose_parser.add_argument("--cache", required=True, type=Path)
    diagnose_parser.add_argument("--output", type=Path)

    segmentation_parser = subparsers.add_parser(
        "diagnose-segmentation",
        help="Measure unsupervised graph alignment with segmentation labels",
    )
    segmentation_parser.add_argument("--cache-dir", required=True, type=Path)
    segmentation_parser.add_argument("--output", required=True, type=Path)

    summary_parser = subparsers.add_parser(
        "summarize-runs",
        help="Aggregate multi-seed readout reports with paired t intervals",
    )
    summary_parser.add_argument("--report", required=True, type=Path, action="append")
    summary_parser.add_argument("--metric", required=True)
    summary_parser.add_argument("--reference")
    summary_parser.add_argument("--output", required=True, type=Path)

    matrix_parser = subparsers.add_parser(
        "run-readout-matrix",
        help="Run a resumable representation-by-seed train/val/test matrix",
    )
    matrix_parser.add_argument("--config", required=True, type=Path)
    matrix_parser.add_argument("--train-cache-dir", required=True, type=Path)
    matrix_parser.add_argument("--val-cache-dir", required=True, type=Path)
    matrix_parser.add_argument("--test-cache-dir", required=True, type=Path)
    matrix_parser.add_argument("--output-dir", required=True, type=Path)
    matrix_parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "segmentation", "depth", "normals"],
    )
    matrix_parser.add_argument(
        "--representation",
        required=True,
        action="append",
    )
    matrix_parser.add_argument("--seed", required=True, type=int, action="append")
    matrix_parser.add_argument("--seed-workers", type=int, default=1)
    matrix_parser.add_argument("--readout-runtime-profile", type=Path)
    matrix_parser.add_argument("--validate-model-features", action="store_true")
    matrix_parser.add_argument("--strict-host-sync", action="store_true")
    matrix_parser.add_argument("--epochs", required=True, type=int)
    matrix_parser.add_argument("--batch-size", required=True, type=int)
    matrix_parser.add_argument("--learning-rate", type=float, default=1e-3)
    matrix_parser.add_argument("--weight-decay", type=float, default=1e-4)
    matrix_parser.add_argument("--reference")
    matrix_parser.add_argument("--num-classes", type=int)
    matrix_parser.add_argument("--segmentation-classes", type=int)

    readout_runtime_parser = subparsers.add_parser(
        "readout-runtime-gate",
        help="Select an exact, resource-safe cached-readout seed parallelism profile",
    )
    readout_runtime_parser.add_argument("--config", required=True, type=Path)
    readout_runtime_parser.add_argument("--train-cache-dir", required=True, type=Path)
    readout_runtime_parser.add_argument("--val-cache-dir", required=True, type=Path)
    readout_runtime_parser.add_argument("--test-cache-dir", required=True, type=Path)
    readout_runtime_parser.add_argument("--output", required=True, type=Path)

    gate_parser = subparsers.add_parser(
        "audit-signal-gate",
        help="Apply the pre-registered promotion rule to completed signal-gate outputs",
    )
    gate_parser.add_argument("--cache-dir", required=True, type=Path, action="append")
    gate_parser.add_argument("--voc-report", required=True, type=Path)
    gate_parser.add_argument("--cifar-matrix", required=True, type=Path)
    gate_parser.add_argument("--output", required=True, type=Path)

    budget_parser = subparsers.add_parser(
        "plan-cache-budget",
        help="Project a full cache from a measured sparse cache before extraction",
    )
    budget_parser.add_argument("--measurement-cache", required=True, type=Path)
    budget_parser.add_argument(
        "--target-split",
        required=True,
        action="append",
        help="Split count as NAME=COUNT",
    )
    budget_parser.add_argument("--filesystem-path", required=True, type=Path)
    budget_parser.add_argument(
        "--storage-policy",
        choices=["dense", "readout_sparse"],
        default="readout_sparse",
    )
    budget_parser.add_argument("--additional-required-bytes", type=int, default=0)
    budget_parser.add_argument(
        "--split-extra-bytes-per-sample",
        action="append",
        default=[],
        help="Extra target payload as SPLIT=BYTES_PER_SAMPLE",
    )
    budget_parser.add_argument("--reserve-gib", type=float, default=10.0)
    budget_parser.add_argument("--safety-factor", type=float, default=1.15)
    budget_parser.add_argument("--output", required=True, type=Path)

    split_audit_parser = subparsers.add_parser(
        "audit-dataset-splits",
        help="Verify benchmark train/val/test source IDs are unique and disjoint",
    )
    split_audit_parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "voc2012", "imagenet", "imagenet100", "ade20k", "nyuv2"],
    )
    split_audit_parser.add_argument("--root", required=True, type=Path)
    split_audit_parser.add_argument("--image-size", type=int, default=256)
    split_audit_parser.add_argument(
        "--expected-split",
        action="append",
        default=[],
        help="Expected split count as NAME=COUNT",
    )
    split_audit_parser.add_argument(
        "--classes-file",
        type=Path,
        help="One ImageNet synset per line; order defines contiguous labels",
    )
    split_audit_parser.add_argument("--output", required=True, type=Path)

    evidence_parser = subparsers.add_parser(
        "audit-full-evidence",
        help="Apply the pre-registered cross-task FieldScope conclusion gate",
    )
    for dataset in ("imagenet100", "voc2012", "ade20k", "nyuv2"):
        evidence_parser.add_argument(
            f"--{dataset}-matrix",
            required=True,
            type=Path,
        )
    evidence_parser.add_argument("--voc-unsupervised", required=True, type=Path)
    evidence_parser.add_argument("--backbone-asset", required=True, type=Path)
    evidence_parser.add_argument("--runtime-profile", required=True, type=Path)
    evidence_parser.add_argument("--readout-runtime-profile", required=True, type=Path)
    for dataset in ("imagenet100", "voc2012", "ade20k", "nyuv2"):
        evidence_parser.add_argument(
            f"--{dataset}-split-audit",
            required=True,
            type=Path,
        )
    evidence_parser.add_argument("--output", required=True, type=Path)

    asset_parser = subparsers.add_parser(
        "audit-backbone-assets",
        help="Verify the exact AuraFlow FP16 weight files used by formal runs",
    )
    asset_parser.add_argument("--config", required=True, type=Path)
    asset_parser.add_argument("--output", required=True, type=Path)

    causal_parser = subparsers.add_parser(
        "audit-causal-evidence",
        help="Apply the final causal and condition conclusion gate",
    )
    causal_parser.add_argument("--main-evidence", required=True, type=Path)
    causal_parser.add_argument("--runtime-profile", required=True, type=Path)
    for variant in (
        "random-flow",
        "spatially-shuffled-probe",
        "neutral-prompt",
        "unrelated-prompt",
    ):
        causal_parser.add_argument(f"--{variant}", required=True, type=Path)
    causal_parser.add_argument("--output", required=True, type=Path)

    imagenet_asset_parser = subparsers.add_parser(
        "audit-imagenet1k-asset",
        help="Verify the registered full ImageNet-1k server export",
    )
    imagenet_asset_parser.add_argument("--root", required=True, type=Path)
    imagenet_asset_parser.add_argument("--manifest", required=True, type=Path)
    imagenet_asset_parser.add_argument("--export-summary", required=True, type=Path)
    imagenet_asset_parser.add_argument("--output", required=True, type=Path)

    ablation_config_parser = subparsers.add_parser(
        "build-extension-ablation-configs",
        help="Materialize the registered 15 one-factor high-cost configs",
    )
    ablation_config_parser.add_argument("--base-config", required=True, type=Path)
    ablation_config_parser.add_argument("--output-dir", required=True, type=Path)
    ablation_config_parser.add_argument("--output", required=True, type=Path)

    extension_evidence_parser = subparsers.add_parser(
        "audit-extension-evidence",
        help="Audit conditional ImageNet-1k and high-cost ablation evidence",
    )
    extension_evidence_parser.add_argument("--main-evidence", required=True, type=Path)
    extension_evidence_parser.add_argument("--imagenet-matrix", required=True, type=Path)
    extension_evidence_parser.add_argument(
        "--imagenet-asset-audit", required=True, type=Path
    )
    extension_evidence_parser.add_argument(
        "--imagenet-split-audit", required=True, type=Path
    )
    extension_evidence_parser.add_argument(
        "--main-runtime-profile", required=True, type=Path
    )
    extension_evidence_parser.add_argument(
        "--readout-runtime-profile", required=True, type=Path
    )
    extension_evidence_parser.add_argument("--base-voc-report", required=True, type=Path)
    extension_evidence_parser.add_argument("--base-config", required=True, type=Path)
    extension_evidence_parser.add_argument(
        "--ablation-registry", required=True, type=Path
    )
    extension_evidence_parser.add_argument(
        "--ablation-report",
        required=True,
        action="append",
        help="Registered ablation report as NAME=PATH",
    )
    extension_evidence_parser.add_argument(
        "--ablation-runtime-profile",
        required=True,
        action="append",
        help="Registered ablation runtime profile as NAME=PATH",
    )
    extension_evidence_parser.add_argument("--output", required=True, type=Path)

    final_evidence_parser = subparsers.add_parser(
        "audit-final-evidence",
        help="Combine main, causal, and conditionally required extension evidence",
    )
    final_evidence_parser.add_argument("--main-evidence", required=True, type=Path)
    final_evidence_parser.add_argument("--causal-evidence", required=True, type=Path)
    final_evidence_parser.add_argument("--extension-evidence", type=Path)
    final_evidence_parser.add_argument("--output", required=True, type=Path)

    inspect_parser = subparsers.add_parser("inspect-cache", help="Print cache manifest")
    inspect_parser.add_argument("--cache", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code = 0
    if args.command == "doctor":
        report = doctor(load_config(args.config))
        if args.output:
            _json_dump(args.output, report)
    elif args.command == "smoke":
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        config = load_config(args.config)
        report = run_smoke(config, args.steps)
        _json_dump(Path(config.runtime.output_dir) / "smoke_report.json", report)
    elif args.command == "runtime-gate":
        if os.environ.get("FIELDSCOPE_RUNTIME_PROFILE"):
            raise ValueError("Unset FIELDSCOPE_RUNTIME_PROFILE while running the gate")
        config = load_config(args.config)
        if args.output.is_file():
            report = load_runtime_gate_report(config, args.output)
        else:
            report = run_runtime_gate(config)
            _json_dump(args.output, report)
    elif args.command == "extract":
        report = extract_images(load_config(args.config), list(args.image), args.output)
    elif args.command == "extract-dataset":
        class_names = None
        if args.classes_file:
            class_names = [
                line.strip()
                for line in args.classes_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        report = extract_dataset(
            load_config(args.config),
            dataset_name=args.dataset,
            dataset_root=args.root,
            split=args.split,
            output_dir=args.output,
            limit=args.limit,
            offset=args.offset,
            resume=args.resume,
            class_names=class_names,
            storage_policy=args.storage_policy,
        )
    elif args.command == "train-cache":
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config = load_config(args.config)
        if args.num_classes is not None or args.segmentation_classes is not None:
            config = replace(
                config,
                tokenizer=replace(
                    config.tokenizer,
                    num_classes=(
                        args.num_classes
                        if args.num_classes is not None
                        else config.tokenizer.num_classes
                    ),
                    segmentation_classes=(
                        args.segmentation_classes
                        if args.segmentation_classes is not None
                        else config.tokenizer.segmentation_classes
                    ),
                ),
            )
        report = train_cache(
            config,
            cache_dir=args.cache_dir,
            task=args.task,
            representation=args.representation,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            val_cache_dir=args.val_cache_dir,
            output_dir=args.output_dir,
            weight_decay=args.weight_decay,
            seed=args.seed,
            resume_checkpoint=args.resume,
            batch_size=args.batch_size,
        )
    elif args.command == "evaluate-cache":
        report = evaluate_checkpoint(
            load_config(args.config),
            checkpoint=args.checkpoint,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )
        _json_dump(args.output, report)
    elif args.command == "diagnose":
        features, _, manifest = load_features(args.cache)
        report = {"manifest": manifest, "diagnostics": graph_diagnostics(features)}
        if args.output:
            _json_dump(args.output, report)
    elif args.command == "diagnose-segmentation":
        report = diagnose_segmentation_cache(args.cache_dir)
        _json_dump(args.output, report)
    elif args.command == "summarize-runs":
        report = summarize_run_reports(
            list(args.report),
            metric=args.metric,
            reference=args.reference,
        )
        _json_dump(args.output, report)
    elif args.command == "run-readout-matrix":
        config = load_config(args.config)
        if args.num_classes is not None or args.segmentation_classes is not None:
            config = replace(
                config,
                tokenizer=replace(
                    config.tokenizer,
                    num_classes=args.num_classes or config.tokenizer.num_classes,
                    segmentation_classes=(
                        args.segmentation_classes or config.tokenizer.segmentation_classes
                    ),
                ),
            )
        seed_workers = args.seed_workers
        readout_runtime_profile = None
        if args.readout_runtime_profile is not None:
            if seed_workers != 1:
                raise ValueError("Do not combine --seed-workers with --readout-runtime-profile")
            load_readout_runtime_gate_report(config, args.readout_runtime_profile)
            readout_runtime_profile = readout_runtime_profile_identity(args.readout_runtime_profile)
            seed_workers = int(readout_runtime_profile["selected_profile"]["seed_workers"])
        report = run_readout_matrix_seed_parallel(
            config,
            train_cache_dir=args.train_cache_dir,
            val_cache_dir=args.val_cache_dir,
            test_cache_dir=args.test_cache_dir,
            output_dir=args.output_dir,
            task=args.task,
            representations=list(args.representation),
            seeds=list(args.seed),
            seed_workers=seed_workers,
            readout_runtime_profile=readout_runtime_profile,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            reference=args.reference,
            validate_model_features=args.validate_model_features,
            strict_host_sync=args.strict_host_sync,
        )
    elif args.command == "readout-runtime-gate":
        config = load_config(args.config)
        if args.output.is_file():
            report = load_readout_runtime_gate_report(config, args.output)
        else:
            report = run_readout_runtime_gate(
                config,
                config_path=args.config,
                train_cache_dir=args.train_cache_dir,
                val_cache_dir=args.val_cache_dir,
                test_cache_dir=args.test_cache_dir,
                output_path=args.output,
            )
            _json_dump(args.output, report)
    elif args.command == "audit-signal-gate":
        report = audit_signal_gate(
            cache_dirs=list(args.cache_dir),
            voc_report_path=args.voc_report,
            cifar_matrix_path=args.cifar_matrix,
        )
        _json_dump(args.output, report)
    elif args.command == "plan-cache-budget":
        target_samples: dict[str, int] = {}
        for specification in args.target_split:
            name, separator, count = specification.partition("=")
            if not separator or not name or not count:
                raise ValueError("--target-split must use NAME=COUNT")
            if name in target_samples:
                raise ValueError(f"Duplicate target split: {name}")
            target_samples[name] = int(count)
        additional_bytes_per_sample: dict[str, int] = {}
        for specification in args.split_extra_bytes_per_sample:
            name, separator, value = specification.partition("=")
            if not separator or not name or not value:
                raise ValueError("--split-extra-bytes-per-sample must use SPLIT=BYTES_PER_SAMPLE")
            if name in additional_bytes_per_sample:
                raise ValueError(f"Duplicate split extra bytes: {name}")
            additional_bytes_per_sample[name] = int(value)
        report = plan_cache_budget(
            measurement_cache=args.measurement_cache,
            target_samples=target_samples,
            filesystem_path=args.filesystem_path,
            reserve_bytes=int(args.reserve_gib * 1024**3),
            safety_factor=args.safety_factor,
            required_storage_policy=args.storage_policy,
            additional_required_bytes=args.additional_required_bytes,
            additional_bytes_per_sample=additional_bytes_per_sample,
        )
        _json_dump(args.output, report)
    elif args.command == "audit-dataset-splits":
        class_names = None
        if args.classes_file:
            class_names = [
                line.strip()
                for line in args.classes_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        expected_counts: dict[str, int] = {}
        for specification in args.expected_split:
            name, separator, count = specification.partition("=")
            if not separator or not name or not count:
                raise ValueError("--expected-split must use NAME=COUNT")
            if name in expected_counts:
                raise ValueError(f"Duplicate expected split: {name}")
            expected_counts[name] = int(count)
        report = audit_dataset_splits(
            dataset_name=args.dataset,
            dataset_root=args.root,
            image_size=args.image_size,
            class_names=class_names,
            expected_counts=expected_counts,
        )
        _json_dump(args.output, report)
        if report["status"] != "passed":
            exit_code = 2
    elif args.command == "audit-full-evidence":
        report = audit_full_evidence(
            matrix_paths={
                "imagenet100": args.imagenet100_matrix,
                "voc2012": args.voc2012_matrix,
                "ade20k": args.ade20k_matrix,
                "nyuv2": args.nyuv2_matrix,
            },
            voc_unsupervised_path=args.voc_unsupervised,
            backbone_asset_path=args.backbone_asset,
            runtime_profile_path=args.runtime_profile,
            readout_runtime_profile_path=args.readout_runtime_profile,
            split_audit_paths={
                "imagenet100": args.imagenet100_split_audit,
                "voc2012": args.voc2012_split_audit,
                "ade20k": args.ade20k_split_audit,
                "nyuv2": args.nyuv2_split_audit,
            },
        )
        _json_dump(args.output, report)
        if report["verdict"] == "incomplete":
            exit_code = 2
    elif args.command == "audit-backbone-assets":
        report = audit_backbone_assets(load_config(args.config))
        _json_dump(args.output, report)
        if report["status"] != "passed":
            exit_code = 2
    elif args.command == "audit-causal-evidence":
        report = audit_causal_evidence(
            main_evidence_path=args.main_evidence,
            runtime_profile_path=args.runtime_profile,
            causal_report_paths={
                "random_flow": args.random_flow,
                "spatially_shuffled_probe": args.spatially_shuffled_probe,
                "neutral_prompt": args.neutral_prompt,
                "unrelated_prompt": args.unrelated_prompt,
            },
        )
        _json_dump(args.output, report)
        if report["verdict"] == "incomplete":
            exit_code = 2
    elif args.command == "audit-imagenet1k-asset":
        report = audit_imagenet1k_asset(
            dataset_root=args.root,
            manifest_path=args.manifest,
            export_summary_path=args.export_summary,
        )
        _json_dump(args.output, report)
        if report["status"] != "passed":
            exit_code = 2
    elif args.command == "build-extension-ablation-configs":
        report = build_high_cost_ablation_configs(args.base_config, args.output_dir)
        _json_dump(args.output, report)
    elif args.command == "audit-extension-evidence":
        ablation_reports: dict[str, Path] = {}
        for specification in args.ablation_report:
            name, separator, path = specification.partition("=")
            if not separator or not name or not path or name in ablation_reports:
                raise ValueError("--ablation-report must contain unique NAME=PATH values")
            ablation_reports[name] = Path(path)
        ablation_runtime_profiles: dict[str, Path] = {}
        for specification in args.ablation_runtime_profile:
            name, separator, path = specification.partition("=")
            if (
                not separator
                or not name
                or not path
                or name in ablation_runtime_profiles
            ):
                raise ValueError(
                    "--ablation-runtime-profile must contain unique NAME=PATH values"
                )
            ablation_runtime_profiles[name] = Path(path)
        report = audit_extension_evidence(
            main_evidence_path=args.main_evidence,
            imagenet_matrix_path=args.imagenet_matrix,
            imagenet_asset_audit_path=args.imagenet_asset_audit,
            imagenet_split_audit_path=args.imagenet_split_audit,
            main_runtime_profile_path=args.main_runtime_profile,
            readout_runtime_profile_path=args.readout_runtime_profile,
            base_voc_report_path=args.base_voc_report,
            base_config_path=args.base_config,
            ablation_registry_path=args.ablation_registry,
            ablation_report_paths=ablation_reports,
            ablation_runtime_profile_paths=ablation_runtime_profiles,
        )
        _json_dump(args.output, report)
        if report["verdict"] == "incomplete":
            exit_code = 2
    elif args.command == "audit-final-evidence":
        report = audit_final_evidence(
            main_evidence_path=args.main_evidence,
            causal_evidence_path=args.causal_evidence,
            extension_evidence_path=args.extension_evidence,
        )
        _json_dump(args.output, report)
        if report["verdict"] == "incomplete":
            exit_code = 2
    elif args.command == "inspect-cache":
        _, _, report = load_features(args.cache)
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
