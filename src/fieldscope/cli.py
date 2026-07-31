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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fieldscope.backends import build_backend
from fieldscope.cache import load_features, save_features
from fieldscope.config import RunConfig, load_config
from fieldscope.datasets import SyntheticShapesDataset
from fieldscope.diagnostics import graph_diagnostics
from fieldscope.experiments import (
    atomic_json_dump,
    diagnose_segmentation_cache,
    evaluate_checkpoint,
    extract_dataset_cache,
    set_experiment_seed,
    train_cached_readout,
)
from fieldscope.losses import multitask_loss
from fieldscope.model import FieldScopeModel
from fieldscope.response import FieldResponseExtractor


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
            "baselines": {
                name: list(value.shape) for name, value in features.baselines.items()
            },
            "graphs": {
                name: list(value.shape) for name, value in features.graphs.items()
            },
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


def extract_images(
    config: RunConfig, image_paths: list[Path], output: Path
) -> dict[str, Any]:
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
        choices=["cifar10", "voc2012", "imagenet", "imagenet100", "nyuv2"],
    )
    dataset_parser.add_argument("--root", required=True, type=Path)
    dataset_parser.add_argument("--split", required=True)
    dataset_parser.add_argument("--output", required=True, type=Path)
    dataset_parser.add_argument("--limit", type=int)
    dataset_parser.add_argument("--offset", type=int, default=0)
    dataset_parser.add_argument("--resume", action="store_true")
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
            "full",
            "state",
            "response",
            "response_local",
            "state_graph",
            "dit_hidden_local",
            "dit_hidden_attention",
            "z0",
            "zt",
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

    evaluate_parser = subparsers.add_parser(
        "evaluate-cache", help="Evaluate a saved readout on a cached split"
    )
    evaluate_parser.add_argument("--config", required=True, type=Path)
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--cache-dir", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)

    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose a feature cache")
    diagnose_parser.add_argument("--cache", required=True, type=Path)
    diagnose_parser.add_argument("--output", type=Path)

    segmentation_parser = subparsers.add_parser(
        "diagnose-segmentation",
        help="Measure unsupervised graph alignment with segmentation labels",
    )
    segmentation_parser.add_argument("--cache-dir", required=True, type=Path)
    segmentation_parser.add_argument("--output", required=True, type=Path)

    inspect_parser = subparsers.add_parser("inspect-cache", help="Print cache manifest")
    inspect_parser.add_argument("--cache", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
    elif args.command == "extract":
        report = extract_images(
            load_config(args.config), list(args.image), args.output
        )
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
        )
    elif args.command == "train-cache":
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        report = train_cache(
            load_config(args.config),
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
        )
    elif args.command == "evaluate-cache":
        report = evaluate_checkpoint(
            load_config(args.config),
            checkpoint=args.checkpoint,
            cache_dir=args.cache_dir,
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
    elif args.command == "inspect-cache":
        _, _, report = load_features(args.cache)
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
