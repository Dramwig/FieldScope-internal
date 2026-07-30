"""Command-line entry points for validation, extraction, and diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from fieldscope.backends import build_backend
from fieldscope.cache import load_features, save_features
from fieldscope.cached_dataset import CachedFeatureDataset, collate_cached
from fieldscope.config import RunConfig, load_config
from fieldscope.datasets import SyntheticShapesDataset, build_torchvision_dataset
from fieldscope.diagnostics import graph_diagnostics
from fieldscope.feature_ops import select_representation
from fieldscope.losses import multitask_loss
from fieldscope.model import FieldScopeModel
from fieldscope.response import FieldResponseExtractor


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _set_seed(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


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
    _set_seed(config.probe.seed, config.runtime.deterministic)
    backend = build_backend(config.backend)
    extractor = FieldResponseExtractor(backend, config.probe)
    batch = _synthetic_batch(config)
    images = batch["image"]
    assert isinstance(images, torch.Tensor)
    features = extractor.extract(images)
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

    cache_path = Path(config.runtime.output_dir) / "smoke_features.pt"
    manifest = save_features(
        cache_path,
        features,
        targets={name: value.cpu() for name, value in targets.items()},
        sample_ids=list(batch["sample_id"]),
    )
    reloaded, _, _ = load_features(cache_path)
    return {
        "status": "passed",
        "backend": backend.describe(),
        "feature_shapes": {
            "state": list(features.state.shape),
            "response": list(features.response.shape),
            "affinity": list(features.affinity.shape),
        },
        "diagnostics": graph_diagnostics(features),
        "train_steps": steps,
        "losses": losses,
        "loss_components": components,
        "cache": str(cache_path),
        "cache_fingerprint": manifest["fingerprint"],
        "cache_reload_equal": bool(torch.equal(features.state.cpu(), reloaded.state)),
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
) -> dict[str, Any]:
    dataset = build_torchvision_dataset(
        dataset_name,
        dataset_root,
        split,
        config.backend.image_size,
        download=False,
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
    )
    backend = build_backend(config.backend)
    extractor = FieldResponseExtractor(backend, config.probe)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    sample_offset = 0
    for shard_index, (images, target) in enumerate(loader):
        features = extractor.extract(images)
        target_name = "classification" if dataset_name == "cifar10" else "segmentation"
        path = output_dir / f"shard-{shard_index:06d}.pt"
        sample_ids = [
            f"{dataset_name}-{split}-{index:08d}"
            for index in range(sample_offset, sample_offset + images.shape[0])
        ]
        manifest = save_features(
            path,
            features,
            targets={target_name: target},
            sample_ids=sample_ids,
        )
        shards.append(
            {
                "path": path.name,
                "num_samples": images.shape[0],
                "fingerprint": manifest["fingerprint"],
            }
        )
        sample_offset += images.shape[0]
    report = {
        "format_version": 1,
        "dataset": dataset_name,
        "split": split,
        "source_root": str(dataset_root.resolve()),
        "num_samples": sample_offset,
        "backend": backend.describe(),
        "config": config.to_dict(),
        "shards": shards,
    }
    _json_dump(output_dir / "dataset_manifest.json", report)
    return report


def train_cache(
    config: RunConfig,
    *,
    cache_dir: Path,
    task: str,
    representation: str,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    _set_seed(config.probe.seed, config.runtime.deterministic)
    dataset = CachedFeatureDataset(cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cached,
    )
    first = dataset[0]
    selected, mode = select_representation(first["features"], representation)
    device = torch.device(config.backend.device)
    model = FieldScopeModel(
        selected.state.shape[-1],
        selected.response.shape[-1],
        config.tokenizer,
        mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for cached_batch in loader:
            features, _ = select_representation(
                cached_batch["features"].to(device, dtype=torch.float32), representation
            )
            targets = {
                name: tensor.to(device) for name, tensor in cached_batch["targets"].items()
            }
            if task not in targets:
                raise ValueError(f"Requested task {task!r} is absent from cache")
            if task == "segmentation":
                output_size = tuple(targets[task].shape[-2:])
            elif task in {"depth", "normals"}:
                output_size = tuple(targets[task].shape[-2:])
            else:
                output_size = features.grid_size
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features, output_size=output_size)
            loss, _ = multitask_loss(
                predictions,
                {task: targets[task]},
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, batches)})
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{task}_{representation}_readout.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "task": task,
            "representation": representation,
            "history": history,
        },
        checkpoint,
    )
    report = {
        "status": "passed",
        "task": task,
        "representation": representation,
        "epochs": epochs,
        "num_samples": len(dataset),
        "history": history,
        "checkpoint": str(checkpoint),
    }
    _json_dump(output_dir / f"{task}_{representation}_train_report.json", report)
    return report


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
    dataset_parser.add_argument("--dataset", required=True, choices=["cifar10", "voc2012"])
    dataset_parser.add_argument("--root", required=True, type=Path)
    dataset_parser.add_argument("--split", required=True)
    dataset_parser.add_argument("--output", required=True, type=Path)
    dataset_parser.add_argument("--limit", type=int)

    train_parser = subparsers.add_parser(
        "train-cache", help="Train a lightweight readout from cached features"
    )
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--cache-dir", required=True, type=Path)
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
            "z0",
            "zt",
            "velocity",
            "mismatch",
            "endpoint",
        ],
    )
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)

    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose a feature cache")
    diagnose_parser.add_argument("--cache", required=True, type=Path)
    diagnose_parser.add_argument("--output", type=Path)

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
        report = extract_dataset(
            load_config(args.config),
            dataset_name=args.dataset,
            dataset_root=args.root,
            split=args.split,
            output_dir=args.output,
            limit=args.limit,
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
        )
    elif args.command == "diagnose":
        features, _, manifest = load_features(args.cache)
        report = {"manifest": manifest, "diagnostics": graph_diagnostics(features)}
        if args.output:
            _json_dump(args.output, report)
    elif args.command == "inspect-cache":
        _, _, report = load_features(args.cache)
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
