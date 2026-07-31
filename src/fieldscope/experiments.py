"""Resumable feature extraction and reproducible cached-readout experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fieldscope.backends import build_backend
from fieldscope.cache import load_features, save_features
from fieldscope.cached_dataset import CachedFeatureDataset, collate_cached
from fieldscope.config import RunConfig
from fieldscope.datasets import build_vision_dataset
from fieldscope.evaluation import (
    ClassificationMeter,
    DepthMeter,
    SegmentationMeter,
    graph_segmentation_metrics,
)
from fieldscope.feature_ops import select_representation
from fieldscope.graph import cosine_affinity
from fieldscope.losses import multitask_loss
from fieldscope.model import FieldScopeModel
from fieldscope.response import FieldResponseExtractor


def atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def set_experiment_seed(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def extraction_signature(
    config: RunConfig,
    *,
    dataset_name: str,
    dataset_root: Path,
    split: str,
    class_names: list[str] | None,
    offset: int,
    stop: int,
) -> str:
    payload = {
        "config": config.to_dict(),
        "dataset": dataset_name,
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "class_names": class_names,
        "offset": offset,
        "stop": stop,
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_tensors(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in batch.items()
        if name in {"classification", "segmentation", "depth", "normals"}
        and isinstance(value, torch.Tensor)
    }


def extract_dataset_cache(
    config: RunConfig,
    *,
    dataset_name: str,
    dataset_root: Path,
    split: str,
    output_dir: Path,
    limit: int | None = None,
    offset: int = 0,
    resume: bool = False,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Extract deterministic shards and checkpoint the manifest after every shard."""

    total_started = time.perf_counter()
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("offset must be non-negative and limit must be positive")
    dataset = build_vision_dataset(
        dataset_name,
        dataset_root,
        split,
        config.backend.image_size,
        class_names=class_names,
    )
    stop = len(dataset) if limit is None else min(len(dataset), offset + limit)
    if offset >= stop:
        raise ValueError("Requested extraction range is empty")
    selected_indices = list(range(offset, stop))
    dataset = Subset(dataset, selected_indices)
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    signature = extraction_signature(
        config,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        split=split,
        class_names=class_names,
        offset=offset,
        stop=stop,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"{manifest_path} exists; pass resume=True to continue"
            )
        if existing.get("extraction_signature") != signature:
            raise ValueError("Existing cache was produced by a different extraction")

    backend_started = time.perf_counter()
    backend = build_backend(config.backend)
    backend_seconds = time.perf_counter() - backend_started
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    extractor = FieldResponseExtractor(backend, config.probe)
    shards: list[dict[str, Any]] = []
    sample_count = 0
    written_samples = 0
    extraction_seconds = 0.0
    cache_bytes = 0
    for batch_index, batch in enumerate(loader):
        shard_started = time.perf_counter()
        images = batch["image"]
        if not isinstance(images, torch.Tensor):
            raise TypeError("Dataset image batch must be a tensor")
        sample_ids = [str(value) for value in batch["sample_id"]]
        global_start = offset + batch_index * config.runtime.batch_size
        global_end = global_start + images.shape[0]
        path = output_dir / f"shard-{global_start:08d}-{global_end:08d}.pt"
        if resume and path.is_file():
            _, cached_targets, cached_manifest = load_features(path)
            if cached_manifest.get("sample_ids") != sample_ids:
                raise ValueError(f"Sample IDs do not match resumed shard {path.name}")
            batch_targets = _target_tensors(batch)
            if set(cached_targets) != set(batch_targets):
                raise ValueError(f"Targets do not match resumed shard {path.name}")
            if any(
                not torch.equal(cached_targets[name], batch_targets[name])
                for name in cached_targets
            ):
                raise ValueError(f"Target values do not match resumed shard {path.name}")
            feature_manifest = cached_manifest
            status = "reused"
            shard_seconds = time.perf_counter() - shard_started
        else:
            features = extractor.extract(images)
            feature_manifest = save_features(
                path,
                features,
                targets=_target_tensors(batch),
                sample_ids=sample_ids,
            )
            status = "written"
            shard_seconds = time.perf_counter() - shard_started
            extraction_seconds += shard_seconds
            written_samples += images.shape[0]
        shard_bytes = path.stat().st_size
        cache_bytes += shard_bytes
        shard = {
            "path": path.name,
            "start": global_start,
            "end": global_end,
            "num_samples": images.shape[0],
            "fingerprint": feature_manifest["fingerprint"],
            "status": status,
            "seconds": shard_seconds,
            "bytes": shard_bytes,
        }
        shards.append(shard)
        sample_count += images.shape[0]
        report = {
            "format_version": 2,
            "dataset": dataset_name,
            "split": split,
            "source_root": str(dataset_root.resolve()),
            "offset": offset,
            "requested_stop": stop,
            "num_samples": sample_count,
            "complete": sample_count == len(selected_indices),
            "extraction_signature": signature,
            "backend": backend.describe(),
            "config": config.to_dict(),
            "class_names": class_names,
            "shards": shards,
            "runtime": {
                "backend_initialization_seconds": backend_seconds,
                "elapsed_seconds": time.perf_counter() - total_started,
                "feature_extraction_seconds": extraction_seconds,
                "written_samples": written_samples,
                "written_samples_per_second": (
                    written_samples / extraction_seconds
                    if extraction_seconds > 0
                    else None
                ),
                "cache_bytes": cache_bytes,
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
                **getattr(backend, "runtime_stats", lambda: {})(),
            },
        }
        atomic_json_dump(manifest_path, report)
    return report


def _task_output_size(
    task: str,
    targets: Mapping[str, torch.Tensor],
    grid_size: tuple[int, int],
) -> tuple[int, int]:
    if task in {"segmentation", "depth", "normals"}:
        return tuple(targets[task].shape[-2:])
    return grid_size


def _make_meter(task: str, config: RunConfig) -> Any:
    if task == "classification":
        return ClassificationMeter()
    if task == "segmentation":
        return SegmentationMeter(config.tokenizer.segmentation_classes)
    if task == "depth":
        return DepthMeter()
    if task == "normals":
        return []
    raise ValueError(f"Unsupported task: {task}")


def _update_meter(
    meter: Any,
    task: str,
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> None:
    if task in {"classification", "segmentation", "depth"}:
        meter.update(predictions[task], targets[task])
    elif task == "normals":
        predicted = torch.nn.functional.normalize(predictions[task], dim=1)
        expected = torch.nn.functional.normalize(targets[task], dim=1)
        cosine = (predicted * expected).sum(dim=1).clamp(-1, 1)
        meter.extend(torch.rad2deg(torch.acos(cosine)).detach().cpu().flatten().tolist())


def _compute_meter(meter: Any, task: str) -> dict[str, float]:
    if task in {"classification", "segmentation", "depth"}:
        return meter.compute()
    values = torch.tensor(meter, dtype=torch.float32)
    return {
        "mean_angular_error": float(values.mean().item()),
        "median_angular_error": float(values.median().item()),
    }


@torch.no_grad()
def evaluate_cached_readout(
    model: FieldScopeModel,
    config: RunConfig,
    *,
    cache_dir: Path,
    task: str,
    representation: str,
    device: torch.device,
) -> dict[str, Any]:
    dataset = CachedFeatureDataset(cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cached,
    )
    meter = _make_meter(task, config)
    total_loss = 0.0
    total_samples = 0
    model.eval()
    for cached_batch in loader:
        features, _ = select_representation(
            cached_batch["features"].to(device, dtype=torch.float32),
            representation,
        )
        targets = {
            name: value.to(device) for name, value in cached_batch["targets"].items()
        }
        if task not in targets:
            raise ValueError(f"Requested task {task!r} is absent from cache")
        predictions = model(
            features,
            output_size=_task_output_size(task, targets, features.grid_size),
        )
        loss, _ = multitask_loss(predictions, {task: targets[task]})
        batch_size = features.state.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        _update_meter(meter, task, predictions, targets)
    return {
        "loss": total_loss / max(1, total_samples),
        "num_samples": total_samples,
        "metrics": _compute_meter(meter, task),
    }


def _primary_metric(task: str, report: Mapping[str, Any]) -> tuple[float, bool]:
    metrics = report["metrics"]
    if task == "classification":
        return float(metrics["top1"]), True
    if task == "segmentation":
        return float(metrics["mean_iou"]), True
    if task == "depth":
        return float(metrics["abs_rel"]), False
    return float(metrics["mean_angular_error"]), False


def train_cached_readout(
    config: RunConfig,
    *,
    train_cache_dir: Path,
    val_cache_dir: Path,
    output_dir: Path,
    task: str,
    representation: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_experiment_seed(seed, config.runtime.deterministic)
    train_dataset = CachedFeatureDataset(train_cache_dir)
    val_dataset = CachedFeatureDataset(val_cache_dir)
    first = train_dataset[0]
    selected, mode = select_representation(first["features"], representation)
    device = torch.device(config.backend.device)
    model = FieldScopeModel(
        selected.state.shape[-1],
        selected.response.shape[-1],
        config.tokenizer,
        mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = 0
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if payload.get("task") != task:
            raise ValueError("Resume checkpoint task does not match requested task")
        if payload.get("representation") != representation:
            raise ValueError(
                "Resume checkpoint representation does not match requested representation"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        history = list(payload.get("history", []))
        best_value = payload.get("best_primary_metric")
        best_epoch = int(payload.get("best_epoch", 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{task}_{representation}_seed{seed}_best.pt"
    last_path = output_dir / f"{task}_{representation}_seed{seed}_last.pt"
    report: dict[str, Any] | None = None
    for epoch in range(start_epoch, epochs):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.runtime.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=0,
            collate_fn=collate_cached,
        )
        model.train()
        total_loss = 0.0
        total_samples = 0
        for cached_batch in train_loader:
            features, _ = select_representation(
                cached_batch["features"].to(device, dtype=torch.float32),
                representation,
            )
            targets = {
                name: tensor.to(device)
                for name, tensor in cached_batch["targets"].items()
            }
            if task not in targets:
                raise ValueError(f"Requested task {task!r} is absent from cache")
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                features,
                output_size=_task_output_size(task, targets, features.grid_size),
            )
            loss, _ = multitask_loss(predictions, {task: targets[task]})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = features.state.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
        scheduler.step()
        validation = evaluate_cached_readout(
            model,
            config,
            cache_dir=val_cache_dir,
            task=task,
            representation=representation,
            device=device,
        )
        entry = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(1, total_samples),
            "validation": validation,
        }
        history.append(entry)
        value, maximize = _primary_metric(task, validation)
        is_best = best_value is None or (
            value > best_value if maximize else value < best_value
        )
        if is_best:
            best_value = value
            best_epoch = epoch + 1
        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "config": config.to_dict(),
            "task": task,
            "representation": representation,
            "mode": mode,
            "state_dim": selected.state.shape[-1],
            "response_dim": selected.response.shape[-1],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_primary_metric": best_value,
        }
        atomic_torch_save(last_path, checkpoint_payload)
        if is_best:
            atomic_torch_save(best_path, checkpoint_payload)
        report = {
            "status": "running" if epoch + 1 < epochs else "passed",
            "task": task,
            "representation": representation,
            "seed": seed,
            "epochs": epochs,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "best_epoch": best_epoch,
            "best_primary_metric": best_value,
            "checkpoint": str(best_path),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history": history,
        }
        atomic_json_dump(
            output_dir / f"{task}_{representation}_seed{seed}_report.json",
            report,
        )
    if report is None:
        validation = evaluate_cached_readout(
            model,
            config,
            cache_dir=val_cache_dir,
            task=task,
            representation=representation,
            device=device,
        )
        report = {
            "status": "passed",
            "task": task,
            "representation": representation,
            "seed": seed,
            "epochs": epochs,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "best_epoch": best_epoch,
            "best_primary_metric": best_value,
            "checkpoint": str(best_path),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(resume_checkpoint),
            "history": history,
            "validation": validation,
        }
    return report


def evaluate_checkpoint(
    config: RunConfig,
    *,
    checkpoint: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    device = torch.device(config.backend.device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FieldScopeModel(
        payload["state_dim"],
        payload["response_dim"],
        config.tokenizer,
        mode=payload["mode"],
    ).to(device)
    model.load_state_dict(payload["model"])
    evaluation = evaluate_cached_readout(
        model,
        config,
        cache_dir=cache_dir,
        task=payload["task"],
        representation=payload["representation"],
        device=device,
    )
    return {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "task": payload["task"],
        "representation": payload["representation"],
        "seed": payload["seed"],
        "evaluation": evaluation,
    }


def diagnose_segmentation_cache(cache_dir: Path) -> dict[str, Any]:
    dataset = CachedFeatureDataset(cache_dir)
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        features = sample["features"]
        targets = sample["targets"]
        if "segmentation" not in targets:
            raise ValueError("Segmentation targets are required for graph diagnosis")
        representations = {
            "response": features.affinity.float(),
            "state": cosine_affinity(features.state.float()),
            **{
                name: graph.float()
                for name, graph in features.graphs.items()
                if not name.endswith("_adjacency")
            },
            **{
                name: cosine_affinity(value.float())
                for name, value in features.baselines.items()
            },
        }
        target = targets["segmentation"].unsqueeze(0)
        for name, affinity in representations.items():
            metrics = graph_segmentation_metrics(
                affinity,
                target,
                features.grid_size,
            )
            accumulator = totals.setdefault(
                name, {metric_name: 0.0 for metric_name in metrics}
            )
            finite = all(np.isfinite(value) for value in metrics.values())
            if not finite:
                continue
            for metric_name, value in metrics.items():
                accumulator[metric_name] += value
            counts[name] = counts.get(name, 0) + 1
    return {
        "status": "passed",
        "cache_dir": str(cache_dir),
        "num_samples": len(dataset),
        "representations": {
            name: {
                metric_name: value / max(1, counts.get(name, 0))
                for metric_name, value in metrics.items()
            }
            for name, metrics in totals.items()
        },
        "valid_samples": counts,
    }
