"""Resumable feature extraction and reproducible cached-readout experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fieldscope.backends import build_backend
from fieldscope.cache import load_features, save_features
from fieldscope.cached_dataset import (
    CachedFeatureDataset,
    RandomFeatureCachedDataset,
    ShardShuffleSampler,
    ShuffledResponseCachedDataset,
    cached_control_contract,
    cached_control_contract_for_cache,
    collate_cached,
    shared_memory_cache_stats,
)
from fieldscope.config import RunConfig, runtime_profile_identity
from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.datasets import build_vision_dataset
from fieldscope.evaluation import (
    ClassificationMeter,
    DepthMeter,
    SegmentationMeter,
    graph_segmentation_metrics,
)
from fieldscope.feature_ops import select_representation, stack_features
from fieldscope.graph import cosine_affinity
from fieldscope.losses import multitask_loss
from fieldscope.model import FieldScopeModel
from fieldscope.response import FieldResponseExtractor
from fieldscope.statistics import bootstrap_mean_interval, summarize_run_reports

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def code_provenance() -> dict[str, Any]:
    digest = hashlib.sha256()
    source_root = _REPOSITORY_ROOT / "src" / "fieldscope"
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(_REPOSITORY_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        status = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        dirty: bool | None = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {
        "code_revision": git_revision(),
        "code_dirty": dirty,
        "code_tree_sha256": digest.hexdigest(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(cache_dir: Path) -> dict[str, Any]:
    """Return the immutable manifest identity consumed by a readout run."""

    resolved = cache_dir.resolve()
    manifest_path = resolved / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"Cache manifest is incomplete: {manifest_path}")
    return {
        "path": str(resolved),
        "manifest_sha256": file_sha256(manifest_path),
        "extraction_signature": manifest.get("extraction_signature"),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "num_samples": manifest.get("num_samples"),
        "sample_ids_sha256": manifest.get("sample_ids_sha256"),
        "storage_policy": manifest.get("storage_policy", "dense"),
        "code_revision": manifest.get("code_revision"),
        "code_tree_sha256": manifest.get("code_tree_sha256"),
    }


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _require_report_contract(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    report_path: Path,
    kind: str,
) -> None:
    for name, value in expected.items():
        if payload.get(name) != value:
            if name == "code_revision":
                raise ValueError(f"Stale {kind} report revision: {report_path}")
            if name == "code_tree_sha256":
                raise ValueError(f"Stale {kind} report code tree: {report_path}")
            raise ValueError(
                f"Incompatible {kind} report {name}: {report_path}"
            )


def _verified_cached_shard(output_dir: Path, shard: Mapping[str, Any]) -> bool:
    path = output_dir / str(shard["path"])
    if not path.is_file() or path.stat().st_size != int(shard.get("bytes", -1)):
        return False
    expected_sha256 = shard.get("sha256")
    if expected_sha256 is not None:
        return file_sha256(path) == expected_sha256
    try:
        _, _, manifest = load_features(path)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        manifest.get("fingerprint") == shard.get("fingerprint")
        and int(manifest.get("num_samples", -1)) == int(shard.get("num_samples", -2))
    )


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


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG that can affect cached-readout optimization."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a cached-readout RNG checkpoint exactly."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Resume checkpoint is missing RNG state: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("Resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def extraction_signature(
    config: RunConfig,
    *,
    dataset_name: str,
    dataset_root: Path,
    split: str,
    class_names: list[str] | None,
    offset: int,
    stop: int,
    code_tree_sha256: str,
    storage_policy: str,
    runtime_profile: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "config": config.to_dict(),
        "dataset": dataset_name,
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "class_names": class_names,
        "offset": offset,
        "stop": stop,
        "code_tree_sha256": code_tree_sha256,
        "storage_policy": storage_policy,
        "runtime_profile": runtime_profile,
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


def sample_noise_seed(sample_id: str, base_seed: int) -> int:
    """Derive a stable per-sample path-noise seed independent of batching."""

    encoded = f"{base_seed}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)


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
    storage_policy: str = "dense",
) -> dict[str, Any]:
    """Extract deterministic shards and checkpoint the manifest after every shard."""

    if storage_policy not in {"dense", "readout_sparse"}:
        raise ValueError("storage_policy must be dense or readout_sparse")
    total_started = time.perf_counter()
    provenance = code_provenance()
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
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    runtime_profile_path = os.environ.get("FIELDSCOPE_RUNTIME_PROFILE")
    runtime_profile = (
        runtime_profile_identity(runtime_profile_path)
        if runtime_profile_path
        else None
    )
    if runtime_profile is not None:
        if runtime_profile.get("code_revision") != provenance["code_revision"]:
            raise ValueError("Runtime profile revision does not match current code")
        if runtime_profile.get("code_tree_sha256") != provenance["code_tree_sha256"]:
            raise ValueError("Runtime profile code tree does not match current code")
        if runtime_profile.get("code_dirty") is not False:
            raise ValueError("Runtime profile came from a dirty worktree")
    signature = extraction_signature(
        config,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        split=split,
        class_names=class_names,
        offset=offset,
        stop=stop,
        code_tree_sha256=provenance["code_tree_sha256"],
        storage_policy=storage_policy,
        runtime_profile=runtime_profile,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"{manifest_path} exists; pass resume=True to continue"
            )
        if existing.get("extraction_signature") != signature:
            raise ValueError("Existing cache was produced by a different extraction")
        if (
            existing.get("complete") is True
            and int(existing.get("num_samples", -1)) == len(selected_indices)
            and sum(
                int(shard.get("num_samples", 0))
                for shard in existing.get("shards", [])
            )
            == len(selected_indices)
            and all(
                _verified_cached_shard(output_dir, shard)
                for shard in existing.get("shards", [])
            )
        ):
            return {
                **existing,
                "resume_status": "already_complete",
            }
        if existing.get("complete") is True:
            raise ValueError("Completed cache failed shard integrity verification")

    backend_started = time.perf_counter()
    backend = build_backend(config.backend)
    backend_seconds = time.perf_counter() - backend_started
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    extractor = FieldResponseExtractor(backend, config.probe)
    shards: list[dict[str, Any]] = []
    cache_sample_ids: list[str] = []
    sample_count = 0
    written_samples = 0
    extraction_seconds = 0.0
    cache_bytes = 0
    shard_size = config.runtime.cache_shard_size
    for local_start in range(0, len(dataset), shard_size):
        shard_started = time.perf_counter()
        local_end = min(len(dataset), local_start + shard_size)
        global_start = offset + local_start
        global_end = offset + local_end
        path = output_dir / f"shard-{global_start:08d}-{global_end:08d}.pt"
        shard_dataset = Subset(dataset, range(local_start, local_end))
        loader = DataLoader(
            shard_dataset,
            batch_size=config.runtime.batch_size,
            shuffle=False,
            num_workers=config.runtime.num_workers,
        )
        sample_ids: list[str] = []
        target_batches: dict[str, list[torch.Tensor]] = {}
        if resume and path.is_file():
            for batch in loader:
                sample_ids.extend(str(value) for value in batch["sample_id"])
                for name, value in _target_tensors(batch).items():
                    target_batches.setdefault(name, []).append(value)
            shard_targets = {
                name: torch.cat(values, dim=0)
                for name, values in target_batches.items()
            }
            _, cached_targets, cached_manifest = load_features(path)
            if cached_manifest.get("sample_ids") != sample_ids:
                raise ValueError(f"Sample IDs do not match resumed shard {path.name}")
            if set(cached_targets) != set(shard_targets):
                raise ValueError(f"Targets do not match resumed shard {path.name}")
            if any(
                not torch.equal(cached_targets[name], shard_targets[name])
                for name in cached_targets
            ):
                raise ValueError(f"Target values do not match resumed shard {path.name}")
            feature_manifest = cached_manifest
            status = "reused"
            shard_seconds = time.perf_counter() - shard_started
        else:
            feature_batches = []
            for batch in loader:
                images = batch["image"]
                if not isinstance(images, torch.Tensor):
                    raise TypeError("Dataset image batch must be a tensor")
                batch_sample_ids = [str(value) for value in batch["sample_id"]]
                sample_ids.extend(batch_sample_ids)
                for name, value in _target_tensors(batch).items():
                    target_batches.setdefault(name, []).append(value)
                feature_batches.append(
                    extractor.extract(
                        images,
                        noise_seeds=[
                            sample_noise_seed(sample_id, config.probe.seed)
                            for sample_id in batch_sample_ids
                        ],
                    ).detached_cpu()
                )
            features = stack_features(feature_batches)
            shard_targets = {
                name: torch.cat(values, dim=0)
                for name, values in target_batches.items()
            }
            feature_manifest = save_features(
                path,
                features,
                targets=shard_targets,
                sample_ids=sample_ids,
                storage_policy=storage_policy,
            )
            status = "written"
            shard_seconds = time.perf_counter() - shard_started
            extraction_seconds += shard_seconds
            written_samples += local_end - local_start
        shard_bytes = path.stat().st_size
        shard_sha256 = file_sha256(path)
        cache_bytes += shard_bytes
        shard = {
            "path": path.name,
            "start": global_start,
            "end": global_end,
            "num_samples": local_end - local_start,
            "fingerprint": feature_manifest["fingerprint"],
            "status": status,
            "seconds": shard_seconds,
            "bytes": shard_bytes,
            "sha256": shard_sha256,
        }
        shards.append(shard)
        cache_sample_ids.extend(sample_ids)
        sample_count += local_end - local_start
        report = {
            "status": "passed" if sample_count == len(selected_indices) else "running",
            "format_version": 3,
            "dataset": dataset_name,
            "split": split,
            "source_root": str(dataset_root.resolve()),
            "offset": offset,
            "requested_stop": stop,
            "num_samples": sample_count,
            "sample_ids_sha256": sample_ids_sha256(cache_sample_ids),
            "complete": sample_count == len(selected_indices),
            "extraction_signature": signature,
            "backend": backend.describe(),
            "config": config.to_dict(),
            "class_names": class_names,
            "storage_policy": storage_policy,
            "runtime_profile": runtime_profile,
            "dense_affinity_available": storage_policy == "dense",
            "randomness": {
                "path_noise": "sample_id_sha256_seeded_v1",
                "probe_basis": "shared_fixed_seed_v1",
                "base_seed": config.probe.seed,
            },
            **provenance,
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
        return grid_size
    return grid_size


def _task_loss_targets(
    task: str,
    targets: Mapping[str, torch.Tensor],
    output_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    target = targets[task]
    if task == "segmentation" and tuple(target.shape[-2:]) != output_size:
        target = torch.nn.functional.interpolate(
            target[:, None].float(),
            size=output_size,
            mode="nearest",
        ).squeeze(1).to(dtype=target.dtype)
    elif task == "depth" and tuple(target.shape[-2:]) != output_size:
        valid = torch.isfinite(target) & (target > 0)
        values = torch.where(valid, target, torch.zeros_like(target))
        pooled_values = torch.nn.functional.interpolate(
            values,
            size=output_size,
            mode="area",
        )
        pooled_valid = torch.nn.functional.interpolate(
            valid.to(dtype=target.dtype),
            size=output_size,
            mode="area",
        )
        target = torch.where(
            pooled_valid > 0,
            pooled_values / pooled_valid.clamp_min(1e-8),
            torch.zeros_like(pooled_values),
        )
    elif task == "normals" and tuple(target.shape[-2:]) != output_size:
        target = torch.nn.functional.interpolate(
            target,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
    return {task: target}


def _task_trainable_parameters(model: FieldScopeModel, task: str) -> int:
    prefixes = ("tokenizer.", f"heads.{task}.")
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(prefixes)
    )


def _cached_dataset(
    cache_dir: Path,
    representation: str,
    seed: int,
    memory_cache_gib: float = 0.0,
) -> CachedFeatureDataset | ShuffledResponseCachedDataset:
    memory_cache_bytes = int(memory_cache_gib * 1024**3)
    if representation in {"response_shuffled", "full_shuffled"}:
        return ShuffledResponseCachedDataset(
            cache_dir,
            seed,
            memory_cache_bytes=memory_cache_bytes,
        )
    if representation == "random_feature_local":
        return RandomFeatureCachedDataset(
            cache_dir,
            seed,
            memory_cache_bytes=memory_cache_bytes,
        )
    return CachedFeatureDataset(
        cache_dir,
        memory_cache_bytes=memory_cache_bytes,
    )


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


def _compute_meter(meter: Any, task: str) -> dict[str, Any]:
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
    batch_size: int | None = None,
    shuffle_seed: int = 4121,
) -> dict[str, Any]:
    dataset = _cached_dataset(
        cache_dir,
        representation,
        shuffle_seed,
        config.runtime.readout_memory_cache_gib,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config.runtime.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_cached, representation=representation),
    )
    meter = _make_meter(task, config)
    total_loss = 0.0
    total_samples = 0
    model.eval()
    for cached_batch in loader:
        features = cached_batch["features"].to(device, dtype=torch.float32)
        targets = {
            name: value.to(device) for name, value in cached_batch["targets"].items()
        }
        if task not in targets:
            raise ValueError(f"Requested task {task!r} is absent from cache")
        predictions = model(
            features,
            output_size=_task_output_size(task, targets, features.grid_size),
            task=task,
        )
        loss_targets = _task_loss_targets(
            task,
            targets,
            tuple(predictions[task].shape[-2:]),
        )
        loss, _ = multitask_loss(predictions, loss_targets)
        batch_size = features.state.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        metric_predictions = predictions
        if task == "segmentation" and tuple(
            predictions[task].shape[-2:]
        ) != tuple(targets[task].shape[-2:]):
            metric_predictions = {
                task: torch.nn.functional.interpolate(
                    predictions[task].argmax(dim=1, keepdim=True).float(),
                    size=targets[task].shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
            }
        elif task in {"depth", "normals"} and tuple(
            predictions[task].shape[-2:]
        ) != tuple(targets[task].shape[-2:]):
            metric_predictions = {
                task: torch.nn.functional.interpolate(
                    predictions[task],
                    size=targets[task].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            }
        _update_meter(meter, task, metric_predictions, targets)
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
    batch_size: int | None = None,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    run_started = time.perf_counter()
    provenance = code_provenance()
    report_config = _json_compatible(config.to_dict())
    train_cache = cache_identity(train_cache_dir)
    validation_cache = cache_identity(val_cache_dir)
    set_experiment_seed(seed, config.runtime.deterministic)
    train_dataset = _cached_dataset(
        train_cache_dir,
        representation,
        seed,
        config.runtime.readout_memory_cache_gib,
    )
    val_dataset = _cached_dataset(
        val_cache_dir,
        representation,
        seed,
        config.runtime.readout_memory_cache_gib,
    )
    train_control_contract = cached_control_contract(train_dataset)
    validation_control_contract = cached_control_contract(val_dataset)
    first = train_dataset[0]
    selected, mode = select_representation(first["features"], representation)
    device = torch.device(config.backend.device)
    model = FieldScopeModel(
        selected.state.shape[-1],
        selected.response.shape[-1],
        config.tokenizer,
        mode=mode,
    ).to(device)
    for head_name, head in model.heads.named_children():
        if head_name != task:
            head.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = 0
    readout_batch_size = batch_size or config.runtime.batch_size
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if payload.get("task") != task:
            raise ValueError("Resume checkpoint task does not match requested task")
        if payload.get("representation") != representation:
            raise ValueError(
                "Resume checkpoint representation does not match requested representation"
            )
        if int(payload.get("seed", -1)) != seed:
            raise ValueError("Resume checkpoint seed does not match requested seed")
        if payload.get("config") != config.to_dict():
            raise ValueError("Resume checkpoint config does not match requested config")
        if float(payload.get("learning_rate", float("nan"))) != learning_rate:
            raise ValueError("Resume checkpoint learning rate does not match")
        if float(payload.get("weight_decay", float("nan"))) != weight_decay:
            raise ValueError("Resume checkpoint weight decay does not match")
        if int(payload.get("batch_size", -1)) != readout_batch_size:
            raise ValueError("Resume checkpoint batch size does not match")
        if int(payload.get("target_epochs", -1)) != epochs:
            raise ValueError("Resume checkpoint target epochs do not match")
        if payload.get("code_revision") != provenance["code_revision"]:
            raise ValueError("Resume checkpoint revision does not match current code")
        if payload.get("code_tree_sha256") != provenance["code_tree_sha256"]:
            raise ValueError("Resume checkpoint code tree does not match current code")
        if payload.get("code_dirty") != provenance["code_dirty"]:
            raise ValueError("Resume checkpoint worktree state does not match current code")
        if payload.get("train_cache") != train_cache:
            raise ValueError("Resume checkpoint train cache does not match")
        if payload.get("validation_cache") != validation_cache:
            raise ValueError("Resume checkpoint validation cache does not match")
        if payload.get("train_control_contract") != train_control_contract:
            raise ValueError("Resume checkpoint train control contract does not match")
        if payload.get("validation_control_contract") != validation_control_contract:
            raise ValueError(
                "Resume checkpoint validation control contract does not match"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        history = list(payload.get("history", []))
        best_value = payload.get("best_primary_metric")
        best_epoch = int(payload.get("best_epoch", 0))
        restore_rng_state(payload.get("rng_state", {}))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{task}_{representation}_seed{seed}_best.pt"
    last_path = output_dir / f"{task}_{representation}_seed{seed}_last.pt"
    report: dict[str, Any] | None = None
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        train_loader = DataLoader(
            train_dataset,
            batch_size=readout_batch_size,
            sampler=ShardShuffleSampler(train_dataset, seed=seed + epoch),
            num_workers=0,
            collate_fn=partial(collate_cached, representation=representation),
        )
        model.train()
        total_loss = 0.0
        total_samples = 0
        for cached_batch in train_loader:
            features = cached_batch["features"].to(device, dtype=torch.float32)
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
                task=task,
            )
            loss_targets = _task_loss_targets(
                task,
                targets,
                tuple(predictions[task].shape[-2:]),
            )
            loss, _ = multitask_loss(predictions, loss_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = features.state.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
        train_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        validation_started = time.perf_counter()
        validation = evaluate_cached_readout(
            model,
            config,
            cache_dir=val_cache_dir,
            task=task,
            representation=representation,
            device=device,
            batch_size=readout_batch_size,
            shuffle_seed=seed,
        )
        validation_seconds = time.perf_counter() - validation_started
        entry = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(1, total_samples),
            "train_seconds": train_seconds,
            "train_samples_per_second": total_samples / max(train_seconds, 1e-12),
            "validation_seconds": validation_seconds,
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
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": readout_batch_size,
            "target_epochs": epochs,
            "train_cache": train_cache,
            "validation_cache": validation_cache,
            "train_control_contract": train_control_contract,
            "validation_control_contract": validation_control_contract,
            "rng_state": capture_rng_state(),
            **provenance,
            "best_epoch": best_epoch,
            "best_primary_metric": best_value,
        }
        if is_best:
            atomic_torch_save(best_path, checkpoint_payload)
        # The last checkpoint is the epoch commit marker; write it only after
        # every companion artifact required to resume/evaluate this epoch.
        atomic_torch_save(last_path, checkpoint_payload)
        report = {
            "status": "running" if epoch + 1 < epochs else "passed",
            "task": task,
            "representation": representation,
            "seed": seed,
            **provenance,
            "config": report_config,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_cache": train_cache,
            "validation_cache": validation_cache,
            "train_control_contract": train_control_contract,
            "validation_control_contract": validation_control_contract,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "trainable_parameters": _task_trainable_parameters(model, task),
            "batch_size": readout_batch_size,
            "readout_memory_cache": shared_memory_cache_stats(),
            "runtime": {
                "elapsed_seconds": time.perf_counter() - run_started,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None
                ),
            },
            "best_epoch": best_epoch,
            "best_primary_metric": best_value,
            "checkpoint": str(best_path),
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": file_sha256(best_path),
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
            batch_size=readout_batch_size,
            shuffle_seed=seed,
        )
        report = {
            "status": "passed",
            "task": task,
            "representation": representation,
            "seed": seed,
            **provenance,
            "config": report_config,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_cache": train_cache,
            "validation_cache": validation_cache,
            "train_control_contract": train_control_contract,
            "validation_control_contract": validation_control_contract,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "trainable_parameters": _task_trainable_parameters(model, task),
            "batch_size": readout_batch_size,
            "readout_memory_cache": shared_memory_cache_stats(),
            "runtime": {
                "elapsed_seconds": time.perf_counter() - run_started,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None
                ),
            },
            "best_epoch": best_epoch,
            "best_primary_metric": best_value,
            "checkpoint": str(best_path),
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": file_sha256(best_path),
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
    batch_size: int | None = None,
) -> dict[str, Any]:
    device = torch.device(config.backend.device)
    provenance = code_provenance()
    test_cache = cache_identity(cache_dir)
    checkpoint_sha256 = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    test_dataset = _cached_dataset(
        cache_dir,
        str(payload["representation"]),
        int(payload["seed"]),
        config.runtime.readout_memory_cache_gib,
    )
    test_control_contract = cached_control_contract(test_dataset)
    if payload.get("code_revision") != provenance["code_revision"]:
        raise ValueError("Checkpoint revision does not match current code")
    if payload.get("code_tree_sha256") != provenance["code_tree_sha256"]:
        raise ValueError("Checkpoint code tree does not match current code")
    if payload.get("code_dirty") != provenance["code_dirty"]:
        raise ValueError("Checkpoint worktree state does not match current code")
    if payload.get("config") != config.to_dict():
        raise ValueError("Checkpoint config does not match evaluation config")
    checkpoint_config = RunConfig.from_mapping(payload["config"])
    model = FieldScopeModel(
        payload["state_dim"],
        payload["response_dim"],
        checkpoint_config.tokenizer,
        mode=payload["mode"],
    ).to(device)
    for head_name, head in model.heads.named_children():
        if head_name != payload["task"]:
            head.requires_grad_(False)
    model.load_state_dict(payload["model"])
    evaluation = evaluate_cached_readout(
        model,
        checkpoint_config,
        cache_dir=cache_dir,
        task=payload["task"],
        representation=payload["representation"],
        device=device,
        batch_size=batch_size,
        shuffle_seed=int(payload["seed"]),
    )
    return {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "task": payload["task"],
        "representation": payload["representation"],
        "seed": payload["seed"],
        **provenance,
        "config": _json_compatible(payload["config"]),
        "batch_size": batch_size or checkpoint_config.runtime.batch_size,
        "test_cache": test_cache,
        "test_control_contract": test_control_contract,
        "evaluation": evaluation,
    }


def diagnose_segmentation_cache(cache_dir: Path) -> dict[str, Any]:
    dataset = CachedFeatureDataset(cache_dir)
    if dataset.manifest.get("storage_policy", "dense") != "dense":
        raise ValueError(
            "Unsupervised graph diagnosis requires a dense-affinity cache"
        )
    shuffled_dataset = ShuffledResponseCachedDataset(cache_dir, seed=4121)
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    per_sample: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        features = sample["features"]
        targets = sample["targets"]
        if "segmentation" not in targets:
            raise ValueError("Segmentation targets are required for graph diagnosis")
        representations = {
            "response": features.affinity.float(),
            "response_shuffled": shuffled_dataset[index][
                "features"
            ].affinity.float(),
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
        sample_report: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "representations": {},
        }
        for name, affinity in representations.items():
            metrics = graph_segmentation_metrics(
                affinity,
                target,
                features.grid_size,
            )
            accumulator = totals.setdefault(
                name, {metric_name: 0.0 for metric_name in metrics}
            )
            metric_counts = counts.setdefault(
                name, {metric_name: 0 for metric_name in metrics}
            )
            metric_values = values.setdefault(
                name, {metric_name: [] for metric_name in metrics}
            )
            sample_report["representations"][name] = metrics
            for metric_name, value in metrics.items():
                if not np.isfinite(value):
                    continue
                accumulator[metric_name] += value
                metric_counts[metric_name] += 1
                metric_values[metric_name].append(value)
        per_sample.append(sample_report)
    return {
        "status": "passed",
        **code_provenance(),
        "cache_dir": str(cache_dir),
        "num_samples": len(dataset),
        "representations": {
            name: {
                "means": {
                    metric_name: value
                    / max(1, counts.get(name, {}).get(metric_name, 0))
                    for metric_name, value in metrics.items()
                },
                "bootstrap": {
                    metric_name: bootstrap_mean_interval(
                        metric_values,
                        seed=4121,
                    )
                    for metric_name, metric_values in values[name].items()
                    if metric_values
                },
                "valid_samples": counts.get(name, {}),
            }
            for name, metrics in totals.items()
        },
        "per_sample": per_sample,
    }


def run_readout_matrix(
    config: RunConfig,
    *,
    train_cache_dir: Path,
    val_cache_dir: Path,
    test_cache_dir: Path,
    output_dir: Path,
    task: str,
    representations: list[str],
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    reference: str | None = None,
) -> dict[str, Any]:
    """Run and resume a representation/seed matrix through held-out test."""

    if not representations or not seeds:
        raise ValueError("representations and seeds must be non-empty")
    metric = {
        "classification": "top1",
        "segmentation": "mean_iou",
        "depth": "abs_rel",
        "normals": "mean_angular_error",
    }[task]
    output_dir.mkdir(parents=True, exist_ok=True)
    test_reports: list[Path] = []
    runs: list[dict[str, Any]] = []
    matrix_path = output_dir / "matrix_report.json"
    provenance = code_provenance()
    config_payload = _json_compatible(config.to_dict())
    readout_batch_size = batch_size or config.runtime.batch_size
    train_cache = cache_identity(train_cache_dir)
    validation_cache = cache_identity(val_cache_dir)
    test_cache = cache_identity(test_cache_dir)
    for representation in representations:
        for seed in seeds:
            train_control_contract = cached_control_contract_for_cache(
                train_cache_dir, representation, seed
            )
            validation_control_contract = cached_control_contract_for_cache(
                val_cache_dir, representation, seed
            )
            test_control_contract = cached_control_contract_for_cache(
                test_cache_dir, representation, seed
            )
            run_dir = output_dir / representation / f"seed-{seed}"
            training_report_path = (
                run_dir / f"{task}_{representation}_seed{seed}_report.json"
            )
            last_checkpoint = run_dir / f"{task}_{representation}_seed{seed}_last.pt"
            if training_report_path.is_file():
                training_report = json.loads(
                    training_report_path.read_text(encoding="utf-8")
                )
                _require_report_contract(
                    training_report,
                    {
                        "task": task,
                        "representation": representation,
                        "seed": seed,
                        "code_revision": provenance["code_revision"],
                        "code_dirty": provenance["code_dirty"],
                        "code_tree_sha256": provenance["code_tree_sha256"],
                        "config": config_payload,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "batch_size": readout_batch_size,
                        "train_cache": train_cache,
                        "validation_cache": validation_cache,
                        "train_control_contract": train_control_contract,
                        "validation_control_contract": validation_control_contract,
                    },
                    report_path=training_report_path,
                    kind="training",
                )
            else:
                training_report = {}
            if training_report.get("status") != "passed":
                training_report = train_cached_readout(
                    config,
                    train_cache_dir=train_cache_dir,
                    val_cache_dir=val_cache_dir,
                    output_dir=run_dir,
                    task=task,
                    representation=representation,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=seed,
                    resume_checkpoint=(
                        last_checkpoint if last_checkpoint.is_file() else None
                    ),
                    batch_size=batch_size,
                )
            best_checkpoint = Path(training_report["best_checkpoint"])
            if not best_checkpoint.is_file():
                raise FileNotFoundError(f"Missing best checkpoint: {best_checkpoint}")
            best_checkpoint_sha256 = file_sha256(best_checkpoint)
            if training_report.get("best_checkpoint_sha256") != best_checkpoint_sha256:
                raise ValueError(
                    f"Best checkpoint SHA-256 mismatch: {best_checkpoint}"
                )
            test_report_path = run_dir / f"{task}_{representation}_seed{seed}_test.json"
            if test_report_path.is_file():
                test_report = json.loads(test_report_path.read_text(encoding="utf-8"))
                _require_report_contract(
                    test_report,
                    {
                        "task": task,
                        "representation": representation,
                        "seed": seed,
                        "code_revision": provenance["code_revision"],
                        "code_dirty": provenance["code_dirty"],
                        "code_tree_sha256": provenance["code_tree_sha256"],
                        "config": config_payload,
                        "batch_size": readout_batch_size,
                        "test_cache": test_cache,
                        "test_control_contract": test_control_contract,
                        "checkpoint": str(best_checkpoint),
                        "checkpoint_sha256": best_checkpoint_sha256,
                    },
                    report_path=test_report_path,
                    kind="test",
                )
            else:
                test_report = evaluate_checkpoint(
                    config,
                    checkpoint=best_checkpoint,
                    cache_dir=test_cache_dir,
                    batch_size=batch_size,
                )
                atomic_json_dump(test_report_path, test_report)
            test_reports.append(test_report_path)
            runs.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "training_report": str(training_report_path),
                    "best_checkpoint": str(best_checkpoint),
                    "best_checkpoint_sha256": best_checkpoint_sha256,
                    "test_report": str(test_report_path),
                    "test_metric": test_report["evaluation"]["metrics"][metric],
                }
            )
            atomic_json_dump(
                matrix_path,
                {
                    "status": "running",
                    "task": task,
                    "metric": metric,
                    "runs": runs,
                },
            )
    summary = summarize_run_reports(
        test_reports,
        metric=metric,
        reference=reference,
    )
    report = {
        "status": "passed",
        **code_provenance(),
        "task": task,
        "metric": metric,
        "representations": representations,
        "seeds": seeds,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": readout_batch_size,
        "config": config_payload,
        "train_cache_dir": str(train_cache_dir),
        "val_cache_dir": str(val_cache_dir),
        "test_cache_dir": str(test_cache_dir),
        "train_cache": train_cache,
        "validation_cache": validation_cache,
        "test_cache": test_cache,
        "runs": runs,
        "summary": summary,
    }
    atomic_json_dump(matrix_path, report)
    return report
