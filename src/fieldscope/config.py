"""Typed configuration and validation for FieldScope experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

_RUNTIME_PROFILE_SCHEMA_VERSION = 1
_REGISTERED_RUNTIME_PROFILES = {(2, 8), (2, 16), (4, 32), (8, 64)}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class BackendConfig:
    name: str = "toy"
    model_path: str | None = None
    variant: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    image_size: int = 128
    prompt: str = ""
    max_sequence_length: int = 256
    local_files_only: bool = True
    offload_text_encoder: bool = True
    random_transformer: bool = False
    random_transformer_seed: int = 104729

    def validate(self) -> None:
        if self.name not in {"toy", "auraflow"}:
            raise ValueError(f"Unsupported backend: {self.name}")
        if self.name == "auraflow" and not self.model_path:
            raise ValueError("backend.model_path is required for the auraflow backend")
        if self.variant is not None and not self.variant.strip():
            raise ValueError("backend.variant must be null or a non-empty string")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.image_size <= 0 or self.image_size % 16:
            raise ValueError("backend.image_size must be a positive multiple of 16")
        if self.random_transformer and self.name != "auraflow":
            raise ValueError("backend.random_transformer is only supported by auraflow")
        if self.random_transformer_seed < 0:
            raise ValueError("backend.random_transformer_seed must be non-negative")


@dataclass(frozen=True)
class ProbeConfig:
    times: tuple[float, ...] = (0.2, 0.5, 0.8)
    num_directions: int = 4
    probe_type: str = "structured"
    eta: float = 0.03
    difference: str = "central"
    graph_grid: tuple[int, int] = (8, 8)
    topk: int = 8
    local_radius: int = 1
    probe_batch_size: int = 4
    antithetic_noise: bool = True
    hidden_baseline_dim: int = 768
    seed: int = 4121

    def validate(self) -> None:
        if not self.times or any(not 0.0 < time < 1.0 for time in self.times):
            raise ValueError("probe.times must contain values strictly between 0 and 1")
        if tuple(sorted(self.times)) != self.times:
            raise ValueError("probe.times must be sorted in ascending clean-time order")
        if self.num_directions < 1:
            raise ValueError("probe.num_directions must be positive")
        if self.probe_type not in {"structured", "gaussian", "spatially_shuffled"}:
            raise ValueError(
                "probe.probe_type must be structured, gaussian, or spatially_shuffled"
            )
        if self.eta <= 0:
            raise ValueError("probe.eta must be positive")
        if self.difference not in {"central", "forward"}:
            raise ValueError("probe.difference must be central or forward")
        if min(self.graph_grid) < 1:
            raise ValueError("probe.graph_grid entries must be positive")
        if self.topk < 0 or self.local_radius < 0 or self.probe_batch_size < 1:
            raise ValueError("topk/local_radius/probe_batch_size are invalid")
        if self.hidden_baseline_dim < 1:
            raise ValueError("probe.hidden_baseline_dim must be positive")


@dataclass(frozen=True)
class TokenizerConfig:
    hidden_dim: int = 128
    input_dim: int = 768
    num_layers: int = 3
    dropout: float = 0.0
    num_classes: int = 10
    segmentation_classes: int = 21

    def validate(self) -> None:
        if self.hidden_dim < 8 or self.input_dim < 1 or self.num_layers < 1:
            raise ValueError("tokenizer dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("tokenizer.dropout must be in [0, 1)")
        if self.num_classes < 1 or self.segmentation_classes < 1:
            raise ValueError("task class counts must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str = "outputs/smoke"
    cache_dir: str = "artifacts/cache"
    batch_size: int = 2
    cache_shard_size: int = 64
    readout_memory_cache_gib: float = 0.0
    num_workers: int = 0
    deterministic: bool = True

    def validate(self) -> None:
        if (
            self.batch_size < 1
            or self.cache_shard_size < self.batch_size
            or self.readout_memory_cache_gib < 0
            or self.num_workers < 0
        ):
            raise ValueError(
                "runtime batch_size/cache_shard_size/readout cache/num_workers "
                "are invalid"
            )


@dataclass(frozen=True)
class RunConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> RunConfig:
        expanded = _expand(dict(mapping))
        backend = _section(expanded, "backend")
        if backend.get("model_path"):
            backend["model_path"] = os.path.normpath(backend["model_path"])
        probe = _section(expanded, "probe")
        if "times" in probe:
            probe["times"] = tuple(float(value) for value in probe["times"])
        if "graph_grid" in probe:
            probe["graph_grid"] = tuple(int(value) for value in probe["graph_grid"])
        runtime = _section(expanded, "runtime")
        for name in ("output_dir", "cache_dir"):
            if runtime.get(name):
                runtime[name] = os.path.normpath(runtime[name])
        config = cls(
            backend=BackendConfig(**backend),
            probe=ProbeConfig(**probe),
            tokenizer=TokenizerConfig(**_section(expanded, "tokenizer")),
            runtime=RuntimeConfig(**runtime),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.backend.validate()
        self.probe.validate()
        self.tokenizer.validate()
        self.runtime.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def runtime_method_contract(config: RunConfig) -> dict[str, Any]:
    """Return method fields that a runtime batching profile may not change."""

    return {
        "backend": {
            "name": config.backend.name,
            "model_name": (
                Path(config.backend.model_path).name
                if config.backend.model_path is not None
                else None
            ),
            "variant": config.backend.variant,
            "device": config.backend.device,
            "dtype": config.backend.dtype,
            "image_size": config.backend.image_size,
            "max_sequence_length": config.backend.max_sequence_length,
            "local_files_only": config.backend.local_files_only,
            "offload_text_encoder": config.backend.offload_text_encoder,
        },
        "probe": {
            "times": list(config.probe.times),
            "num_directions": config.probe.num_directions,
            "eta": config.probe.eta,
            "difference": config.probe.difference,
            "graph_grid": list(config.probe.graph_grid),
            "topk": config.probe.topk,
            "local_radius": config.probe.local_radius,
            "antithetic_noise": config.probe.antithetic_noise,
            "hidden_baseline_dim": config.probe.hidden_baseline_dim,
            "seed": config.probe.seed,
        },
        "runtime": {
            "cache_shard_size": config.runtime.cache_shard_size,
            "deterministic": config.runtime.deterministic,
        },
    }


def runtime_method_contract_sha256(config: RunConfig) -> str:
    encoded = json.dumps(
        runtime_method_contract(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_profile_identity(path: str | Path) -> dict[str, Any]:
    """Load the immutable identity embedded in caches using a runtime gate."""

    profile_path = Path(path).expanduser().resolve()
    raw = profile_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Runtime profile must be a JSON object")
    selected = payload.get("selected_profile", {})
    image_batch_size = int(selected.get("image_batch_size", -1))
    probe_batch_size = int(selected.get("probe_batch_size", -1))
    if payload.get("schema_version") != _RUNTIME_PROFILE_SCHEMA_VERSION:
        raise ValueError("Runtime profile schema version mismatch")
    if payload.get("status") != "passed":
        raise ValueError("Runtime profile is not passed")
    if (image_batch_size, probe_batch_size) not in _REGISTERED_RUNTIME_PROFILES:
        raise ValueError("Runtime profile selected an unregistered batch shape")
    return {
        "path": str(profile_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": payload["schema_version"],
        "code_revision": payload.get("code_revision"),
        "code_tree_sha256": payload.get("code_tree_sha256"),
        "method_runtime_contract_sha256": payload.get(
            "method_runtime_contract_sha256"
        ),
        "selected_profile": {
            "image_batch_size": image_batch_size,
            "probe_batch_size": probe_batch_size,
        },
    }


def apply_runtime_profile(config: RunConfig, path: str | Path) -> RunConfig:
    identity = runtime_profile_identity(path)
    if identity["method_runtime_contract_sha256"] != runtime_method_contract_sha256(
        config
    ):
        raise ValueError("Runtime profile method contract does not match the config")
    selected = identity["selected_profile"]
    return replace(
        config,
        probe=replace(
            config.probe,
            probe_batch_size=selected["probe_batch_size"],
        ),
        runtime=replace(
            config.runtime,
            batch_size=selected["image_batch_size"],
        ),
    )


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Top-level configuration must be a mapping")
    config = RunConfig.from_mapping(raw)
    runtime_profile = os.environ.get("FIELDSCOPE_RUNTIME_PROFILE")
    if runtime_profile:
        config = apply_runtime_profile(config, runtime_profile)
    return config
