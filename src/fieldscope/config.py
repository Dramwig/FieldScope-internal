"""Typed configuration and validation for FieldScope experiments."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


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


@dataclass(frozen=True)
class ProbeConfig:
    times: tuple[float, ...] = (0.2, 0.5, 0.8)
    num_directions: int = 4
    eta: float = 0.03
    difference: str = "central"
    graph_grid: tuple[int, int] = (8, 8)
    topk: int = 8
    local_radius: int = 1
    probe_batch_size: int = 4
    antithetic_noise: bool = True
    seed: int = 4121

    def validate(self) -> None:
        if not self.times or any(not 0.0 < time < 1.0 for time in self.times):
            raise ValueError("probe.times must contain values strictly between 0 and 1")
        if tuple(sorted(self.times)) != self.times:
            raise ValueError("probe.times must be sorted in ascending clean-time order")
        if self.num_directions < 1:
            raise ValueError("probe.num_directions must be positive")
        if self.eta <= 0:
            raise ValueError("probe.eta must be positive")
        if self.difference not in {"central", "forward"}:
            raise ValueError("probe.difference must be central or forward")
        if min(self.graph_grid) < 1:
            raise ValueError("probe.graph_grid entries must be positive")
        if self.topk < 0 or self.local_radius < 0 or self.probe_batch_size < 1:
            raise ValueError("topk/local_radius/probe_batch_size are invalid")


@dataclass(frozen=True)
class TokenizerConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0
    num_classes: int = 10
    segmentation_classes: int = 21

    def validate(self) -> None:
        if self.hidden_dim < 8 or self.num_layers < 1:
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
    num_workers: int = 0
    deterministic: bool = True

    def validate(self) -> None:
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("runtime batch_size/num_workers are invalid")


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


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Top-level configuration must be a mapping")
    return RunConfig.from_mapping(raw)
