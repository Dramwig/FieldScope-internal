"""Backend construction."""

from __future__ import annotations

from fieldscope.config import BackendConfig
from fieldscope.contracts import FieldBackend


def build_backend(config: BackendConfig) -> FieldBackend:
    if config.name == "toy":
        from fieldscope.backends.toy import ToyFieldBackend

        return ToyFieldBackend(
            device=config.device,
            dtype=config.dtype,
            image_size=config.image_size,
        )
    if config.name == "auraflow":
        from fieldscope.backends.auraflow import AuraFlowBackend

        return AuraFlowBackend(config)
    raise ValueError(f"Unsupported backend: {config.name}")

