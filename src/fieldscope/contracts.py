"""Shared tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


class FieldBackend(Protocol):
    """Frozen image-to-latent vector-field interface.

    Public time is clean-time: 0 is noise, 1 is the encoded image. Backends are
    responsible for converting this convention to their native scheduler time.
    """

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def model_id(self) -> str: ...

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B,3,H,W] images in [0,1] into latent states."""

    def query_velocity(self, latents: torch.Tensor, clean_time: torch.Tensor) -> torch.Tensor:
        """Return dz/d(clean_time) at the supplied latent states."""

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe backend description."""


@dataclass
class FieldFeatures:
    """Cached FieldScope representation for a batch."""

    state: torch.Tensor
    response: torch.Tensor
    affinity: torch.Tensor
    adjacency: torch.Tensor
    grid_size: tuple[int, int]
    baselines: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.state.ndim != 3 or self.response.ndim != 3:
            raise ValueError("state and response must have shape [B,P,D]")
        batch, patches, _ = self.state.shape
        if self.response.shape[:2] != (batch, patches):
            raise ValueError("state and response patch dimensions must match")
        if self.affinity.shape != (batch, patches, patches):
            raise ValueError("affinity must have shape [B,P,P]")
        if self.adjacency.shape != self.affinity.shape:
            raise ValueError("adjacency and affinity shapes must match")
        if patches != self.grid_size[0] * self.grid_size[1]:
            raise ValueError("grid_size does not match the number of patches")
        for name, tensor in self.baselines.items():
            if tensor.shape[:2] != (batch, patches):
                raise ValueError(f"Baseline {name!r} does not match batch/patch dimensions")
        tensors = [self.state, self.response, self.affinity, self.adjacency]
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("FieldFeatures contains non-finite values")

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> FieldFeatures:
        return FieldFeatures(
            state=self.state.to(device=device, dtype=dtype),
            response=self.response.to(device=device, dtype=dtype),
            affinity=self.affinity.to(device=device, dtype=dtype),
            adjacency=self.adjacency.to(device=device, dtype=dtype),
            grid_size=self.grid_size,
            baselines={
                name: value.to(device=device, dtype=dtype)
                for name, value in self.baselines.items()
            },
            metadata=dict(self.metadata),
        )

    def detached_cpu(self) -> FieldFeatures:
        return FieldFeatures(
            state=self.state.detach().cpu(),
            response=self.response.detach().cpu(),
            affinity=self.affinity.detach().cpu(),
            adjacency=self.adjacency.detach().cpu(),
            grid_size=self.grid_size,
            baselines={name: value.detach().cpu() for name, value in self.baselines.items()},
            metadata=dict(self.metadata),
        )
