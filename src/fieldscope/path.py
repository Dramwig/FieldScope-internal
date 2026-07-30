"""Image-anchored rectified-flow path utilities."""

from __future__ import annotations

import torch


def _time_view(clean_time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if clean_time.ndim == 0:
        clean_time = clean_time.expand(target.shape[0])
    if clean_time.shape != (target.shape[0],):
        raise ValueError("clean_time must be scalar or shape [B]")
    return clean_time.reshape(target.shape[0], *([1] * (target.ndim - 1)))


def rectified_state(
    z0: torch.Tensor, noise: torch.Tensor, clean_time: torch.Tensor
) -> torch.Tensor:
    """Return z_t=(1-t) noise + t z0 with t in clean-time coordinates."""

    if z0.shape != noise.shape:
        raise ValueError("z0 and noise must have identical shapes")
    time = _time_view(clean_time.to(device=z0.device, dtype=z0.dtype), z0)
    return (1.0 - time) * noise + time * z0


def rectified_tangent(z0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if z0.shape != noise.shape:
        raise ValueError("z0 and noise must have identical shapes")
    return z0 - noise


def endpoint_estimate(
    state: torch.Tensor, velocity: torch.Tensor, clean_time: torch.Tensor
) -> torch.Tensor:
    """First-order clean endpoint estimate for clean-time velocity."""

    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have identical shapes")
    time = _time_view(clean_time.to(device=state.device, dtype=state.dtype), state)
    return state + (1.0 - time) * velocity
