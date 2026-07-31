"""Structured perturbation directions for randomized field sketches."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rademacher(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    values = torch.randint(0, 2, shape, generator=generator, device=device)
    return values.to(dtype=dtype).mul_(2).sub_(1)


def _normalize(direction: torch.Tensor) -> torch.Tensor:
    rms = direction.square().mean(dim=(-3, -2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    return direction / rms


def generate_structured_probes(
    reference: torch.Tensor,
    num_directions: int,
    seed: int,
) -> torch.Tensor:
    """Generate [B,R,C,H,W] probes cycling through four structure families."""

    if reference.ndim != 4:
        raise ValueError("reference must have shape [B,C,H,W]")
    if num_directions < 1:
        raise ValueError("num_directions must be positive")
    batch, channels, height, width = reference.shape
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed)
    probes: list[torch.Tensor] = []

    yy = torch.linspace(0, 2 * math.pi, height, device=reference.device, dtype=reference.dtype)
    xx = torch.linspace(0, 2 * math.pi, width, device=reference.device, dtype=reference.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    for index in range(num_directions):
        family = index % 4
        if family == 0:
            direction = _rademacher(
                (batch, channels, height, width),
                generator,
                reference.device,
                reference.dtype,
            )
        elif family == 1:
            coarse_h = max(1, height // (2 + index % 3))
            coarse_w = max(1, width // (2 + (index + 1) % 3))
            coarse = _rademacher(
                (batch, channels, coarse_h, coarse_w),
                generator,
                reference.device,
                reference.dtype,
            )
            direction = F.interpolate(coarse, size=(height, width), mode="nearest")
        elif family == 2:
            frequency_x = 1 + index % max(1, min(4, width // 2))
            frequency_y = 1 + (index // 2) % max(1, min(4, height // 2))
            phase = torch.rand(
                (batch, channels, 1, 1),
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
            phase = phase * (2 * math.pi)
            channel_sign = _rademacher(
                (batch, channels, 1, 1),
                generator,
                reference.device,
                reference.dtype,
            )
            wave = torch.cos(frequency_x * grid_x + frequency_y * grid_y)
            direction = channel_sign * torch.cos(wave[None, None] * math.pi + phase)
        else:
            direction = 0.05 * _rademacher(
                (batch, channels, height, width),
                generator,
                reference.device,
                reference.dtype,
            )
            for batch_index in range(batch):
                window_h = max(1, height // (2 + (index % 2)))
                window_w = max(1, width // (2 + ((index + 1) % 2)))
                top = int(
                    torch.randint(
                        0,
                        max(1, height - window_h + 1),
                        (1,),
                        generator=generator,
                        device=reference.device,
                    ).item()
                )
                left = int(
                    torch.randint(
                        0,
                        max(1, width - window_w + 1),
                        (1,),
                        generator=generator,
                        device=reference.device,
                    ).item()
                )
                signs = _rademacher(
                    (channels, 1, 1), generator, reference.device, reference.dtype
                )
                direction[
                    batch_index, :, top : top + window_h, left : left + window_w
                ] += signs
        probes.append(_normalize(direction))

    return torch.stack(probes, dim=1)


def generate_probes(
    reference: torch.Tensor,
    num_directions: int,
    seed: int,
    probe_type: str,
) -> torch.Tensor:
    """Generate structured, Gaussian, or spatially destroyed probe controls."""

    if probe_type == "structured":
        return generate_structured_probes(reference, num_directions, seed)
    if probe_type == "gaussian":
        generator = torch.Generator(device=reference.device).manual_seed(seed)
        probes = torch.randn(
            (
                reference.shape[0],
                num_directions,
                *reference.shape[1:],
            ),
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        return _normalize(probes)
    if probe_type == "spatially_shuffled":
        probes = generate_structured_probes(reference, num_directions, seed)
        generator = torch.Generator(device=reference.device).manual_seed(seed + 7919)
        flat = probes.flatten(-2)
        for batch_index in range(flat.shape[0]):
            for direction_index in range(flat.shape[1]):
                permutation = torch.randperm(
                    flat.shape[-1],
                    generator=generator,
                    device=reference.device,
                )
                flat[batch_index, direction_index] = flat[
                    batch_index,
                    direction_index,
                    :,
                    permutation,
                ]
        return flat.reshape_as(probes)
    raise ValueError(f"Unsupported probe_type: {probe_type}")
