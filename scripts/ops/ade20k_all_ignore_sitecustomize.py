"""H200 compatibility shims for fixed-revision FieldScope readout recovery.

This file is installed as ``sitecustomize.py`` in a content-addressed external
runtime directory. It keeps the existing checkpoint RNG-device compatibility
shim and defines the mathematically empty ADE20K segmentation batch as a
differentiable zero contribution. Every batch containing at least one valid
pixel continues through PyTorch's original cross-entropy implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional

_original_torch_load = torch.load
_original_cross_entropy = functional.cross_entropy


def _rng_compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
    payload = _original_torch_load(*args, **kwargs)
    if not isinstance(payload, dict):
        return payload
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        return payload
    torch_cpu = rng_state.get("torch_cpu")
    if isinstance(torch_cpu, torch.Tensor) and torch_cpu.device.type != "cpu":
        rng_state["torch_cpu"] = torch_cpu.cpu()
    torch_cuda = rng_state.get("torch_cuda")
    if isinstance(torch_cuda, list):
        rng_state["torch_cuda"] = [
            value.cpu()
            if isinstance(value, torch.Tensor) and value.device.type != "cpu"
            else value
            for value in torch_cuda
        ]
    return payload


def _ignore_index(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    # ``cross_entropy(input, target, weight, size_average, ignore_index, ...)``
    if "ignore_index" in kwargs:
        return int(kwargs["ignore_index"])
    if len(args) >= 3:
        return int(args[2])
    return -100


def _reduction(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    # Apply the compatibility rule only to the modern default mean reduction
    # used by FieldScope. Legacy size_average/reduce overrides remain untouched.
    positional_size_average = args[1] if len(args) >= 2 else None
    positional_reduce = args[3] if len(args) >= 4 else None
    if (
        kwargs.get("size_average") is not None
        or kwargs.get("reduce") is not None
        or positional_size_average is not None
        or positional_reduce is not None
    ):
        return "legacy_override"
    if "reduction" in kwargs:
        return str(kwargs["reduction"])
    if len(args) >= 5:
        return str(args[4])
    return "mean"


def _finite_all_ignore_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    ignore_index = _ignore_index(args, kwargs)
    if (
        ignore_index == 255
        and _reduction(args, kwargs) == "mean"
        and target.numel() > 0
        and not bool(torch.any(target != ignore_index).item())
    ):
        # No label contributes to the registered pixel loss. Preserve the
        # optimizer-step and sample-exposure ledger while producing exactly
        # zero gradient. If logits are non-finite, ``sum() * 0`` stays
        # non-finite and the fixed-revision finite-value gate still rejects it.
        return input.sum() * 0.0
    return _original_cross_entropy(input, target, *args, **kwargs)


torch.load = _rng_compatible_torch_load
functional.cross_entropy = _finite_all_ignore_cross_entropy
