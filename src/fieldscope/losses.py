"""Task, graph-stability, and representation-consistency losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def graph_stability_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("graph tensors must have identical shapes")
    return F.mse_loss(first, second)


def representation_consistency_loss(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("representation tensors must have identical shapes")
    return F.mse_loss(first, second.detach())


def multitask_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    losses: dict[str, torch.Tensor] = {}
    if "classification" in targets:
        losses["classification"] = F.cross_entropy(
            predictions["classification"], targets["classification"].long()
        )
    if "segmentation" in targets:
        logits = predictions["segmentation"].permute(0, 2, 3, 1)
        classes = logits.shape[-1]
        losses["segmentation"] = F.cross_entropy(
            logits.reshape(-1, classes),
            targets["segmentation"].long().reshape(-1),
            ignore_index=ignore_index,
        )
    if "depth" in targets:
        valid = torch.isfinite(targets["depth"]) & (targets["depth"] > 0)
        if valid.any():
            losses["depth"] = F.l1_loss(
                predictions["depth"][valid], targets["depth"][valid]
            )
    if "normals" in targets:
        target_normals = F.normalize(targets["normals"], dim=1, eps=1e-6)
        cosine = (predictions["normals"] * target_normals).sum(dim=1)
        losses["normals"] = (1.0 - cosine).mean()
    if not losses:
        raise ValueError("No supported targets were supplied")
    total = sum(weights.get(name, 1.0) * value for name, value in losses.items())
    return total, losses
