"""Small, dependency-light evaluation utilities."""

from __future__ import annotations

import torch


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def segmentation_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int = 255,
) -> dict[str, float]:
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    ious: list[torch.Tensor] = []
    for label in range(num_classes):
        predicted = (predictions == label) & valid
        expected = (targets == label) & valid
        union = (predicted | expected).sum()
        if union:
            ious.append((predicted & expected).sum().float() / union.float())
    mean_iou = torch.stack(ious).mean().item() if ious else float("nan")
    pixel_accuracy = ((predictions == targets) & valid).sum().float() / valid.sum().clamp_min(1)
    return {"mean_iou": float(mean_iou), "pixel_accuracy": float(pixel_accuracy.item())}


def depth_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    valid = torch.isfinite(target) & (target > 0)
    difference = prediction[valid] - target[valid]
    return {
        "mae": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
    }


def normal_angular_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.nn.functional.normalize(prediction, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    cosine = (prediction * target).sum(dim=1).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cosine)).mean().item())

