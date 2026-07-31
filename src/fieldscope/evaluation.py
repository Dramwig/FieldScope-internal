"""Dependency-light task and unsupervised-structure evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F

from fieldscope.graph import spectral_binary_partition


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    targets = targets.long()
    topk = min(5, logits.shape[1])
    predictions = logits.topk(topk, dim=1).indices
    matches = predictions.eq(targets[:, None])
    return {
        "top1": float(matches[:, :1].any(dim=1).float().mean().item()),
        "top5": float(matches.any(dim=1).float().mean().item()),
    }


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
    predicted = prediction[valid].clamp_min(1e-6)
    expected = target[valid].clamp_min(1e-6)
    difference = predicted - expected
    ratio = torch.maximum(predicted / expected, expected / predicted)
    return {
        "mae": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "abs_rel": float((difference.abs() / expected).mean().item()),
        "delta1": float((ratio < 1.25).float().mean().item()),
        "delta2": float((ratio < 1.25**2).float().mean().item()),
        "delta3": float((ratio < 1.25**3).float().mean().item()),
    }


def normal_angular_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.nn.functional.normalize(prediction, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    cosine = (prediction * target).sum(dim=1).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cosine)).mean().item())


@dataclass
class ClassificationMeter:
    correct1: int = 0
    correct5: int = 0
    count: int = 0
    class_correct1: dict[int, int] = field(default_factory=dict)
    class_count: dict[int, int] = field(default_factory=dict)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        targets = targets.long()
        predictions = logits.topk(min(5, logits.shape[1]), dim=1).indices
        matches = predictions.eq(targets[:, None])
        self.correct1 += int(matches[:, :1].any(dim=1).sum().item())
        self.correct5 += int(matches.any(dim=1).sum().item())
        self.count += targets.numel()
        correct = matches[:, :1].any(dim=1)
        for label in targets.unique().tolist():
            class_mask = targets == label
            self.class_correct1[label] = self.class_correct1.get(label, 0) + int(
                correct[class_mask].sum().item()
            )
            self.class_count[label] = self.class_count.get(label, 0) + int(
                class_mask.sum().item()
            )

    def compute(self) -> dict[str, Any]:
        denominator = max(1, self.count)
        per_class = {
            str(label): self.class_correct1.get(label, 0) / count
            for label, count in sorted(self.class_count.items())
            if count > 0
        }
        return {
            "top1": self.correct1 / denominator,
            "top5": self.correct5 / denominator,
            "macro_top1": (
                sum(per_class.values()) / len(per_class)
                if per_class
                else float("nan")
            ),
            "minimum_class_top1": min(per_class.values(), default=float("nan")),
            "median_class_top1": (
                median(per_class.values())
                if per_class
                else float("nan")
            ),
            "maximum_class_top1": max(per_class.values(), default=float("nan")),
            "per_class_top1": per_class,
        }


@dataclass
class SegmentationMeter:
    num_classes: int
    ignore_index: int = 255
    confusion: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1).detach().cpu()
        targets = targets.detach().cpu().long()
        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
        )
        encoded = targets[valid] * self.num_classes + predictions[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        matrix = self.confusion.float()
        intersection = matrix.diag()
        union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
        valid_classes = union > 0
        mean_iou = (
            (intersection[valid_classes] / union[valid_classes]).mean()
            if valid_classes.any()
            else torch.tensor(float("nan"))
        )
        return {
            "mean_iou": float(mean_iou.item()),
            "pixel_accuracy": float(
                (intersection.sum() / matrix.sum().clamp_min(1)).item()
            ),
        }


@dataclass
class DepthMeter:
    absolute_error: float = 0.0
    squared_error: float = 0.0
    absolute_relative_error: float = 0.0
    delta1: int = 0
    delta2: int = 0
    delta3: int = 0
    count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        valid = torch.isfinite(target) & (target > 0)
        predicted = prediction[valid].detach().float().clamp_min(1e-6)
        expected = target[valid].detach().float().clamp_min(1e-6)
        difference = predicted - expected
        ratio = torch.maximum(predicted / expected, expected / predicted)
        self.absolute_error += float(difference.abs().sum().item())
        self.squared_error += float(difference.square().sum().item())
        self.absolute_relative_error += float(
            (difference.abs() / expected).sum().item()
        )
        self.delta1 += int((ratio < 1.25).sum().item())
        self.delta2 += int((ratio < 1.25**2).sum().item())
        self.delta3 += int((ratio < 1.25**3).sum().item())
        self.count += expected.numel()

    def compute(self) -> dict[str, float]:
        count = max(1, self.count)
        return {
            "mae": self.absolute_error / count,
            "rmse": (self.squared_error / count) ** 0.5,
            "abs_rel": self.absolute_relative_error / count,
            "delta1": self.delta1 / count,
            "delta2": self.delta2 / count,
            "delta3": self.delta3 / count,
        }


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute tie-aware AUROC from one-dimensional scores and binary labels."""

    scores = scores.detach().float().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    valid = torch.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    positives = int(labels.sum().item())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ranks = torch.empty_like(sorted_scores)
    start = 0
    for count in counts.tolist():
        end = start + count
        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end
    positive_rank_sum = ranks[sorted_labels].sum()
    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return float(auc.item())


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    valid = torch.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    positives = int(labels.sum().item())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    precision = sorted_labels.cumsum(0) / torch.arange(
        1, sorted_labels.numel() + 1, dtype=torch.float32
    )
    return float((precision * sorted_labels).sum().item() / positives)


def best_binary_f1(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    valid = torch.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    positives = int(labels.sum().item())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    true_positive = sorted_labels.cumsum(0)
    predicted_positive = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    precision = true_positive / predicted_positive
    recall = true_positive / positives
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return float(f1.max().item())


def graph_segmentation_metrics(
    affinity: torch.Tensor,
    targets: torch.Tensor,
    grid_size: tuple[int, int],
    *,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Compare an affinity graph with segmentation labels without training a head."""

    if affinity.ndim != 3:
        raise ValueError("affinity must have shape [B,P,P]")
    labels = F.interpolate(
        targets[:, None].float(),
        size=grid_size,
        mode="nearest",
    ).squeeze(1).long()
    partitions = spectral_binary_partition(affinity.clamp_min(0))
    pairwise_auc: list[float] = []
    boundary_auc: list[float] = []
    boundary_ap: list[float] = []
    boundary_f1: list[float] = []
    foreground_iou: list[float] = []
    height, width = grid_size
    patch_index = torch.arange(height * width, device=affinity.device).reshape(
        height, width
    )
    edge_first = torch.cat(
        [patch_index[:, :-1].flatten(), patch_index[:-1, :].flatten()]
    )
    edge_second = torch.cat(
        [patch_index[:, 1:].flatten(), patch_index[1:, :].flatten()]
    )
    upper = torch.triu(
        torch.ones(
            (height * width, height * width),
            dtype=torch.bool,
            device=affinity.device,
        ),
        diagonal=1,
    )
    for batch_index in range(affinity.shape[0]):
        flat_labels = labels[batch_index].flatten()
        valid = flat_labels != ignore_index
        pair_valid = valid[:, None] & valid[None, :] & upper
        same_region = flat_labels[:, None] == flat_labels[None, :]
        pairwise_auc.append(
            binary_auroc(
                affinity[batch_index][pair_valid],
                same_region[pair_valid],
            )
        )
        local_valid = valid[edge_first] & valid[edge_second]
        local_boundary = (
            flat_labels[edge_first] != flat_labels[edge_second]
        )[local_valid]
        local_score = (
            1.0
            - affinity[batch_index, edge_first, edge_second].float()
        )[local_valid]
        boundary_auc.append(binary_auroc(local_score, local_boundary))
        boundary_ap.append(average_precision(local_score, local_boundary))
        boundary_f1.append(best_binary_f1(local_score, local_boundary))

        foreground = (flat_labels > 0) & valid
        partition = partitions[batch_index].bool()
        ious: list[torch.Tensor] = []
        for candidate in (partition, ~partition):
            intersection = (candidate & foreground).sum().float()
            union = ((candidate | foreground) & valid).sum().float()
            ious.append(intersection / union.clamp_min(1))
        foreground_iou.append(float(torch.stack(ious).max().item()))

    def finite_mean(values: list[float]) -> float:
        tensor = torch.tensor(values, dtype=torch.float32)
        finite = torch.isfinite(tensor)
        return float(tensor[finite].mean().item()) if finite.any() else float("nan")

    return {
        "pairwise_auroc": finite_mean(pairwise_auc),
        "boundary_auroc": finite_mean(boundary_auc),
        "boundary_average_precision": finite_mean(boundary_ap),
        "boundary_best_f1": finite_mean(boundary_f1),
        "foreground_spectral_iou": finite_mean(foreground_iou),
    }
