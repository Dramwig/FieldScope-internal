import pytest
import torch

from fieldscope.evaluation import (
    ClassificationMeter,
    DepthMeter,
    SegmentationMeter,
    graph_segmentation_metrics,
)


def test_task_meters_report_perfect_predictions() -> None:
    classification = ClassificationMeter()
    classification.update(
        torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        torch.tensor([0, 1]),
    )
    assert classification.compute() == {"top1": 1.0, "top5": 1.0}

    segmentation = SegmentationMeter(num_classes=2)
    segmentation.update(
        torch.tensor(
            [
                [
                    [[5.0, 5.0], [0.0, 0.0]],
                    [[0.0, 0.0], [5.0, 5.0]],
                ]
            ]
        ),
        torch.tensor([[[0, 0], [1, 1]]]),
    )
    assert segmentation.compute() == {"mean_iou": 1.0, "pixel_accuracy": 1.0}

    depth = DepthMeter()
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    depth.update(target.clone(), target)
    metrics = depth.compute()
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["abs_rel"] == 0.0
    assert metrics["delta1"] == 1.0


def test_graph_segmentation_metrics_detect_perfect_region_graph() -> None:
    labels = torch.tensor([[[0, 0], [1, 1]]])
    flat = labels.flatten()
    affinity = (flat[:, None] == flat[None, :]).float().unsqueeze(0)
    metrics = graph_segmentation_metrics(affinity, labels, (2, 2))
    assert metrics["pairwise_auroc"] == pytest.approx(1.0)
    assert metrics["boundary_auroc"] == pytest.approx(1.0)
    assert metrics["boundary_average_precision"] == pytest.approx(1.0)
    assert metrics["boundary_best_f1"] == pytest.approx(1.0)
    assert metrics["foreground_spectral_iou"] == pytest.approx(1.0)
