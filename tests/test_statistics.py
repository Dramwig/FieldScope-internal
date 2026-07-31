import json
from pathlib import Path

import pytest

from fieldscope.statistics import (
    bootstrap_mean_interval,
    paired_t_interval,
    summarize_run_reports,
    t_interval_summary,
)


def test_t_interval_and_paired_difference() -> None:
    summary = t_interval_summary([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["sample_std"] == 1.0
    assert summary["ci95"] == pytest.approx([-0.4842, 4.4842], abs=1e-4)

    paired = paired_t_interval(
        {1: 3.0, 2: 4.0, 3: 5.0},
        {1: 1.0, 2: 2.0, 3: 3.0},
    )
    assert paired["mean"] == 2.0
    assert paired["ci95"] == [2.0, 2.0]
    assert paired["ci95_excludes_zero"] is True
    bootstrap = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=5, resamples=200)
    assert bootstrap["mean"] == 2.0
    assert bootstrap["ci95"][0] <= 2.0 <= bootstrap["ci95"][1]


def test_summarize_run_reports_pairs_representations(tmp_path: Path) -> None:
    paths: list[Path] = []
    for representation, values in {"full": [0.7, 0.8, 0.9], "state": [0.6, 0.7, 0.8]}.items():
        for seed, value in enumerate(values, start=1):
            path = tmp_path / f"{representation}-{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "task": "classification",
                        "representation": representation,
                        "seed": seed,
                        "best_epoch": 1,
                        "history": [
                            {"validation": {"metrics": {"top1": value}}}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
    report = summarize_run_reports(paths, metric="top1", reference="state")
    assert report["groups"]["classification/full"]["summary"]["mean"] == pytest.approx(
        0.8
    )
    comparison = report["comparisons"]["classification/full-minus-state"]
    assert comparison["mean"] == pytest.approx(0.1)
