"""Small-sample summaries for multi-seed FieldScope experiments."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

_T_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    13: 2.1604,
    14: 2.1448,
    15: 2.1314,
    16: 2.1199,
    17: 2.1098,
    18: 2.1009,
    19: 2.0930,
    20: 2.0860,
    21: 2.0796,
    22: 2.0739,
    23: 2.0687,
    24: 2.0639,
    25: 2.0595,
    26: 2.0555,
    27: 2.0518,
    28: 2.0484,
    29: 2.0452,
    30: 2.0423,
}


def t_interval_summary(values: list[float]) -> dict[str, Any]:
    """Report mean, sample SD, and a two-sided 95% Student-t interval."""

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("values must be a non-empty finite list")
    count = len(values)
    average = mean(values)
    if count == 1:
        return {
            "count": 1,
            "mean": average,
            "sample_std": None,
            "ci95": None,
            "minimum": values[0],
            "median": values[0],
            "maximum": values[0],
        }
    sample_std = stdev(values)
    critical = _T_975.get(count - 1, 1.96)
    half_width = critical * sample_std / count**0.5
    return {
        "count": count,
        "mean": average,
        "sample_std": sample_std,
        "ci95": [average - half_width, average + half_width],
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def paired_t_interval(
    candidate: Mapping[int, float],
    reference: Mapping[int, float],
) -> dict[str, Any]:
    """Summarize paired seed differences and whether their 95% CI excludes zero."""

    seeds = sorted(set(candidate) & set(reference))
    if not seeds:
        raise ValueError("candidate and reference have no shared seeds")
    differences = [candidate[seed] - reference[seed] for seed in seeds]
    summary = t_interval_summary(differences)
    interval = summary["ci95"]
    observed = abs(mean(differences))
    permutations = 1 << len(differences)
    as_extreme = 0
    for mask in range(permutations):
        permuted = [
            value if mask & (1 << index) else -value
            for index, value in enumerate(differences)
        ]
        if abs(mean(permuted)) >= observed - 1e-15:
            as_extreme += 1
    return {
        "seeds": seeds,
        "differences": differences,
        **summary,
        "paired_sign_flip_pvalue": as_extreme / permutations,
        "ci95_excludes_zero": bool(
            interval is not None and (interval[0] > 0 or interval[1] < 0)
        ),
    }


def holm_adjusted_pvalues(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm family-wise adjusted p-values."""

    if any(not 0.0 <= value <= 1.0 for value in pvalues.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int = 4121,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Return a deterministic percentile bootstrap interval for the mean."""

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("values must be a non-empty finite list")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if len(values) == 1:
        return {"mean": values[0], "ci95": None, "resamples": resamples}
    generator = random.Random(seed)
    sample_count = len(values)
    estimates = sorted(
        mean(generator.choices(values, k=sample_count))
        for _ in range(resamples)
    )
    lower = estimates[round(0.025 * (resamples - 1))]
    upper = estimates[round(0.975 * (resamples - 1))]
    return {
        "mean": mean(values),
        "ci95": [lower, upper],
        "resamples": resamples,
    }


def _report_metric(report: Mapping[str, Any], metric: str) -> float:
    if "evaluation" in report:
        return float(report["evaluation"]["metrics"][metric])
    best_epoch = int(report["best_epoch"])
    history = report["history"]
    return float(history[best_epoch - 1]["validation"]["metrics"][metric])


def summarize_run_reports(
    paths: list[Path],
    *,
    metric: str,
    reference: str | None = None,
) -> dict[str, Any]:
    """Aggregate readout reports by task/representation and pair by seed."""

    observations: dict[tuple[str, str], dict[int, float]] = {}
    sources: list[str] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (str(report["task"]), str(report["representation"]))
        seed = int(report["seed"])
        if seed in observations.setdefault(key, {}):
            raise ValueError(f"Duplicate task/representation/seed in {path}")
        observations[key][seed] = _report_metric(report, metric)
        sources.append(str(path))
    groups = {
        f"{task}/{representation}": {
            "task": task,
            "representation": representation,
            "seeds": seed_values,
            "summary": t_interval_summary(list(seed_values.values())),
        }
        for (task, representation), seed_values in sorted(observations.items())
    }
    comparisons: dict[str, Any] = {}
    if reference is not None:
        for (task, representation), seed_values in observations.items():
            if representation == reference:
                continue
            reference_values = observations.get((task, reference))
            if reference_values is not None:
                comparisons[f"{task}/{representation}-minus-{reference}"] = (
                    paired_t_interval(seed_values, reference_values)
                )
        adjusted = holm_adjusted_pvalues(
            {
                name: comparison["paired_sign_flip_pvalue"]
                for name, comparison in comparisons.items()
            }
        )
        for name, value in adjusted.items():
            comparisons[name]["holm_adjusted_pvalue"] = value
            comparisons[name]["holm_reject_alpha_0_05"] = value < 0.05
    return {
        "status": "passed",
        "metric": metric,
        "groups": groups,
        "comparisons": comparisons,
        "sources": sources,
    }
