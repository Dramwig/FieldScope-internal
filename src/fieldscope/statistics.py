"""Small-sample summaries for multi-seed FieldScope experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, stdev
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
        }
    sample_std = stdev(values)
    critical = _T_975.get(count - 1, 1.96)
    half_width = critical * sample_std / count**0.5
    return {
        "count": count,
        "mean": average,
        "sample_std": sample_std,
        "ci95": [average - half_width, average + half_width],
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
    return {
        "seeds": seeds,
        "differences": differences,
        **summary,
        "ci95_excludes_zero": bool(
            interval is not None and (interval[0] > 0 or interval[1] < 0)
        ),
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
    return {
        "status": "passed",
        "metric": metric,
        "groups": groups,
        "comparisons": comparisons,
        "sources": sources,
    }
