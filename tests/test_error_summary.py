import json
from pathlib import Path

import pytest

import fieldscope.error_summary as error_summary
from fieldscope.dataset_audit import sample_ids_sha256
from fieldscope.error_summary import (
    REGISTERED_REPRESENTATIONS,
    REGISTERED_SEEDS,
    summarize_readout_errors,
)


def _records(task: str, representation: str) -> list[dict[str, object]]:
    candidate = representation in {"response", "full"}
    records = []
    for index in range(8):
        record: dict[str, object] = {"sample_id": f"sample-{index}"}
        if task == "classification":
            record.update(
                {
                    "target_class": index % 2,
                    "top1_correct": int(candidate),
                }
            )
        elif task == "segmentation":
            record.update(
                {
                    "mean_iou": 0.2 + (0.05 * index if candidate else 0.0),
                    "boundary_density": index / 10,
                    "semantic_class_count": (index % 4) + 1,
                    "dominant_class_fraction": (8 - index) / 10,
                }
            )
        else:
            record.update(
                {
                    "abs_rel": 0.4 - (0.02 * index if candidate else 0.0),
                    "depth_discontinuity_density": index / 10,
                    "valid_fraction": 0.5 + index / 20,
                    "depth_p05_p95_range": 1.0 + index,
                }
            )
        records.append(record)
    return records


def _write_matrix(tmp_path: Path, task: str) -> list[Path]:
    sample_ids = [f"sample-{index}" for index in range(8)]
    digest = sample_ids_sha256(sample_ids)
    paths = []
    for representation in REGISTERED_REPRESENTATIONS:
        for seed in REGISTERED_SEEDS:
            path = tmp_path / f"{task}-{representation}-seed{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "passed",
                        "evidence_scope": (
                            "prospective_secondary_supervised_error_analysis"
                        ),
                        "changes_main_verdict": False,
                        "checkpoint": {
                            "path": f"/formal/{task}/{representation}/seed{seed}/best.pt",
                            "sha256": f"checkpoint-{representation}-{seed}",
                            "code_revision": "formal-revision",
                            "code_tree_sha256": "formal-tree",
                            "task": task,
                            "representation": representation,
                            "seed": seed,
                        },
                        "source_files": {
                            kind: {"path": f"/{kind}.json", "sha256": f"{kind}-sha"}
                            for kind in (
                                "matrix_report",
                                "training_report",
                                "test_report",
                            )
                        },
                        "test_cache": {
                            "path": f"/cache/{task}/test",
                            "manifest_sha256": "manifest-sha",
                            "num_samples": 8,
                            "sample_ids_sha256": digest,
                            "dataset": f"synthetic-{task}",
                            "split": "test",
                        },
                        "test_control_contract": {"status": "passed"},
                        "analyzer": {
                            "code_revision": "replay-revision",
                            "code_tree_sha256": "replay-tree",
                            "code_dirty": False,
                            "replay_core_compatibility": {"status": "passed"},
                        },
                        "num_samples": 8,
                        "sample_ids_sha256": digest,
                        "per_sample": _records(task, representation),
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


@pytest.fixture(autouse=True)
def _clean_analyzer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        error_summary,
        "code_provenance",
        lambda: {
            "code_revision": "summary-revision",
            "code_tree_sha256": "summary-tree",
            "code_dirty": False,
        },
    )


def test_summarize_complete_classification_matrix(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path, "classification")
    report = summarize_readout_errors(paths, command=["fieldscope", "summarize"])
    comparison = report["comparisons"]["response-vs-response_shuffled"]

    assert report["status"] == "passed"
    assert report["changes_main_verdict"] is False
    assert len(report["input_reports"]) == 33
    assert len(report["comparisons"]) == 12
    assert comparison["seeds"]["4121"]["summary"]["mean_delta"] == 1.0
    assert comparison["cross_seed"]["summary"]["mean_delta"] == 1.0
    assert comparison["cross_seed"]["summary"]["paired_sample_bootstrap"][
        "resamples"
    ] == 2000
    fixed = comparison["cross_seed"]["fixed_analysis"]
    assert sorted(fixed["per_class"]) == ["0", "1"]
    assert fixed["macro_class_distribution"]["macro_mean_delta"] == 1.0


def test_summarize_requires_every_registered_report(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path, "classification")
    with pytest.raises(ValueError, match="Expected exactly 33"):
        summarize_readout_errors(paths[:-1])


def test_summarize_rejects_target_descriptor_drift(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path, "segmentation")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["per_sample"][0]["boundary_density"] = 0.123
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Target descriptors differ"):
        summarize_readout_errors(paths)


def test_summarize_rejects_misaligned_sample_order(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path, "classification")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["per_sample"].reverse()
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Sample IDs are not exactly aligned"):
        summarize_readout_errors(paths)


def test_segmentation_fixed_strata_and_correlations(tmp_path: Path) -> None:
    report = summarize_readout_errors(_write_matrix(tmp_path, "segmentation"))
    fixed = report["comparisons"]["response-vs-state"]["cross_seed"][
        "fixed_analysis"
    ]

    boundary = fixed["quartiles"]["boundary_density"]
    assert [
        boundary["strata"][f"Q{index}"]["summary"]["num_samples"]
        for index in range(1, 5)
    ] == [2, 2, 2, 2]
    assert set(fixed["semantic_class_count_groups"]) == {"1", "2", "3+"}
    assert fixed["pearson_correlations"]["boundary_density"]["coefficient"] == (
        pytest.approx(1.0)
    )


def test_depth_fixed_strata_use_abs_rel_improvement(tmp_path: Path) -> None:
    report = summarize_readout_errors(_write_matrix(tmp_path, "depth"))
    comparison = report["comparisons"]["full-vs-full_shuffled"]
    fixed = comparison["cross_seed"]["fixed_analysis"]

    assert comparison["delta_definition"] == "full_shuffled_abs_rel - full_abs_rel"
    assert comparison["cross_seed"]["summary"]["mean_delta"] == pytest.approx(0.07)
    assert set(fixed["quartiles"]) == {
        "depth_discontinuity_density",
        "valid_fraction",
        "depth_p05_p95_range",
    }
    assert fixed["pearson_correlations"]["depth_p05_p95_range"][
        "coefficient"
    ] == pytest.approx(1.0)
