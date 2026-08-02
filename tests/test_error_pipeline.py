import json
from pathlib import Path

import pytest

import fieldscope.error_pipeline as error_pipeline
from fieldscope.error_pipeline import (
    FORMAL_REPRESENTATIONS,
    _formal_runs,
    run_readout_error_analysis,
)
from fieldscope.error_summary import REGISTERED_SEEDS


def _matrix(source_root: Path) -> dict[str, object]:
    runs = []
    for representation in sorted(FORMAL_REPRESENTATIONS):
        for seed in REGISTERED_SEEDS:
            base = Path("outputs") / representation / f"seed-{seed}"
            runs.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "best_checkpoint": str(base / "best.pt"),
                    "training_report": str(base / "training.json"),
                    "test_report": str(base / "test.json"),
                }
            )
    return {
        "status": "passed",
        "task": "classification",
        "code_revision": "formal-revision",
        "code_tree_sha256": "formal-tree",
        "representations": sorted(FORMAL_REPRESENTATIONS),
        "seeds": list(REGISTERED_SEEDS),
        "test_cache_dir": str(source_root / "cache"),
        "runs": runs,
    }


def test_formal_runs_requires_exact_twenty_by_three_matrix(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path)
    assert len(_formal_runs(matrix)) == 60
    matrix["runs"] = matrix["runs"][:-1]
    with pytest.raises(ValueError, match="exactly 60"):
        _formal_runs(matrix)


def test_error_pipeline_dispatches_registered_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "formal"
    source_root.mkdir()
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_matrix(source_root)), encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        error_pipeline,
        "code_provenance",
        lambda: {
            "code_revision": "analyzer-revision",
            "code_tree_sha256": "analyzer-tree",
            "code_dirty": False,
        },
    )
    monkeypatch.setattr(
        error_pipeline,
        "audit_source_repository",
        lambda revision, *, repository_root: {
            "status": "passed",
            "path": str(repository_root),
            "source_revision": revision,
            "actual_revision": revision,
            "dirty_paths": [],
        },
    )
    monkeypatch.setattr(
        error_pipeline,
        "audit_replay_core_compatibility",
        lambda revision, **_kwargs: {
            "status": "passed",
            "source_revision": revision,
            "analyzer_revision": "analyzer-revision",
        },
    )

    def fake_replay(**kwargs):
        calls.append(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(error_pipeline, "replay_readout_errors", fake_replay)

    def fake_summary(paths, *, command):
        assert len(paths) == 33
        assert all(path.is_file() for path in paths)
        assert command == ["fieldscope", "pipeline"]
        return {"status": "passed", "changes_main_verdict": False}

    monkeypatch.setattr(error_pipeline, "summarize_readout_errors", fake_summary)
    output_dir = tmp_path / "analysis"
    report = run_readout_error_analysis(
        matrix_report_path=matrix_path,
        source_repository_root=source_root,
        output_dir=output_dir,
        batch_size=7,
        command=["fieldscope", "pipeline"],
    )

    assert len(calls) == 33
    assert report["pipeline"]["generated_count"] == 33
    assert report["pipeline"]["reused_count"] == 0
    assert Path(calls[0]["source_repository_root"]) == source_root
    assert calls[0]["batch_size"] == 7
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))[
        "status"
    ] == "passed"


def test_error_pipeline_reuses_only_audited_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "formal"
    source_root.mkdir()
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_matrix(source_root)), encoding="utf-8")
    monkeypatch.setattr(
        error_pipeline,
        "code_provenance",
        lambda: {
            "code_revision": "analyzer-revision",
            "code_tree_sha256": "analyzer-tree",
            "code_dirty": False,
        },
    )
    monkeypatch.setattr(
        error_pipeline,
        "audit_source_repository",
        lambda revision, *, repository_root: {"status": "passed"},
    )
    monkeypatch.setattr(
        error_pipeline,
        "audit_replay_core_compatibility",
        lambda revision, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(error_pipeline, "_reusable_report", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        error_pipeline,
        "replay_readout_errors",
        lambda **_kwargs: pytest.fail("reusable report was unexpectedly replayed"),
    )
    monkeypatch.setattr(
        error_pipeline,
        "summarize_readout_errors",
        lambda _paths, *, command: {"status": "passed"},
    )

    report = run_readout_error_analysis(
        matrix_report_path=matrix_path,
        source_repository_root=source_root,
        output_dir=tmp_path / "analysis",
    )
    assert report["pipeline"]["generated_count"] == 0
    assert report["pipeline"]["reused_count"] == 33
