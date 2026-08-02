import json
from pathlib import Path

import pytest

from fieldscope.cli import doctor, main, run_smoke
from fieldscope.config import RunConfig


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "backend": {"name": "toy", "image_size": 32},
            "probe": {
                "times": [0.5],
                "num_directions": 2,
                "graph_grid": [4, 4],
                "topk": 3,
                "probe_batch_size": 4,
                "antithetic_noise": False,
                "seed": 3,
            },
            "tokenizer": {
                "hidden_dim": 32,
                "num_layers": 1,
                "num_classes": 2,
                "segmentation_classes": 3,
            },
            "runtime": {
                "output_dir": str(tmp_path / "smoke"),
                "batch_size": 1,
                "num_workers": 0,
            },
        }
    )


def test_doctor_and_smoke(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert doctor(config)["ready"] is True
    report = run_smoke(config, steps=1)
    assert report["status"] == "passed"
    assert report["cache_reload_equal"] is True
    assert Path(report["cache"]).is_file()


def test_dataset_audit_cli_returns_nonzero_on_failed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fieldscope.cli.audit_dataset_splits",
        lambda **_kwargs: {"status": "failed", "problems": ["overlap"]},
    )
    output = tmp_path / "audit.json"
    exit_code = main(
        [
            "audit-dataset-splits",
            "--dataset",
            "cifar10",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_supervised_error_analysis_cli_binds_all_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_replay(**kwargs):
        captured.update(kwargs)
        return {"status": "passed", "changes_main_verdict": False}

    monkeypatch.setattr("fieldscope.cli.replay_readout_errors", fake_replay)
    output = tmp_path / "analysis.json"
    arguments = [
        "analyze-readout-errors",
        "--checkpoint",
        str(tmp_path / "best.pt"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--matrix-report",
        str(tmp_path / "matrix.json"),
        "--training-report",
        str(tmp_path / "training.json"),
        "--test-report",
        str(tmp_path / "test.json"),
        "--output",
        str(output),
        "--batch-size",
        "4",
    ]
    assert main(arguments) == 0
    assert captured["checkpoint"] == tmp_path / "best.pt"
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["matrix_report_path"] == tmp_path / "matrix.json"
    assert captured["training_report_path"] == tmp_path / "training.json"
    assert captured["test_report_path"] == tmp_path / "test.json"
    assert captured["batch_size"] == 4
    assert captured["command"] == ["fieldscope", *arguments]
    assert json.loads(output.read_text(encoding="utf-8"))["changes_main_verdict"] is False


def test_supervised_error_summary_cli_binds_all_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_summary(report_paths, *, command):
        captured["report_paths"] = report_paths
        captured["command"] = command
        return {"status": "passed", "changes_main_verdict": False}

    monkeypatch.setattr("fieldscope.cli.summarize_readout_errors", fake_summary)
    output = tmp_path / "summary.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    arguments = [
        "summarize-readout-errors",
        "--report",
        str(first),
        "--report",
        str(second),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert captured["report_paths"] == [first, second]
    assert captured["command"] == ["fieldscope", *arguments]
    assert json.loads(output.read_text(encoding="utf-8"))["changes_main_verdict"] is False
