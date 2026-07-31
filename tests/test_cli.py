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
