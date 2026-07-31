import json
from pathlib import Path

import pytest

from fieldscope.resource_planning import measured_bytes_per_sample, plan_cache_budget


def _measurement_cache(tmp_path: Path, *, policy: str = "readout_sparse") -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "shard-00000.pt").write_bytes(b"a" * 4000)
    (cache / "shard-00001.pt").write_bytes(b"b" * 6000)
    (cache / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "storage_policy": policy,
                "num_samples": 10,
                "shards": [
                    {
                        "path": "shard-00000.pt",
                        "num_samples": 4,
                        "bytes": 4000,
                    },
                    {
                        "path": "shard-00001.pt",
                        "num_samples": 6,
                        "bytes": 6000,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return cache


def test_cache_budget_uses_measured_bytes_and_safety_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _measurement_cache(tmp_path)
    monkeypatch.setattr(
        "fieldscope.resource_planning.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 100_000})(),
    )
    report = plan_cache_budget(
        measurement_cache=cache,
        target_samples={"train": 50, "val": 10},
        filesystem_path=tmp_path,
        reserve_bytes=10_000,
        safety_factor=1.2,
    )
    assert measured_bytes_per_sample(cache) == 1000.0
    assert report["projected_split_bytes"] == {"train": 60_000, "val": 12_000}
    assert report["required_bytes"] == 82_000
    assert report["additional_required_bytes"] == 0
    assert report["fits"] is True
    assert report["status"] == "passed"
    assert report["headroom_bytes"] == 18_000


def test_cache_budget_blocks_without_shrinking_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _measurement_cache(tmp_path)
    monkeypatch.setattr(
        "fieldscope.resource_planning.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 50_000})(),
    )
    report = plan_cache_budget(
        measurement_cache=cache,
        target_samples={"train": 50, "val": 10},
        filesystem_path=tmp_path,
        reserve_bytes=10_000,
        safety_factor=1.2,
    )
    assert report["fits"] is False
    assert report["status"] == "resource_blocked"
    assert report["target_samples"] == {"train": 50, "val": 10}
    assert report["headroom_bytes"] < 0


def test_cache_budget_accepts_only_the_requested_storage_policy(tmp_path: Path) -> None:
    dense = _measurement_cache(tmp_path, policy="dense")
    with pytest.raises(ValueError, match="readout_sparse"):
        measured_bytes_per_sample(dense)
    assert measured_bytes_per_sample(
        dense,
        required_storage_policy="dense",
    ) == 1000.0
    manifest = json.loads((dense / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["storage_policy"] = "readout_sparse"
    manifest["complete"] = False
    (dense / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        measured_bytes_per_sample(dense)


def test_cache_budget_rejects_changed_measurement_shard(tmp_path: Path) -> None:
    cache = _measurement_cache(tmp_path)
    (cache / "shard-00000.pt").write_bytes(b"short")
    with pytest.raises(ValueError, match="changed size"):
        measured_bytes_per_sample(cache)


def test_cache_budget_includes_other_planned_cache_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _measurement_cache(tmp_path)
    monkeypatch.setattr(
        "fieldscope.resource_planning.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 100_000})(),
    )
    report = plan_cache_budget(
        measurement_cache=cache,
        target_samples={"train": 50},
        filesystem_path=tmp_path,
        reserve_bytes=10_000,
        safety_factor=1.0,
        additional_required_bytes=45_000,
    )
    assert report["required_bytes"] == 105_000
    assert report["fits"] is False


def test_cache_budget_adds_split_specific_target_payload_before_margin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _measurement_cache(tmp_path)
    monkeypatch.setattr(
        "fieldscope.resource_planning.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": 1_000_000})(),
    )
    report = plan_cache_budget(
        measurement_cache=cache,
        target_samples={"classification": 10, "segmentation": 5},
        filesystem_path=tmp_path,
        reserve_bytes=0,
        safety_factor=1.2,
        additional_bytes_per_sample={"segmentation": 256},
    )
    assert report["projected_split_bytes"] == {
        "classification": 12_000,
        "segmentation": 7_536,
    }
    assert report["additional_bytes_per_sample"] == {"segmentation": 256}


def test_cache_budget_rejects_unknown_or_negative_split_overhead(tmp_path: Path) -> None:
    cache = _measurement_cache(tmp_path)
    with pytest.raises(ValueError, match="additional_bytes_per_sample"):
        plan_cache_budget(
            measurement_cache=cache,
            target_samples={"train": 10},
            filesystem_path=tmp_path,
            additional_bytes_per_sample={"unknown": 1},
        )
    with pytest.raises(ValueError, match="additional_bytes_per_sample"):
        plan_cache_budget(
            measurement_cache=cache,
            target_samples={"train": 10},
            filesystem_path=tmp_path,
            additional_bytes_per_sample={"train": -1},
        )
