import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_watcher():
    sys.modules.setdefault(
        "fcntl", types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None)
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ops"
        / "fieldscope_atomic_audit_watcher.py"
    )
    spec = importlib.util.spec_from_file_location("fieldscope_atomic_audit_watcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(epoch: int, *, status: str, problems: list[str]) -> dict:
    checks = {
        "history_contiguous": True,
        "report_epoch_matches": True,
        "report_steps_match": True,
        "report_exposures_match": True,
        "report_checkpoint_history_equal": True,
        "stable_checkpoint_during_hash_and_load": True,
        "stable_report_double_read": True,
    }
    for problem in problems:
        checks[problem] = False
    return {
        "status": status,
        "problems": problems,
        "checks": checks,
        "committed_epoch": epoch,
        "completed_optimizer_steps": epoch * 10,
        "completed_training_sample_exposures": epoch * 100,
        "optimizer_state_steps": [epoch * 10],
        "scheduler_last_epoch": epoch,
        "changes_scientific_verdict": False,
        "fixed_revision": "020c1de",
        "representation": "z0",
        "runtime_profile_seed_workers": 1,
        "seed": 7319,
        "target_epochs": 80,
        "task": "segmentation",
    }


def test_later_stable_audit_classifies_unstable_checkpoint_read(tmp_path: Path) -> None:
    watcher = _load_watcher()
    failed_path = tmp_path / "failed.json"
    later_path = tmp_path / "later.json"
    failed_path.write_text(
        json.dumps(
            _audit(
                66,
                status="failed",
                problems=["stable_checkpoint_during_hash_and_load"],
            )
        ),
        encoding="utf-8",
    )
    later_path.write_text(json.dumps(_audit(69, status="passed", problems=[])), encoding="utf-8")

    result = watcher.later_stable_audit_race(failed_path, 66, [later_path])

    assert result is not None
    assert result["confirmed_epoch"] == 69
    assert result["unresolved_scientific_failure"] is False
    assert result["failed_artifact_preserved"] is True


def test_later_audit_must_match_identity_and_have_all_checks_true(tmp_path: Path) -> None:
    watcher = _load_watcher()
    failed_path = tmp_path / "failed.json"
    later_path = tmp_path / "later.json"
    failed_path.write_text(
        json.dumps(_audit(68, status="failed", problems=["stable_report_double_read"])),
        encoding="utf-8",
    )
    later = _audit(70, status="passed", problems=[])
    later["fixed_revision"] = "wrong-revision"
    later_path.write_text(json.dumps(later), encoding="utf-8")

    assert watcher.later_stable_audit_race(failed_path, 68, [later_path]) is None

    later["fixed_revision"] = "020c1de"
    later["checks"]["stable_report_double_read"] = False
    later_path.write_text(json.dumps(later), encoding="utf-8")
    assert watcher.later_stable_audit_race(failed_path, 68, [later_path]) is None


def test_later_stable_audit_classifies_checkpoint_ahead_of_report(tmp_path: Path) -> None:
    watcher = _load_watcher()
    failed_path = tmp_path / "failed.json"
    later_path = tmp_path / "later.json"
    failed = _audit(
        55,
        status="failed",
        problems=[
            "checkpoint_epoch_matches",
            "checkpoint_steps_match",
            "checkpoint_exposures_match",
            "report_checkpoint_history_equal",
            "optimizer_state_steps_match",
            "scheduler_last_epoch_matches",
        ],
    )
    failed["scheduler_last_epoch"] = 56
    failed["optimizer_state_steps"] = [560]
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    later_path.write_text(json.dumps(_audit(58, status="passed", problems=[])), encoding="utf-8")

    result = watcher.later_stable_audit_race(failed_path, 55, [later_path])

    assert result is not None
    assert result["classification"] == (
        "checkpoint_ahead_of_stable_report_confirmed_by_later_strong_audit"
    )
    assert result["checkpoint_epoch"] == 56
    assert result["confirmed_epoch"] == 58
    assert result["failed_artifact_preserved"] is True


def test_checkpoint_ahead_requires_intact_report_and_checkpoint_boundary(tmp_path: Path) -> None:
    watcher = _load_watcher()
    failed_path = tmp_path / "failed.json"
    later_path = tmp_path / "later.json"
    failed = _audit(
        55,
        status="failed",
        problems=["report_steps_match", "report_checkpoint_history_equal"],
    )
    failed["scheduler_last_epoch"] = 58
    failed["optimizer_state_steps"] = [580]
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    later_path.write_text(json.dumps(_audit(57, status="passed", problems=[])), encoding="utf-8")

    assert watcher.later_stable_audit_race(failed_path, 55, [later_path]) is None
