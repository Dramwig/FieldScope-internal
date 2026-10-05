from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ops"
    / "fieldscope_cross_host_monitor_state.py"
)
SPEC = importlib.util.spec_from_file_location("cross_host_monitor_state", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REVISION = "020c1de567edd88e0eda245fd085335ffe678f47"
NOW = dt.datetime(2026, 8, 24, 10, 30, tzinfo=dt.timezone.utc)


def valid_payload() -> dict:
    return {
        "status": "active",
        "expected_revision": REVISION,
        "watchdog_pid": 41096,
        "observed_at": "2026-08-24T10:21:44+00:00",
        "recovery": {"alive": True, "pid": 41610},
    }


def test_accepts_current_live_fixed_revision_heartbeat() -> None:
    assert MODULE.validate_watchdog_state(
        valid_payload(),
        expected_revision=REVISION,
        max_age_seconds=1200,
        now=NOW,
    ) == []


def test_rejects_stale_or_dead_recovery_heartbeat() -> None:
    payload = valid_payload()
    payload["observed_at"] = "2026-08-24T09:00:00Z"
    payload["recovery"] = {"alive": False, "pid": 41610}
    assert MODULE.validate_watchdog_state(
        payload,
        expected_revision=REVISION,
        max_age_seconds=1200,
        now=NOW,
    ) == ["recovery_not_alive", "heartbeat_stale"]


def test_rejects_wrong_revision_and_naive_time() -> None:
    payload = valid_payload()
    payload["expected_revision"] = "wrong"
    payload["observed_at"] = "2026-08-24T10:21:44"
    assert MODULE.validate_watchdog_state(
        payload,
        expected_revision=REVISION,
        max_age_seconds=1200,
        now=NOW,
    ) == ["revision_mismatch", "observed_at_naive"]
