#!/usr/bin/env python3
"""Validate the shared watchdog heartbeat used by cross-host workers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def validate_watchdog_state(
    payload: Any,
    *,
    expected_revision: str,
    max_age_seconds: int,
    now: dt.datetime | None = None,
) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["state_not_object"]
    if payload.get("status") != "active":
        problems.append("status_not_active")
    if payload.get("expected_revision") != expected_revision:
        problems.append("revision_mismatch")
    recovery = payload.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("alive") is not True:
        problems.append("recovery_not_alive")
    watchdog_pid = payload.get("watchdog_pid")
    if not isinstance(watchdog_pid, int) or isinstance(watchdog_pid, bool):
        problems.append("watchdog_pid_invalid")

    observed_at = payload.get("observed_at")
    if not isinstance(observed_at, str):
        problems.append("observed_at_missing")
        return problems
    try:
        observed = dt.datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        problems.append("observed_at_invalid")
        return problems
    if observed.tzinfo is None:
        problems.append("observed_at_naive")
        return problems

    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age = (
        current.astimezone(dt.timezone.utc) - observed.astimezone(dt.timezone.utc)
    ).total_seconds()
    if age < -60:
        problems.append("observed_at_in_future")
    elif age > max_age_seconds:
        problems.append("heartbeat_stale")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--max-age-seconds", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_age_seconds <= 0:
        raise SystemExit("--max-age-seconds must be positive")
    try:
        payload = json.loads(args.state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "problems": [f"state_unreadable:{error}"]}))
        return 1
    problems = validate_watchdog_state(
        payload,
        expected_revision=args.expected_revision,
        max_age_seconds=args.max_age_seconds,
    )
    print(json.dumps({"status": "passed" if not problems else "failed", "problems": problems}))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
