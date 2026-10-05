from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ops"
    / "fieldscope_cross_lane_handback_gate.py"
)
SPEC = importlib.util.spec_from_file_location("cross_lane_handback_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_proc(
    root: Path,
    pid: int,
    *,
    pgid: int,
    start: int,
    argv: list[str],
    ppid: int = 1,
) -> None:
    proc = root / str(pid)
    proc.mkdir(parents=True)
    # fields after the closing paren: state, ppid, pgrp, ... starttime at index 19
    fields = ["S", str(ppid), str(pgid), "1"] + ["0"] * 15 + [str(start)] + ["0"] * 3
    (proc / "stat").write_text(f"{pid} (worker) " + " ".join(fields))
    (proc / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")


def test_terminal_counts_require_registered_exact_terminal_keys() -> None:
    state = {
        "terminal_audited": {
            "imagenet100/z0/seed-1/terminal": {},
            "nyuv2/z0/seed-1/terminal": {},
            "voc2012/z0/seed-1/terminal": {},
        }
    }
    assert MODULE.terminal_counts(state) == {
        "imagenet100": 1,
        "ade20k": 0,
        "nyuv2": 1,
        "voc2012": 1,
    }
    state["terminal_audited"]["unknown/z0/seed-1/terminal"] = {}
    with pytest.raises(MODULE.GateError, match="unregistered_terminal_dataset"):
        MODULE.terminal_counts(state)


def test_image_manifest_ignores_side_lane_progress() -> None:
    state = {
        "terminal_audited": {
            "imagenet100/z0/seed-1/terminal": {"output": "image.json"},
            "nyuv2/z0/seed-1/terminal": {"output": "nyu-one.json"},
        }
    }
    before = MODULE.image_terminal_manifest_sha256(state)
    state["terminal_audited"]["nyuv2/z0/seed-2/terminal"] = {
        "output": "nyu-two.json"
    }
    assert MODULE.image_terminal_manifest_sha256(state) == before


@pytest.mark.parametrize(
    ("cells", "stopped", "complete", "expected"),
    [
        (13, False, False, "monitor"),
        (58, True, False, "fail_closed"),
        (59, False, False, "freeze"),
        (59, True, False, "wait_for_image_terminal"),
        (60, False, False, "fail_closed"),
        (60, True, False, "release"),
        (60, True, True, "resume"),
        (60, False, True, "complete"),
    ],
)
def test_coordinator_decision_is_fail_closed(
    cells: int, stopped: bool, complete: bool, expected: str
) -> None:
    assert MODULE.coordinator_decision(
        image_terminal_cells=cells,
        recovery_stopped=stopped,
        release_complete=complete,
    ) == expected


def test_read_json_stable_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps([]))
    with pytest.raises(MODULE.GateError, match="json_not_object"):
        MODULE.read_json_stable(path)


def test_exact_signal_rechecks_start_tick_and_argv(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    write_proc(proc_root, 41, pgid=40, start=900, argv=["python", "worker"])
    expected = MODULE.proc_ref(41, proc_root)
    sent: list[tuple[int, int]] = []

    MODULE.signal_exact(
        expected,
        19,
        group=True,
        proc_root=proc_root,
        sender=lambda pid, sig: sent.append((pid, sig)),
    )
    assert sent == [(-40, 19)]

    write = proc_root / "41" / "cmdline"
    write.write_bytes(b"python\0replacement\0")
    with pytest.raises(MODULE.GateError, match="process_identity_mismatch"):
        MODULE.signal_exact(
            expected,
            15,
            group=True,
            proc_root=proc_root,
            sender=lambda pid, sig: sent.append((pid, sig)),
        )
    assert sent == [(-40, 19)]


def test_release_request_must_prove_image_60() -> None:
    request = {
        "status": "release_requested",
        "fixed_revision": MODULE.REVISION,
        "imagenet100_terminal_cells": 59,
    }
    args = type("Args", (), {})()
    with pytest.raises(MODULE.GateError, match="release_requested_before_image_complete"):
        MODULE.release_a800(args, request)


def test_script_process_match_uses_argv_position_not_incidental_argument(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    script = tmp_path / "monitor.sh"
    script.write_text("#!/bin/sh\n")
    supervisor = tmp_path / "supervisor.sh"
    supervisor.write_text("#!/bin/sh\n")
    write_proc(
        proc_root,
        50,
        pgid=50,
        start=1,
        argv=["bash", str(script.resolve()), "root", "repo", "nyuv2"],
    )
    write_proc(
        proc_root,
        51,
        pgid=51,
        start=2,
        argv=[
            "bash",
            str(supervisor.resolve()),
            "root",
            "repo",
            "nyuv2",
            str(script.resolve()),
        ],
    )

    matched = MODULE.require_one_script_process(script, "nyuv2", proc_root)

    assert matched.pid == 50


def test_script_process_match_ignores_transient_bash_subshell(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    script = tmp_path / "supervisor.sh"
    script.write_text("#!/bin/sh\n")
    argv = ["bash", str(script.resolve()), "root", "repo", "nyuv2"]
    write_proc(proc_root, 50, pgid=50, start=1, argv=argv)
    write_proc(proc_root, 51, pgid=50, start=2, argv=argv, ppid=50)

    matched = MODULE.require_one_script_process(script, "nyuv2", proc_root)

    assert matched.pid == 50
