#!/usr/bin/env python3
"""Wait for ADE20K retry-2 epoch 1 and bind it to the recovery shim audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import subprocess
import time
from typing import Any

EXPECTED_REVISION = "020c1de567edd88e0eda245fd085335ffe678f47"
EXPECTED_SHIM_SHA256 = "f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4"
EXPECTED_GATE_SHA256 = "fe59df4ce274fb809dd12acb3f143872cdade242b102a7ce7ee2c0b8d703625d"
EXPECTED_SCAN_SHA256 = "ef4d7b5a0a9474fbe8692de8ca3c87191a80b8335155ce292719cb28fcf83788"
EXPECTED_SAMPLES = 18189
EXPECTED_BATCH_SIZE = 2
EXPECTED_STEPS = math.ceil(EXPECTED_SAMPLES / EXPECTED_BATCH_SIZE)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_environment(pid: int) -> list[str]:
    raw = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def git_identity(repository: pathlib.Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--retry-pid", required=True, type=int)
    parser.add_argument("--cli-pid", required=True, type=int)
    parser.add_argument("--recovery-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    root = args.root.resolve()
    repository = root / "recovery/worktrees/formal-020c1de"
    analysis = root / "recovery/worktrees/analysis-020c1de"
    report = (
        repository
        / "outputs/full_validation/auraflow_v03/ade20k/random_feature_local/seed-4121"
        / "segmentation_random_feature_local_seed4121_report.json"
    )
    shim = (
        root
        / "recovery/orchestration/020c1de"
        / f"sitecompat-ade-all-ignore-{EXPECTED_SHIM_SHA256}/sitecustomize.py"
    )
    gate = (
        root
        / "FieldScope-internal/artifacts/reports"
        / "ade20k_all_ignore_recovery_gate_20260823.json"
    )
    scan = (
        root
        / "FieldScope-internal/artifacts/reports"
        / "ade20k_train_label_full_scan_20260823.json"
    )
    helper = root / "logs/audit_helpers/fieldscope_audit_cell.py"
    profile = next(
        (root / "recovery/fixed-revision-readout").glob(
            "readout_runtime_profile_*_30c6073ce73f6de74803eb29f497ceedf1ee3cafc488340cf236395824ea42e3.json"
        )
    )
    standard_output = args.output.with_name(
        "ade20k_random_feature_local_seed4121_epoch1_standard_strong_audit_retry2.json"
    )

    while alive(args.recovery_pid):
        if report.is_file():
            try:
                payload = json.loads(report.read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(args.poll_seconds)
                continue
            history = payload.get("history", [])
            if history and int(history[-1].get("epoch", 0)) >= 1:
                break
        if not alive(args.retry_pid):
            atomic_json(
                args.output,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "reason": "retry2_exited_before_atomic_epoch1",
                    "retry_pid": args.retry_pid,
                    "cli_pid": args.cli_pid,
                    "execution_complete": False,
                    "method_effectiveness_conclusion": None,
                    "changes_scientific_verdict": False,
                },
            )
            return 1
        time.sleep(args.poll_seconds)
    else:
        atomic_json(
            args.output,
            {
                "schema_version": 1,
                "status": "stopped_primary_recovery_exited",
                "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "execution_complete": False,
                "method_effectiveness_conclusion": None,
                "changes_scientific_verdict": False,
            },
        )
        return 2

    command = [
        str(root / ".venv/bin/python"),
        str(helper),
        "--report",
        str(report),
        "--output",
        str(standard_output),
        "--repository",
        str(repository),
        "--analysis-repository",
        str(analysis),
        "--runtime-profile",
        str(profile),
        "--expected-revision",
        EXPECTED_REVISION,
        "--expected-epoch",
        "1",
        "--expected-task",
        "segmentation",
        "--expected-representation",
        "random_feature_local",
        "--expected-seed",
        "4121",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    payload = json.loads(report.read_text())
    history = payload.get("history", [])
    entry = history[-1] if history else {}
    formal_revision, formal_clean = git_identity(repository)
    analysis_revision, analysis_clean = git_identity(analysis)
    environment = process_environment(args.cli_pid) if alive(args.cli_pid) else []
    expected_shim_path = str(shim.parent)
    problems: list[str] = []
    checks = {
        "standard_audit_passed": completed.returncode == 0,
        "report_status_running": payload.get("status") == "running",
        "committed_epoch_one": int(entry.get("epoch", -1)) == 1,
        "optimizer_steps_exact": int(payload.get("completed_optimizer_steps", -1))
        == EXPECTED_STEPS,
        "sample_exposures_exact": int(
            payload.get("completed_training_sample_exposures", -1)
        )
        == EXPECTED_SAMPLES,
        "train_loss_finite": math.isfinite(float(entry.get("train_loss", math.nan))),
        "formal_revision_matches": formal_revision == EXPECTED_REVISION,
        "formal_worktree_clean": formal_clean,
        "analysis_revision_matches": analysis_revision == EXPECTED_REVISION,
        "analysis_worktree_clean": analysis_clean,
        "shim_hash_matches": sha256(shim) == EXPECTED_SHIM_SHA256,
        "gate_hash_matches": sha256(gate) == EXPECTED_GATE_SHA256,
        "scan_hash_matches": sha256(scan) == EXPECTED_SCAN_SHA256,
        "cli_gpu_binding_matches": "CUDA_VISIBLE_DEVICES=7" in environment,
        "cli_shim_identity_matches": (
            f"FIELDSCOPE_ADE_ALL_IGNORE_SHIM_SHA256={EXPECTED_SHIM_SHA256}"
            in environment
        ),
        "cli_pythonpath_contains_shim": any(
            item.startswith("PYTHONPATH=") and expected_shim_path in item
            for item in environment
        ),
    }
    for name, passed in checks.items():
        if not passed:
            problems.append(name)
    result = {
        "schema_version": 1,
        "status": "passed" if not problems else "failed",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "ade20k_retry2_epoch1_shim_bound_external_strong_audit",
        "task": "ade20k",
        "representation": "random_feature_local",
        "seed": 4121,
        "committed_epoch": int(entry.get("epoch", -1)),
        "completed_optimizer_steps": payload.get("completed_optimizer_steps"),
        "completed_training_sample_exposures": payload.get(
            "completed_training_sample_exposures"
        ),
        "train_loss": entry.get("train_loss"),
        "report": str(report),
        "report_sha256": sha256(report),
        "standard_audit": str(standard_output),
        "standard_audit_sha256": (
            sha256(standard_output) if standard_output.is_file() else None
        ),
        "standard_audit_stdout": completed.stdout,
        "standard_audit_stderr": completed.stderr,
        "fixed_revision": EXPECTED_REVISION,
        "shim_sha256": EXPECTED_SHIM_SHA256,
        "recovery_gate_sha256": EXPECTED_GATE_SHA256,
        "label_scan_sha256": EXPECTED_SCAN_SHA256,
        "registered_all_ignore_batch_index": 1697,
        "registered_all_ignore_cache_indices": [5336, 5328],
        "checks": checks,
        "problems": problems,
        "execution_complete": False,
        "method_effectiveness_conclusion": None,
        "changes_scientific_verdict": False,
    }
    atomic_json(args.output, result)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
