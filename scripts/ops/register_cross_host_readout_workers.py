#!/usr/bin/env python3
"""Atomically register exact cross-host readout worker identities."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import socket
import subprocess
from pathlib import Path

REVISION = "020c1de567edd88e0eda245fd085335ffe678f47"


def process_identity(
    *, repository: Path, python_bin: Path, dataset: str, gpu: int, lock_file: Path
) -> dict:
    matches = []
    expected_output = f"outputs/full_validation/auraflow_v03/{dataset}"
    expected_cache = f"/feature_cache/auraflow_v03/{dataset}_train"
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = [
                item.decode(errors="replace")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if len(argv) < 4 or argv[1:4] != [
                "-m",
                "fieldscope.cli",
                "run-readout-matrix",
            ]:
                continue
            if Path(argv[0]).resolve() != python_bin:
                continue
            if expected_output not in argv:
                continue
            if not any(value.endswith(expected_cache) for value in argv):
                continue
            if (proc / "cwd").resolve() != repository:
                raise RuntimeError(f"worker cwd mismatch: {proc.name}")
            environment = {}
            for item in (proc / "environ").read_bytes().split(b"\0"):
                if b"=" in item:
                    key, value = item.split(b"=", 1)
                    environment[key.decode(errors="replace")] = value.decode(
                        errors="replace"
                    )
            if environment.get("CUDA_VISIBLE_DEVICES") != str(gpu):
                raise RuntimeError(f"worker GPU mismatch: {proc.name}")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            matches.append(
                {
                    "worker_pid": int(proc.name),
                    "parent_pid": int(stat[1]),
                    "pgid": int(stat[2]),
                    "worker_start_time_ticks": int(stat[19]),
                    "runtime_hostname": socket.gethostname(),
                    "physical_gpu": gpu,
                    "expected_cuda_visible_devices": str(gpu),
                    "expected_output_dir": expected_output,
                    "expected_train_cache_suffix": expected_cache,
                    "expected_lock_file": str(lock_file.resolve()),
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"expected one {dataset} worker, found {len(matches)}")
    with lock_file.open("r", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise RuntimeError(f"worker lock is not held: {lock_file}")
    return matches[0]


def parse_spec(raw: str) -> tuple[str, int, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or parts[0] not in {"nyuv2", "voc2012"}:
        raise argparse.ArgumentTypeError("worker must be DATASET:GPU:LOCK")
    try:
        gpu = int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU must be an integer") from error
    if gpu not in {0, 1}:
        raise argparse.ArgumentTypeError("GPU must be 0 or 1")
    return parts[0], gpu, Path(parts[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="append", type=parse_spec, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repository = args.repository.resolve()
    python_bin = (root / ".venv/bin/python").resolve()
    if subprocess.check_output(
        ["git", "-C", repository, "rev-parse", "HEAD"], text=True
    ).strip() != REVISION:
        raise SystemExit("fixed revision mismatch")
    if subprocess.check_output(
        ["git", "-C", repository, "status", "--porcelain"], text=True
    ).strip():
        raise SystemExit("formal worktree is dirty")
    specs = args.worker
    if len(specs) != 2 or {item[0] for item in specs} != {"nyuv2", "voc2012"}:
        raise SystemExit("exactly one nyuv2 and one voc2012 worker are required")

    workers = {
        dataset: process_identity(
            repository=repository,
            python_bin=python_bin,
            dataset=dataset,
            gpu=gpu,
            lock_file=lock_file,
        )
        for dataset, gpu, lock_file in specs
    }
    payload = {
        "schema_version": 1,
        "status": "active",
        "scope": "registered_cross_host_a800_readout_workers",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fixed_revision": REVISION,
        "repository": str(repository),
        "runtime_hostname": socket.gethostname(),
        "per_task_seed_workers": 1,
        "workers": workers,
        "execution_complete": False,
        "method_effectiveness_conclusion": None,
        "changes_scientific_verdict": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": "passed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
