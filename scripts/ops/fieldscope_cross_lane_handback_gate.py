#!/usr/bin/env python3
"""Fail-closed ownership handback between the H200 and A800 formal lanes."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import fcntl
except ModuleNotFoundError:  # Windows can import and test the pure state machine.
    fcntl = None  # type: ignore[assignment]

REVISION = "020c1de567edd88e0eda245fd085335ffe678f47"
SIDE_DATASETS = ("ade20k", "nyuv2", "voc2012")
A800_DATASETS = ("nyuv2", "voc2012")


class GateError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ProcessRef:
    pid: int
    pgid: int
    start_time_ticks: int
    argv_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProcessRef:
        return cls(
            pid=int(value["pid"]),
            pgid=int(value["pgid"]),
            start_time_ticks=int(value["start_time_ticks"]),
            argv_sha256=str(value["argv_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_stable(path: Path) -> tuple[dict[str, Any], str]:
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise GateError(f"unstable_json_double_read:{path}")
    try:
        payload = json.loads(first)
    except json.JSONDecodeError as error:
        raise GateError(f"invalid_json:{path}:{error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"json_not_object:{path}")
    return payload, sha256_bytes(first)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def proc_ref(pid: int, proc_root: Path = Path("/proc")) -> ProcessRef:
    proc = proc_root / str(pid)
    try:
        stat_tail = (proc / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        fields = stat_tail.split()
        argv = (proc / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise GateError(f"process_unreadable:{pid}") from error
    if not argv.rstrip(b"\0"):
        raise GateError(f"process_argv_empty:{pid}")
    return ProcessRef(
        pid=pid,
        pgid=int(fields[2]),
        start_time_ticks=int(fields[19]),
        argv_sha256=sha256_bytes(argv),
    )


def proc_argv(pid: int, proc_root: Path = Path("/proc")) -> list[str]:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise GateError(f"process_argv_unreadable:{pid}") from error
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def proc_parent_pid(pid: int, proc_root: Path = Path("/proc")) -> int:
    try:
        stat_tail = (proc_root / str(pid) / "stat").read_text(
            encoding="utf-8"
        ).rsplit(")", 1)[1]
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise GateError(f"process_parent_unreadable:{pid}") from error
    return int(stat_tail.split()[1])


def require_exact_process(expected: ProcessRef, proc_root: Path = Path("/proc")) -> None:
    actual = proc_ref(expected.pid, proc_root)
    if actual != expected:
        raise GateError(
            f"process_identity_mismatch:{expected.pid}:"
            f"expected={expected}:actual={actual}"
        )


def signal_exact(
    expected: ProcessRef,
    sig: int,
    *,
    group: bool = False,
    proc_root: Path = Path("/proc"),
    sender: Callable[[int, int], None] = os.kill,
) -> None:
    require_exact_process(expected, proc_root)
    sender(-expected.pgid if group else expected.pid, sig)


def process_state(pid: int, proc_root: Path = Path("/proc")) -> str:
    try:
        return (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise GateError(f"process_state_unreadable:{pid}") from error


def process_alive(expected: ProcessRef, proc_root: Path = Path("/proc")) -> bool:
    try:
        require_exact_process(expected, proc_root)
    except GateError:
        return False
    return True


def find_processes(
    required_tokens: tuple[str, ...], proc_root: Path = Path("/proc")
) -> list[ProcessRef]:
    matches = []
    for proc in proc_root.glob("[0-9]*"):
        try:
            argv = proc_argv(int(proc.name), proc_root)
            if all(token in argv for token in required_tokens):
                matches.append(proc_ref(int(proc.name), proc_root))
        except (GateError, ValueError):
            continue
    return sorted(matches, key=lambda item: item.pid)


def require_one_process(
    required_tokens: tuple[str, ...], proc_root: Path = Path("/proc")
) -> ProcessRef:
    matches = find_processes(required_tokens, proc_root)
    if len(matches) != 1:
        raise GateError(
            f"expected_one_process:{required_tokens}:found={len(matches)}"
        )
    return matches[0]


def require_one_script_process(
    script: Path, dataset: str, proc_root: Path = Path("/proc")
) -> ProcessRef:
    matches = find_script_processes(script, dataset, proc_root)
    if len(matches) != 1:
        raise GateError(
            f"expected_one_script_process:{script.resolve()}:{dataset}:found={len(matches)}"
        )
    return matches[0]


def find_script_processes(
    script: Path, dataset: str, proc_root: Path = Path("/proc")
) -> list[ProcessRef]:
    expected = str(script.resolve())
    matches = []
    for proc in proc_root.glob("[0-9]*"):
        try:
            pid = int(proc.name)
            argv = proc_argv(pid, proc_root)
            if len(argv) >= 3 and argv[0] == "bash" and argv[1] == expected:
                # Command substitutions briefly fork a child Bash with the
                # supervisor's argv. Only the detached long-lived process is
                # an independently owned control process.
                if dataset in argv[2:] and proc_parent_pid(pid, proc_root) == 1:
                    matches.append(proc_ref(pid, proc_root))
        except (GateError, ValueError):
            continue
    return sorted(matches, key=lambda item: item.pid)


def checkout_clean(repository: Path) -> bool:
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return revision == REVISION and not dirty


def matrix_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    payload, _ = read_json_stable(path)
    runs = payload.get("runs")
    return payload.get("status") == "passed" and isinstance(runs, list) and len(runs) == 60


def terminal_counts(state: dict[str, Any]) -> dict[str, int]:
    terminal = state.get("terminal_audited")
    if not isinstance(terminal, dict):
        raise GateError("terminal_audited_missing")
    counts = {dataset: 0 for dataset in ("imagenet100", *SIDE_DATASETS)}
    for key in terminal:
        parts = str(key).split("/")
        if len(parts) != 4 or parts[-1] != "terminal":
            raise GateError(f"invalid_terminal_key:{key}")
        dataset = parts[0]
        if dataset not in counts:
            raise GateError(f"unregistered_terminal_dataset:{dataset}")
        counts[dataset] += 1
    if any(value > 60 for value in counts.values()):
        raise GateError(f"terminal_count_overflow:{counts}")
    return counts


def image_terminal_manifest_sha256(state: dict[str, Any]) -> str:
    terminal = state.get("terminal_audited")
    if not isinstance(terminal, dict):
        raise GateError("terminal_audited_missing")
    entries = {
        key: value
        for key, value in terminal.items()
        if str(key).startswith("imagenet100/")
    }
    return sha256_bytes(json.dumps(entries, sort_keys=True).encode())


def coordinator_decision(
    *, image_terminal_cells: int, recovery_stopped: bool, release_complete: bool
) -> str:
    if not 0 <= image_terminal_cells <= 60:
        return "fail_closed"
    if release_complete:
        if image_terminal_cells != 60:
            return "fail_closed"
        return "resume" if recovery_stopped else "complete"
    if image_terminal_cells < 59:
        return "monitor" if not recovery_stopped else "fail_closed"
    if image_terminal_cells == 59:
        return "wait_for_image_terminal" if recovery_stopped else "freeze"
    return "release" if recovery_stopped else "fail_closed"


def lock_is_free(path: Path) -> bool:
    if fcntl is None:
        raise GateError("flock_unavailable_on_this_platform")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def wait_until(predicate: Callable[[], bool], timeout: float, poll: float = 0.2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll)
    raise GateError("wait_timeout")


def stop_exact_process(expected: ProcessRef, timeout: float) -> None:
    signal_exact(expected, signal.SIGTERM)
    try:
        wait_until(lambda: not process_alive(expected), timeout)
    except GateError:
        signal_exact(expected, signal.SIGKILL)
        wait_until(lambda: not process_alive(expected), timeout)


def quiesce_and_stop_group(expected: ProcessRef, timeout: float) -> None:
    signal_exact(expected, signal.SIGSTOP, group=True)
    wait_until(lambda: process_state(expected.pid) in {"T", "t"}, timeout)
    signal_exact(expected, signal.SIGTERM, group=True)
    os.kill(-expected.pgid, signal.SIGCONT)
    try:
        wait_until(lambda: not process_alive(expected), timeout)
    except GateError:
        if process_alive(expected):
            signal_exact(expected, signal.SIGKILL, group=True)
            os.kill(-expected.pgid, signal.SIGCONT)
        wait_until(lambda: not process_alive(expected), timeout)


def validate_registration(payload: dict[str, Any], role: str) -> None:
    if payload.get("status") != "armed" or payload.get("fixed_revision") != REVISION:
        raise GateError(f"invalid_{role}_registration")
    if payload.get("role") != role:
        raise GateError(f"registration_role_mismatch:{role}")


def make_status(
    *, status: str, role: str, problems: list[str], details: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "scope": "cross_lane_ownership_handback_gate",
        "role": role,
        "observed_at": utc_now().isoformat(),
        "fixed_revision": REVISION,
        "problems": problems,
        "details": details,
        "execution_complete": False,
        "method_effectiveness_conclusion": None,
        "changes_scientific_verdict": False,
    }


def worker_ref_from_registry(registry: dict[str, Any], dataset: str) -> ProcessRef:
    entry = registry.get("workers", {}).get(dataset)
    if not isinstance(entry, dict):
        raise GateError(f"registry_worker_missing:{dataset}")
    ref = ProcessRef(
        pid=int(entry["worker_pid"]),
        pgid=int(entry["pgid"]),
        start_time_ticks=int(entry["worker_start_time_ticks"]),
        argv_sha256=proc_ref(int(entry["worker_pid"])).argv_sha256,
    )
    require_exact_process(ref)
    argv = proc_argv(ref.pid)
    if entry.get("expected_output_dir") not in argv:
        raise GateError(f"worker_output_identity_mismatch:{dataset}")
    if not any(
        item.endswith(str(entry.get("expected_train_cache_suffix", "")))
        for item in argv
    ):
        raise GateError(f"worker_cache_identity_mismatch:{dataset}")
    return ref


def register_h200(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    if not checkout_clean(repository):
        raise GateError("formal_checkout_invalid")
    recovery = proc_ref(args.recovery_pid)
    recovery_argv = proc_argv(args.recovery_pid)
    if str(args.recovery_wrapper.resolve()) not in recovery_argv:
        raise GateError("recovery_wrapper_identity_mismatch")
    image = require_one_process(("run-readout-matrix", args.image_output_token))
    registry, registry_sha = read_json_stable(args.h200_registry)
    ade_output = "outputs/full_validation/auraflow_v03/ade20k"
    ade_complete = matrix_complete(repository / ade_output / "matrix_report.json")
    if ade_complete:
        if find_processes(("run-readout-matrix", ade_output)):
            raise GateError("completed_ade_dataset_still_has_writer")
        if not lock_is_free(args.ade_lock):
            raise GateError("completed_ade_lock_held")
        ade = None
        monitor = None
    else:
        ade = worker_ref_from_registry(registry, "ade20k")
        monitor = require_one_script_process(args.h200_monitor, "ade20k")
        if lock_is_free(args.ade_lock):
            raise GateError("ade_lock_not_held")
    payload = {
        "schema_version": 1,
        "status": "armed",
        "role": "h200",
        "observed_at": utc_now().isoformat(),
        "fixed_revision": REVISION,
        "repository": str(repository),
        "recovery": recovery.to_dict(),
        "image_worker": image.to_dict(),
        "ade_worker": None if ade is None else ade.to_dict(),
        "ade_monitor": None if monitor is None else monitor.to_dict(),
        "ade_matrix_complete": ade_complete,
        "ade_lock": str(args.ade_lock.resolve()),
        "h200_monitor": str(args.h200_monitor.resolve()),
        "h200_registry": str(args.h200_registry.resolve()),
        "h200_registry_sha256": registry_sha,
        "recovery_wrapper_sha256": sha256(args.recovery_wrapper),
        "gate_script_sha256": sha256(Path(__file__).resolve()),
        "execution_complete": False,
        "method_effectiveness_conclusion": None,
        "changes_scientific_verdict": False,
    }
    atomic_write_json(args.output, payload)
    return payload


def discover_a800_processes(
    args: argparse.Namespace, *, allow_partial_release: bool = False
) -> dict[str, Any]:
    repository = args.repository.resolve()
    if not checkout_clean(repository):
        raise GateError("formal_checkout_invalid")
    registry, registry_sha = read_json_stable(args.a800_registry)
    if registry.get("fixed_revision") != REVISION:
        raise GateError("a800_registry_revision_mismatch")
    workers = {}
    supervisors = {}
    monitors = {}
    completed = {}
    for dataset in A800_DATASETS:
        entry = registry["workers"][dataset]
        output_token = str(entry["expected_output_dir"])
        complete = matrix_complete(repository / output_token / "matrix_report.json")
        completed[dataset] = complete
        if complete:
            if find_processes(("run-readout-matrix", output_token)):
                raise GateError(f"completed_a800_dataset_still_has_writer:{dataset}")
            if not lock_is_free(Path(entry["expected_lock_file"])):
                raise GateError(f"completed_a800_dataset_lock_held:{dataset}")
            workers[dataset] = None
            supervisors[dataset] = None
            monitors[dataset] = None
            continue
        try:
            worker = worker_ref_from_registry(registry, dataset)
        except GateError:
            if find_processes(("run-readout-matrix", output_token)):
                raise
            worker = None
        supervisor_matches = find_script_processes(args.a800_supervisor, dataset)
        monitor_matches = find_script_processes(args.a800_monitor, dataset)
        if len(supervisor_matches) > 1 or len(monitor_matches) > 1:
            raise GateError(f"duplicate_a800_control_process:{dataset}")
        supervisor = supervisor_matches[0] if supervisor_matches else None
        monitor = monitor_matches[0] if monitor_matches else None
        lock_free = lock_is_free(Path(entry["expected_lock_file"]))
        if worker is not None and lock_free:
            raise GateError(f"a800_live_worker_lock_not_held:{dataset}")
        if worker is None and not lock_free:
            raise GateError(f"a800_missing_worker_lock_still_held:{dataset}")
        if not allow_partial_release and (
            worker is None or supervisor is None or monitor is None
        ):
            raise GateError(f"a800_active_lane_not_fully_owned:{dataset}")
        workers[dataset] = None if worker is None else worker.to_dict()
        supervisors[dataset] = None if supervisor is None else supervisor.to_dict()
        monitors[dataset] = None if monitor is None else monitor.to_dict()
    return {
        "repository": str(repository),
        "registry": str(args.a800_registry.resolve()),
        "registry_sha256": registry_sha,
        "workers": workers,
        "supervisors": supervisors,
        "monitors": monitors,
        "matrix_complete": completed,
        "locks": {
            dataset: registry["workers"][dataset]["expected_lock_file"]
            for dataset in A800_DATASETS
        },
    }


def register_a800(args: argparse.Namespace) -> dict[str, Any]:
    discovered = discover_a800_processes(args)
    payload = {
        "schema_version": 1,
        "status": "armed",
        "role": "a800",
        "observed_at": utc_now().isoformat(),
        "fixed_revision": REVISION,
        **discovered,
        "supervisor_script_sha256": sha256(args.a800_supervisor),
        "monitor_script_sha256": sha256(args.a800_monitor),
        "gate_script_sha256": sha256(Path(__file__).resolve()),
        "execution_complete": False,
        "method_effectiveness_conclusion": None,
        "changes_scientific_verdict": False,
    }
    atomic_write_json(args.output, payload)
    return payload


def audit_h200(args: argparse.Namespace) -> dict[str, Any]:
    registration, registration_sha = read_json_stable(args.registration)
    validate_registration(registration, "h200")
    if registration.get("gate_script_sha256") != sha256(Path(__file__).resolve()):
        raise GateError("h200_gate_script_sha256_mismatch")
    repository = Path(registration["repository"])
    if not checkout_clean(repository):
        raise GateError("formal_checkout_invalid")
    recovery = ProcessRef.from_dict(registration["recovery"])
    require_exact_process(recovery)
    state, state_sha = read_json_stable(args.watcher_state)
    if state.get("failures"):
        raise GateError("watcher_has_unresolved_failures")
    counts = terminal_counts(state)
    recovery_stopped = process_state(recovery.pid) in {"T", "t"}
    decision = coordinator_decision(
        image_terminal_cells=counts["imagenet100"],
        recovery_stopped=recovery_stopped,
        release_complete=args.release_complete.exists(),
    )
    if decision == "fail_closed":
        raise GateError("coordinator_state_fail_closed")
    if counts["imagenet100"] < 60:
        require_exact_process(ProcessRef.from_dict(registration["image_worker"]))
    ade_output = "outputs/full_validation/auraflow_v03/ade20k"
    ade_complete = matrix_complete(repository / ade_output / "matrix_report.json")
    allow_ade_partial = (
        counts["imagenet100"] == 60
        and recovery_stopped
        and args.request.exists()
    )
    if ade_complete:
        if find_processes(("run-readout-matrix", ade_output)):
            raise GateError("completed_ade_dataset_still_has_writer")
        if not lock_is_free(Path(registration["ade_lock"])):
            raise GateError("completed_ade_lock_held")
    else:
        worker = ProcessRef.from_dict(registration["ade_worker"])
        monitor = ProcessRef.from_dict(registration["ade_monitor"])
        worker_alive = process_alive(worker)
        monitor_alive = process_alive(monitor)
        lock_free = lock_is_free(Path(registration["ade_lock"]))
        if not allow_ade_partial:
            require_exact_process(worker)
            require_exact_process(monitor)
            if lock_free:
                raise GateError("ade_lock_not_held")
        else:
            if not worker_alive and find_processes(("run-readout-matrix", ade_output)):
                raise GateError("unregistered_ade_writer_during_release")
            if not monitor_alive and find_script_processes(
                Path(registration["h200_monitor"]), "ade20k"
            ):
                raise GateError("unregistered_ade_monitor_during_release")
            if worker_alive and lock_free:
                raise GateError("ade_live_worker_lock_not_held")
            if not worker_alive and not lock_free:
                raise GateError("ade_missing_worker_lock_still_held")
    responder_sha = None
    if not args.a800_ack.exists():
        responder, responder_sha = read_json_stable(args.responder_heartbeat)
        if responder.get("status") != "armed" or responder.get("problems"):
            raise GateError("a800_responder_not_armed")
        observed = dt.datetime.fromisoformat(
            str(responder["observed_at"]).replace("Z", "+00:00")
        )
        if (
            utc_now() - observed.astimezone(dt.timezone.utc)
        ).total_seconds() > args.max_age:
            raise GateError("a800_responder_heartbeat_stale")
    return make_status(
        status="passed",
        role="h200",
        problems=[],
        details={
            "decision": decision,
            "terminal_counts": counts,
            "registration_sha256": registration_sha,
            "watcher_state_sha256": state_sha,
            "responder_heartbeat_sha256": responder_sha,
            "recovery_stopped": recovery_stopped,
            "ade_matrix_complete": ade_complete,
        },
    )


def audit_a800(args: argparse.Namespace) -> dict[str, Any]:
    registration, registration_sha = read_json_stable(args.registration)
    validate_registration(registration, "a800")
    if registration.get("gate_script_sha256") != sha256(Path(__file__).resolve()):
        raise GateError("a800_gate_script_sha256_mismatch")
    if registration.get("supervisor_script_sha256") != sha256(args.a800_supervisor):
        raise GateError("a800_supervisor_script_sha256_mismatch")
    if registration.get("monitor_script_sha256") != sha256(args.a800_monitor):
        raise GateError("a800_monitor_script_sha256_mismatch")
    discovered = discover_a800_processes(args)
    return make_status(
        status="armed",
        role="a800",
        problems=[],
        details={
            "registration_sha256": registration_sha,
            "live": discovered,
        },
    )


def release_a800(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    if request.get("status") != "release_requested":
        raise GateError("invalid_release_request_status")
    if request.get("fixed_revision") != REVISION:
        raise GateError("release_request_revision_mismatch")
    if int(request.get("imagenet100_terminal_cells", -1)) != 60:
        raise GateError("release_requested_before_image_complete")
    if request.get("gate_script_sha256") != sha256(Path(__file__).resolve()):
        raise GateError("release_request_gate_script_mismatch")
    watcher, _ = read_json_stable(args.watcher_state)
    if watcher.get("failures") or terminal_counts(watcher)["imagenet100"] != 60:
        raise GateError("shared_watcher_does_not_prove_image_complete")
    if request.get("image_terminal_manifest_sha256") != image_terminal_manifest_sha256(
        watcher
    ):
        raise GateError("release_request_image_terminal_manifest_changed")
    live = discover_a800_processes(args, allow_partial_release=True)
    for dataset in A800_DATASETS:
        if live["supervisors"][dataset] is not None:
            stop_exact_process(
                ProcessRef.from_dict(live["supervisors"][dataset]), args.timeout
            )
    for dataset in A800_DATASETS:
        if live["monitors"][dataset] is not None:
            stop_exact_process(
                ProcessRef.from_dict(live["monitors"][dataset]), args.timeout
            )
    quiesced_state, quiesced_sha = read_json_stable(args.watcher_state)
    if quiesced_state.get("failures"):
        raise GateError("watcher_has_unresolved_failures_before_a800_release")
    for dataset in A800_DATASETS:
        if live["workers"][dataset] is not None:
            quiesce_and_stop_group(
                ProcessRef.from_dict(live["workers"][dataset]), args.timeout
            )
    for dataset, lock in live["locks"].items():
        if not lock_is_free(Path(lock)):
            raise GateError(f"a800_lock_not_released:{dataset}")
    return {
        **make_status(
            status="released",
            role="a800",
            problems=[],
            details={
                "request_id": request.get("request_id"),
                "released_datasets": list(A800_DATASETS),
                "quiesced_watcher_state_sha256": quiesced_sha,
                "live_before_release": live,
            },
        ),
        "request_id": request.get("request_id"),
    }


def run_a800_responder(args: argparse.Namespace) -> int:
    while True:
        if args.ack.exists():
            ack, _ = read_json_stable(args.ack)
            return 0 if ack.get("status") == "released" else 1
        if args.request.exists():
            try:
                request, _ = read_json_stable(args.request)
                ack = release_a800(args, request)
            except Exception as error:
                failed = make_status(
                    status="failed", role="a800", problems=[str(error)], details={}
                )
                stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
                failed_path = args.ack.with_name(
                    f"{args.ack.stem}.failed-attempt-{stamp}.json"
                )
                atomic_write_json(failed_path, failed)
                return 1
            atomic_write_json(args.ack, ack)
            return 0
        try:
            heartbeat = audit_a800(args)
        except Exception as error:  # fail closed and publish the reason
            heartbeat = make_status(
                status="failed", role="a800", problems=[str(error)], details={}
            )
        atomic_write_json(args.heartbeat, heartbeat)
        if args.once:
            return 0 if heartbeat["status"] == "armed" else 1
        time.sleep(args.poll)


def release_ade(registration: dict[str, Any], timeout: float) -> dict[str, Any]:
    repository = Path(registration["repository"])
    output_token = "outputs/full_validation/auraflow_v03/ade20k"
    lock = Path(registration["ade_lock"])
    if matrix_complete(repository / output_token / "matrix_report.json"):
        if find_processes(("run-readout-matrix", output_token)):
            raise GateError("completed_ade_dataset_still_has_writer")
        if not lock_is_free(lock):
            raise GateError("completed_ade_lock_held")
        return {"already_complete": True, "lock": str(lock)}
    monitor = ProcessRef.from_dict(registration["ade_monitor"])
    worker = ProcessRef.from_dict(registration["ade_worker"])
    if process_alive(monitor):
        stop_exact_process(monitor, timeout)
    elif find_script_processes(Path(registration["h200_monitor"]), "ade20k"):
        raise GateError("unregistered_ade_monitor_during_release")
    if process_alive(worker):
        quiesce_and_stop_group(worker, timeout)
    elif find_processes(("run-readout-matrix", output_token)):
        raise GateError("unregistered_ade_writer_during_release")
    if not lock_is_free(lock):
        raise GateError("ade_lock_not_released")
    return {
        "already_complete": False,
        "worker": worker.to_dict(),
        "monitor": monitor.to_dict(),
        "lock": str(lock),
    }


def run_h200_coordinator(args: argparse.Namespace) -> int:
    registration, _ = read_json_stable(args.registration)
    validate_registration(registration, "h200")
    recovery = ProcessRef.from_dict(registration["recovery"])
    require_exact_process(recovery)
    frozen = process_state(recovery.pid) in {"T", "t"}
    while True:
        audit = audit_h200(args)
        atomic_write_json(args.heartbeat, audit)
        decision = audit["details"]["decision"]
        if args.once:
            return 0
        if decision == "monitor":
            time.sleep(args.poll)
            continue
        if decision == "freeze":
            signal_exact(recovery, signal.SIGSTOP)
            wait_until(lambda: process_state(recovery.pid) in {"T", "t"}, args.timeout)
            frozen = True
            continue
        if decision == "wait_for_image_terminal":
            frozen = True
            time.sleep(args.poll)
            continue
        if decision == "resume":
            signal_exact(recovery, signal.SIGCONT)
            return 0
        if decision == "complete":
            return 0
        if decision != "release" or not frozen:
            raise GateError(f"unexpected_coordinator_decision:{decision}")

        state, state_sha = read_json_stable(args.watcher_state)
        counts = terminal_counts(state)
        image_manifest_sha = image_terminal_manifest_sha256(state)
        if args.request.exists():
            request, _ = read_json_stable(args.request)
            request_id = str(request.get("request_id", ""))
            if (
                request.get("status") != "release_requested"
                or request.get("fixed_revision") != REVISION
                or request.get("image_terminal_manifest_sha256")
                != image_manifest_sha
            ):
                raise GateError("existing_release_request_invalid")
        else:
            request_id = sha256_bytes(
                f"{REVISION}:{image_manifest_sha}:ownership-handback".encode()
            )
            request = {
                "schema_version": 1,
                "status": "release_requested",
                "scope": "cross_lane_ownership_handback_request",
                "request_id": request_id,
                "observed_at": utc_now().isoformat(),
                "fixed_revision": REVISION,
                "imagenet100_terminal_cells": counts["imagenet100"],
                "watcher_state_sha256_at_request": state_sha,
                "image_terminal_manifest_sha256": image_manifest_sha,
                "gate_script_sha256": sha256(Path(__file__).resolve()),
                "execution_complete": False,
                "method_effectiveness_conclusion": None,
                "changes_scientific_verdict": False,
            }
            atomic_write_json(args.request, request)
        ade_ack = release_ade(registration, args.timeout)
        wait_until(lambda: args.a800_ack.exists(), args.ack_timeout, args.poll)
        a800_ack, a800_ack_sha = read_json_stable(args.a800_ack)
        if (
            a800_ack.get("status") != "released"
            or a800_ack.get("request_id") != request_id
            or a800_ack.get("problems")
        ):
            raise GateError("a800_release_ack_invalid")
        for lock in (
            Path(registration["ade_lock"]),
            Path(a800_ack["details"]["live_before_release"]["locks"]["nyuv2"]),
            Path(a800_ack["details"]["live_before_release"]["locks"]["voc2012"]),
        ):
            if not lock_is_free(lock):
                raise GateError(f"cross_host_lock_not_free:{lock}")
        completion = {
            **make_status(
                status="released",
                role="h200",
                problems=[],
                details={
                    "request_id": request_id,
                    "ade_ack": ade_ack,
                    "a800_ack_sha256": a800_ack_sha,
                    "terminal_counts": counts,
                },
            ),
            "request_id": request_id,
        }
        atomic_write_json(args.release_complete, completion)
        signal_exact(recovery, signal.SIGCONT)
        return 0


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--watcher-state", type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    hreg = sub.add_parser("register-h200")
    add_common_paths(hreg)
    hreg.add_argument("--recovery-pid", type=int, required=True)
    hreg.add_argument("--recovery-wrapper", type=Path, required=True)
    hreg.add_argument("--image-output-token", required=True)
    hreg.add_argument("--h200-registry", type=Path, required=True)
    hreg.add_argument("--h200-monitor", type=Path, required=True)
    hreg.add_argument("--ade-lock", type=Path, required=True)
    hreg.add_argument("--output", type=Path, required=True)

    areg = sub.add_parser("register-a800")
    add_common_paths(areg)
    areg.add_argument("--a800-registry", type=Path, required=True)
    areg.add_argument("--a800-supervisor", type=Path, required=True)
    areg.add_argument("--a800-monitor", type=Path, required=True)
    areg.add_argument("--output", type=Path, required=True)

    for name in ("audit-h200", "run-h200"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--registration", type=Path, required=True)
        cmd.add_argument("--watcher-state", type=Path, required=True)
        cmd.add_argument("--responder-heartbeat", type=Path, required=True)
        cmd.add_argument("--heartbeat", type=Path, required=True)
        cmd.add_argument("--request", type=Path, required=True)
        cmd.add_argument("--a800-ack", type=Path, required=True)
        cmd.add_argument("--release-complete", type=Path, required=True)
        cmd.add_argument("--max-age", type=int, default=300)
        cmd.add_argument("--poll", type=float, default=30)
        cmd.add_argument("--timeout", type=float, default=30)
        cmd.add_argument("--ack-timeout", type=float, default=600)
        cmd.add_argument("--once", action="store_true")

    for name in ("audit-a800", "run-a800"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--registration", type=Path, required=True)
        cmd.add_argument("--repository", type=Path, required=True)
        cmd.add_argument("--watcher-state", type=Path, required=True)
        cmd.add_argument("--a800-registry", type=Path, required=True)
        cmd.add_argument("--a800-supervisor", type=Path, required=True)
        cmd.add_argument("--a800-monitor", type=Path, required=True)
        cmd.add_argument("--heartbeat", type=Path, required=True)
        cmd.add_argument("--request", type=Path, required=True)
        cmd.add_argument("--ack", type=Path, required=True)
        cmd.add_argument("--poll", type=float, default=30)
        cmd.add_argument("--timeout", type=float, default=30)
        cmd.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "register-h200":
            payload = register_h200(args)
        elif args.command == "register-a800":
            payload = register_a800(args)
        elif args.command == "audit-h200":
            payload = audit_h200(args)
            atomic_write_json(args.heartbeat, payload)
        elif args.command == "audit-a800":
            payload = audit_a800(args)
            atomic_write_json(args.heartbeat, payload)
        elif args.command == "run-h200":
            return run_h200_coordinator(args)
        elif args.command == "run-a800":
            return run_a800_responder(args)
        else:  # pragma: no cover
            raise GateError(f"unsupported_command:{args.command}")
    except Exception as error:
        print(json.dumps({"status": "failed", "problems": [str(error)]}))
        return 1
    print(json.dumps({"status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
