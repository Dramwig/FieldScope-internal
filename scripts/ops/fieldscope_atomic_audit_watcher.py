import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time


def atomic_json(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


RACE_ONLY_PROBLEMS = {
    "history_contiguous",
    "report_epoch_matches",
    "checkpoint_epoch_matches",
    "report_steps_match",
    "checkpoint_steps_match",
    "report_exposures_match",
    "checkpoint_exposures_match",
    "optimizer_state_steps_match",
    "scheduler_last_epoch_matches",
}

UNSTABLE_OBSERVATION_PROBLEMS = {
    "stable_checkpoint_during_hash_and_load",
    "stable_report_double_read",
}

AUDIT_IDENTITY_FIELDS = (
    "changes_scientific_verdict",
    "fixed_revision",
    "representation",
    "runtime_profile_seed_workers",
    "seed",
    "target_epochs",
    "task",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_advanced_race(
    report_path: pathlib.Path,
    audit_path: pathlib.Path,
    expected_epoch: int,
) -> dict | None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("status") != "running" or not report.get("history"):
        return None
    current_epoch = int(report["history"][-1]["epoch"])
    problems = set(audit.get("problems", []))
    if current_epoch <= expected_epoch:
        return None
    if audit.get("status") != "failed" or not problems or not problems <= RACE_ONLY_PROBLEMS:
        return None
    audited_epoch = int(audit.get("scheduler_last_epoch", -1))
    checks = audit.get("checks", {})
    if audited_epoch <= expected_epoch or current_epoch < audited_epoch:
        return None
    if not checks.get("stable_report_double_read") or not checks.get(
        "stable_checkpoint_during_hash_and_load"
    ):
        return None
    if not checks.get("report_checkpoint_history_equal"):
        return None
    return {
        "classification": "report_advanced_during_audit",
        "expected_epoch": expected_epoch,
        "audited_epoch": audited_epoch,
        "current_epoch": current_epoch,
        "audited_report_sha256": audit.get("report_sha256"),
        "current_report_sha256": sha256(report_path),
        "unresolved_scientific_failure": False,
        "failed_artifact_preserved": True,
    }


def later_stable_audit_race(
    audit_path: pathlib.Path,
    expected_epoch: int,
    later_audit_paths: list[pathlib.Path],
) -> dict | None:
    try:
        failed = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    problems = set(failed.get("problems", []))
    checkpoint_ahead_problems = RACE_ONLY_PROBLEMS | {"report_checkpoint_history_equal"}
    allowed = checkpoint_ahead_problems | UNSTABLE_OBSERVATION_PROBLEMS
    checks = failed.get("checks", {})
    unstable_observation = bool(problems & UNSTABLE_OBSERVATION_PROBLEMS) and checks.get(
        "report_checkpoint_history_equal"
    )
    committed_epoch = int(failed.get("committed_epoch", -1))
    checkpoint_epoch = int(failed.get("scheduler_last_epoch", -1))
    report_checks = (
        "history_contiguous",
        "report_epoch_matches",
        "report_steps_match",
        "report_exposures_match",
    )
    checkpoint_ahead = (
        problems <= checkpoint_ahead_problems
        and checks.get("stable_report_double_read")
        and checks.get("stable_checkpoint_during_hash_and_load")
        and checks.get("report_checkpoint_history_equal") is False
        and all(checks.get(name) for name in report_checks)
        and committed_epoch == expected_epoch
        and checkpoint_epoch > committed_epoch
    )
    if (
        failed.get("status") != "failed"
        or not problems
        or not problems <= allowed
        or not (unstable_observation or checkpoint_ahead)
    ):
        return None

    failed_steps = int(failed.get("completed_optimizer_steps", -1))
    failed_exposures = int(failed.get("completed_training_sample_exposures", -1))
    checkpoint_steps = [
        int(value) for value in failed.get("optimizer_state_steps", []) if isinstance(value, int)
    ]
    minimum_confirmed_steps = max([failed_steps, *checkpoint_steps])
    for candidate_path in sorted(later_audit_paths):
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate_epoch = int(candidate.get("committed_epoch", -1))
        checks = candidate.get("checks", {})
        if candidate_epoch <= expected_epoch:
            continue
        if checkpoint_ahead and candidate_epoch < checkpoint_epoch:
            continue
        if candidate.get("status") != "passed" or candidate.get("problems"):
            continue
        if not checks or any(value is not True for value in checks.values()):
            continue
        if any(candidate.get(field) != failed.get(field) for field in AUDIT_IDENTITY_FIELDS):
            continue
        if int(candidate.get("completed_optimizer_steps", -1)) < minimum_confirmed_steps:
            continue
        if int(candidate.get("completed_training_sample_exposures", -1)) < failed_exposures:
            continue
        return {
            "classification": (
                "checkpoint_ahead_of_stable_report_confirmed_by_later_strong_audit"
                if checkpoint_ahead
                else "unstable_observation_confirmed_by_later_strong_audit"
            ),
            "expected_epoch": expected_epoch,
            "confirmed_epoch": candidate_epoch,
            "checkpoint_epoch": checkpoint_epoch if checkpoint_ahead else None,
            "confirmation_audit": str(candidate_path),
            "confirmation_audit_sha256": sha256(candidate_path),
            "failed_audit_sha256": sha256(audit_path),
            "unresolved_scientific_failure": False,
            "failed_artifact_preserved": True,
        }
    return None


def migrate_report_advanced_failures(state: dict, outputs: pathlib.Path) -> None:
    unresolved = []
    transient = state.setdefault("transient_audit_races", [])
    for failure in state.get("failures", []):
        key = failure.get("key", "")
        parts = key.split("/")
        if len(parts) != 4 or not parts[2].startswith("seed-") or not parts[3].startswith("epoch-"):
            unresolved.append(failure)
            continue
        dataset, representation = parts[:2]
        try:
            seed = int(parts[2].removeprefix("seed-"))
            epoch = int(parts[3].removeprefix("epoch-"))
            (report_path,) = (outputs / dataset / representation / f"seed-{seed}").glob(
                "*_report.json"
            )
            audit_path = pathlib.Path(failure["output"])
        except (KeyError, TypeError, ValueError):
            unresolved.append(failure)
            continue
        race = report_advanced_race(report_path, audit_path, epoch)
        if race is None:
            audit_root = audit_path.parent
            later_audits = list(
                audit_root.glob(f"{dataset}_{representation}_seed{seed}_epoch*_strong_audit.json")
            )
            race = later_stable_audit_race(audit_path, epoch, later_audits)
        if race is None:
            unresolved.append(failure)
        else:
            transient.append(
                {
                    **failure,
                    **race,
                    "migrated_at": dt.datetime.now(dt.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
    state["failures"] = unresolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--recovery-pid", type=int)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--selftest-race", nargs=3, metavar=("REPORT", "AUDIT", "EXPECTED_EPOCH"))
    args = parser.parse_args()

    if args.selftest_race:
        report, audit, expected_epoch = args.selftest_race
        result = report_advanced_race(
            pathlib.Path(report), pathlib.Path(audit), int(expected_epoch)
        )
        if result is None:
            raise SystemExit(1)
        print(json.dumps(result, sort_keys=True))
        return
    if args.root is None or args.recovery_pid is None:
        parser.error("--root and --recovery-pid are required outside self-test mode")

    root = pathlib.Path(args.root).resolve()
    repository = root / "recovery/worktrees/formal-020c1de"
    analysis = root / "recovery/worktrees/analysis-020c1de"
    outputs = repository / "outputs/full_validation/auraflow_v03"
    helper = root / "logs/audit_helpers/fieldscope_audit_cell.py"
    terminal_helper = root / "logs/audit_helpers/fieldscope_audit_terminal_cell.py"
    profile = next(
        (root / "recovery/fixed-revision-readout").glob(
            "readout_runtime_profile_*_30c6073ce73f6de74803eb29f497ceedf1ee3cafc488340cf236395824ea42e3.json"
        )
    )
    audit_root = root / "logs/auto-strong-audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    lock_stream = (audit_root / "watcher.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(13) from None
    (audit_root / "watcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    state_path = audit_root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "active"
        state["recovery_pid"] = args.recovery_pid
        state.setdefault("audited", {})
        state.setdefault("terminal_audited", {})
        state.setdefault("failures", [])
        state.setdefault("transient_audit_races", [])
    else:
        state = {
            "schema_version": 1,
            "status": "active",
            "recovery_pid": args.recovery_pid,
            "audited": {},
            "terminal_audited": {},
            "failures": [],
            "transient_audit_races": [],
            "changes_scientific_verdict": False,
        }
    migrate_report_advanced_failures(state, outputs)

    try:
        while alive(args.recovery_pid):
            migrate_report_advanced_failures(state, outputs)
            state["observed_at"] = (
                dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            )
            for report_path in sorted(outputs.glob("*/*/seed-*/*_report.json")):
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    state["failures"].append(
                        {
                            "report": str(report_path),
                            "error": repr(error),
                            "observed_at": state["observed_at"],
                        }
                    )
                    continue
                if report.get("status") != "running" or not report.get("history"):
                    continue
                dataset = report_path.relative_to(outputs).parts[0]
                representation = str(report.get("representation"))
                seed = int(report.get("seed"))
                epoch = int(report["history"][-1]["epoch"])
                key = f"{dataset}/{representation}/seed-{seed}/epoch-{epoch}"
                if key in state["audited"]:
                    continue
                output = (
                    audit_root
                    / f"{dataset}_{representation}_seed{seed}_epoch{epoch}_strong_audit.json"
                )
                command = [
                    str(root / ".venv/bin/python"),
                    str(helper),
                    "--report",
                    str(report_path),
                    "--output",
                    str(output),
                    "--repository",
                    str(repository),
                    "--analysis-repository",
                    str(analysis),
                    "--runtime-profile",
                    str(profile),
                    "--expected-revision",
                    "020c1de567edd88e0eda245fd085335ffe678f47",
                    "--expected-epoch",
                    str(epoch),
                    "--expected-task",
                    str(report.get("task")),
                    "--expected-representation",
                    representation,
                    "--expected-seed",
                    str(seed),
                ]
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                record = {
                    "output": str(output),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "observed_at": state["observed_at"],
                }
                if completed.returncode == 0:
                    state["audited"][key] = record
                else:
                    race = report_advanced_race(report_path, output, epoch)
                    if race is None:
                        state["failures"].append({"key": key, **record})
                    else:
                        state["transient_audit_races"].append({"key": key, **record, **race})

            for report_path in sorted(outputs.glob("*/*/seed-*/*_report.json")):
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if report.get("status") != "passed":
                    continue
                dataset = report_path.relative_to(outputs).parts[0]
                representation = str(report.get("representation"))
                seed = int(report.get("seed"))
                key = f"{dataset}/{representation}/seed-{seed}/terminal"
                if key in state["terminal_audited"]:
                    continue
                test_path = report_path.with_name(
                    report_path.name.replace("_report.json", "_test.json")
                )
                matrix_path = outputs / dataset / "matrix_report.json"
                if not test_path.is_file() or not matrix_path.is_file():
                    continue
                metric = {
                    "classification": "top1",
                    "segmentation": "mean_iou",
                    "depth": "abs_rel",
                }.get(str(report.get("task")))
                if metric is None:
                    state["failures"].append(
                        {
                            "key": key,
                            "error": f"unsupported terminal task: {report.get('task')}",
                            "observed_at": state["observed_at"],
                        }
                    )
                    continue
                output = (
                    audit_root
                    / f"{dataset}_{representation}_seed{seed}_terminal_cell_strong_audit.json"
                )
                command = [
                    str(root / ".venv/bin/python"),
                    str(terminal_helper),
                    "--training-report",
                    str(report_path),
                    "--test-report",
                    str(test_path),
                    "--matrix-report",
                    str(matrix_path),
                    "--output",
                    str(output),
                    "--repository",
                    str(repository),
                    "--analysis-repository",
                    str(analysis),
                    "--runtime-profile",
                    str(profile),
                    "--expected-revision",
                    "020c1de567edd88e0eda245fd085335ffe678f47",
                    "--expected-task",
                    str(report.get("task")),
                    "--expected-representation",
                    representation,
                    "--expected-seed",
                    str(seed),
                    "--expected-epochs",
                    str(report.get("epochs")),
                    "--metric",
                    metric,
                ]
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                record = {
                    "output": str(output),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "observed_at": state["observed_at"],
                }
                if completed.returncode == 0:
                    state["terminal_audited"][key] = record
                else:
                    state["failures"].append({"key": key, **record})
            atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
        state["status"] = "stopped_primary_recovery_exited"
        state["observed_at"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        atomic_json(state_path, state)
    finally:
        try:
            (audit_root / "watcher.pid").unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
