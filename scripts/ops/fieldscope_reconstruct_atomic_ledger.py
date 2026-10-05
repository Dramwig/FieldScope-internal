import argparse
import hashlib
import json
import pathlib
import subprocess

FIXED_REVISION = "020c1de567edd88e0eda245fd085335ffe678f47"
DATASETS = ("imagenet100", "voc2012", "nyuv2", "ade20k")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repository_state(path: pathlib.Path) -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    return revision, not bool(dirty)


def latest_accepted_for_cell(
    state: dict, dataset: str, representation: str, seed: int
) -> tuple[str, dict] | None:
    prefix = f"{dataset}/{representation}/seed-{seed}/epoch-"
    candidates = []
    for key, record in state.get("audited", {}).items():
        if key.startswith(prefix):
            candidates.append((int(key.removeprefix(prefix)), key, record))
    if not candidates:
        return None
    _, key, record = max(candidates)
    return key, record


def reconstruct(root: pathlib.Path) -> dict:
    formal = root / "recovery/worktrees/formal-020c1de"
    analysis = root / "recovery/worktrees/analysis-020c1de"
    output_root = formal / "outputs/full_validation/auraflow_v03"
    state_path = root / "logs/auto-strong-audits/state.json"
    state = read_json(state_path)
    formal_revision, formal_clean = repository_state(formal)
    analysis_revision, analysis_clean = repository_state(analysis)

    problems = []
    if formal_revision != FIXED_REVISION:
        problems.append("formal_revision_mismatch")
    if analysis_revision != FIXED_REVISION:
        problems.append("analysis_revision_mismatch")
    if not formal_clean:
        problems.append("formal_worktree_dirty")
    if not analysis_clean:
        problems.append("analysis_worktree_dirty")
    if state.get("failures"):
        problems.append("unresolved_watcher_failures")

    terminal_cells = {
        dataset: {"count": 0, "optimizer_steps": 0, "training_sample_exposures": 0}
        for dataset in DATASETS
    }
    terminal_keys = set(state.get("terminal_audited", {}))
    terminal_entries = {}
    for key, record in state.get("terminal_audited", {}).items():
        dataset = key.split("/", 1)[0]
        if dataset not in terminal_cells:
            problems.append(f"unregistered_terminal_dataset:{dataset}")
            continue
        audit_path = pathlib.Path(record["output"])
        audit = read_json(audit_path)
        if audit.get("status") != "passed" or audit.get("problems"):
            problems.append(f"invalid_terminal_audit:{key}")
            continue
        steps = int(audit["completed_optimizer_steps"])
        exposures = int(audit["completed_training_sample_exposures"])
        terminal_cells[dataset]["count"] += 1
        terminal_cells[dataset]["optimizer_steps"] += steps
        terminal_cells[dataset]["training_sample_exposures"] += exposures
        terminal_entries[key] = {
            "optimizer_steps": steps,
            "training_sample_exposures": exposures,
            "strong_audit_sha256": sha256(audit_path),
        }

    active_cells = {}
    active_datasets = set()
    for report_path in sorted(output_root.glob("*/*/seed-*/*_report.json")):
        report = read_json(report_path)
        dataset = report_path.relative_to(output_root).parts[0]
        representation = str(report.get("representation"))
        seed = int(report.get("seed"))
        terminal_key = f"{dataset}/{representation}/seed-{seed}/terminal"
        if terminal_key in terminal_keys:
            continue
        accepted = latest_accepted_for_cell(state, dataset, representation, seed)
        if accepted is None:
            continue
        if dataset in active_datasets:
            problems.append(f"multiple_active_cells:{dataset}")
            continue
        key, record = accepted
        audit_path = pathlib.Path(record["output"])
        audit = read_json(audit_path)
        if audit.get("status") != "passed" or audit.get("problems"):
            problems.append(f"invalid_active_audit:{key}")
            continue
        active_datasets.add(dataset)
        active_cells[key] = {
            "optimizer_steps": int(audit["completed_optimizer_steps"]),
            "training_sample_exposures": int(audit["completed_training_sample_exposures"]),
            "strong_audit_sha256": sha256(audit_path),
        }

    terminal_subtotal = {
        "cells": sum(item["count"] for item in terminal_cells.values()),
        "optimizer_steps": sum(item["optimizer_steps"] for item in terminal_cells.values()),
        "training_sample_exposures": sum(
            item["training_sample_exposures"] for item in terminal_cells.values()
        ),
    }
    active_subtotal = {
        "cells": len(active_cells),
        "optimizer_steps": sum(item["optimizer_steps"] for item in active_cells.values()),
        "training_sample_exposures": sum(
            item["training_sample_exposures"] for item in active_cells.values()
        ),
    }
    progress_path = (
        root
        / "FieldScope-internal"
        / "artifacts/reports/2026-08-19_dsw_h200_formal_validation_progress.json"
    )
    registered = read_json(progress_path)["registered_progress"]

    return {
        "schema_version": 2,
        "status": "passed" if not problems else "failed",
        "scope": "independent_reconstruction_of_current_formal_main_matrix_atomic_ledger",
        "observed_at": state.get("observed_at"),
        "machine": "dsw-h200",
        "fixed_revision": FIXED_REVISION,
        "formal_worktree_clean": formal_clean,
        "analysis_worktree_clean": analysis_clean,
        "watcher_state_sha256": sha256(state_path),
        "accepted_intermediate_audits": len(state.get("audited", {})),
        "transient_audit_races": len(state.get("transient_audit_races", [])),
        "unresolved_failures": len(state.get("failures", [])),
        "terminal_cells": terminal_cells,
        "terminal_entries": terminal_entries,
        "active_cells": active_cells,
        "terminal_subtotal": terminal_subtotal,
        "active_subtotal": active_subtotal,
        "reconstructed_global": {
            "optimizer_steps": terminal_subtotal["optimizer_steps"]
            + active_subtotal["optimizer_steps"],
            "training_sample_exposures": terminal_subtotal["training_sample_exposures"]
            + active_subtotal["training_sample_exposures"],
        },
        "registered_global": {
            "optimizer_steps": int(registered["required_optimizer_steps"]),
            "training_sample_exposures": int(registered["required_training_sample_exposures"]),
            "terminal_cells": int(registered["required_matrix_runs"]),
        },
        "problems": problems,
        "execution_complete": (
            terminal_subtotal["cells"] == int(registered["required_matrix_runs"])
            and not active_cells
            and not state.get("failures")
            and not problems
        ),
        "method_effectiveness_conclusion": None,
        "changes_scientific_verdict": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = reconstruct(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
