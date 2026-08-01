from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_causal_runbook_has_real_multiline_cli_arguments() -> None:
    script = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_causal_validation_after_main.sh"
    ).read_text(encoding="utf-8")
    assert " +    --" not in script
    assert "audit-causal-evidence +" not in script
    for argument in (
        "--config",
        "--dataset voc2012",
        "--storage-policy dense",
        "--main-evidence",
        "--runtime-profile",
        "--random-flow",
        "--spatially-shuffled-probe",
        "--neutral-prompt",
        "--unrelated-prompt",
    ):
        assert argument in script


def test_full_runbooks_continue_after_complete_negative_signal() -> None:
    waiter = (REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_when_ready.sh").read_text(
        encoding="utf-8"
    )
    full = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    for script in (waiter, full):
        assert '"$verdict" == "incomplete"' in script
        assert '"$verdict" != "stop_or_redesign"' in script
        assert '"$verdict" != "proceed"' in script


def test_full_waiter_reuses_only_same_revision_immutable_asset_report() -> None:
    waiter = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_when_ready.sh"
    ).read_text(encoding="utf-8")
    verifier = (
        REPOSITORY_ROOT / "scripts" / "data" / "verify_prepare_imagenet100_remote.sh"
    ).read_text(encoding="utf-8")
    assert "existing_asset_gate_passed" in waiter
    assert 'audit.get("code_revision") != revision' in waiter
    assert 'asset.get("archive_mtime_ns")' not in waiter
    assert '"archive_mtime_ns": stat.st_mtime_ns' in waiter
    assert "reusing same-revision ImageNet-100 asset gate" in waiter
    assert '"archive_mtime_ns": $actual_mtime_ns' in verifier


def test_causal_runbook_runs_after_complete_positive_or_negative_main_result() -> None:
    full = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    causal = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_causal_validation_after_main.sh"
    ).read_text(encoding="utf-8")
    final = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_final_conclusion_after_main.sh"
    ).read_text(encoding="utf-8")
    for script in (full, causal):
        assert '"limited_or_negative"' in script
        assert '"main_tasks_supported_pending_causal_audits"' in script
        assert '== "incomplete"' in script
        assert "echo +" not in script
    assert "exec bash scripts/eval/run_final_conclusion_after_main.sh" in full
    assert "bash scripts/eval/run_causal_validation_after_main.sh" in final
    assert "bash scripts/eval/run_extension_after_main.sh" in final
    assert 'if [[ "$main_verdict" == "main_tasks_supported_pending_causal_audits" ]]' in final
    assert "audit-final-evidence" in final


def test_extension_runbook_is_fail_closed_and_keeps_registered_scope() -> None:
    extension = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_extension_after_main.sh"
    ).read_text(encoding="utf-8")
    assert "audit-imagenet1k-asset" in extension
    assert "--dataset imagenet" in extension
    assert "plan-cache-budget" in extension
    assert "refusing to shrink the registered scope" in extension
    assert "train_imagenet_readout.sh" in extension
    assert "build-extension-ablation-configs" in extension
    assert "audit-extension-evidence" in extension
    assert "--resume" in extension


def test_supervisor_tracks_conditional_final_stages_and_final_verdicts() -> None:
    supervisor = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_validation_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "FIELDSCOPE_IMAGENET1K_ROOT" in supervisor
    assert "run_final_conclusion_after_main.sh" in supervisor
    assert "run_extension_after_main.sh" in supervisor
    assert "final_evidence_decision.json" in supervisor
    assert "supports_core_hypothesis_with_scaling_extension" in supervisor
    assert "supports_core_hypothesis_limited_scaling" in supervisor
    assert "flock -n 9" in supervisor


def test_signal_runbook_runs_runtime_gate_before_real_cache_extraction() -> None:
    signal = (REPOSITORY_ROOT / "scripts" / "eval" / "run_signal_gate_after_gpu.sh").read_text(
        encoding="utf-8"
    )
    runtime_gate = signal.index("runtime-gate")
    first_extract = signal.index("extract-dataset")
    assert runtime_gate < first_extract
    assert "unset FIELDSCOPE_RUNTIME_PROFILE" in signal
    assert 'export FIELDSCOPE_RUNTIME_PROFILE="$PWD/$runtime_profile"' in signal


def test_formal_runbooks_bind_runtime_profile_into_both_evidence_gates() -> None:
    scripts = [
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh",
        REPOSITORY_ROOT / "scripts" / "eval" / "run_causal_validation_after_main.sh",
    ]
    for path in scripts:
        script = path.read_text(encoding="utf-8")
        assert "FIELDSCOPE_RUNTIME_PROFILE" in script
        assert '--runtime-profile "$runtime_profile"' in script


def test_readout_runtime_gate_precedes_formal_readouts_and_binds_every_matrix() -> None:
    signal = (REPOSITORY_ROOT / "scripts" / "eval" / "run_signal_gate_after_gpu.sh").read_text(
        encoding="utf-8"
    )
    full = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    matrix = (REPOSITORY_ROOT / "scripts" / "train" / "run_readout_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert signal.index("run-readout-matrix") < signal.index("readout-runtime-gate")
    assert full.index("missing readout runtime profile") < full.index(
        "training full readout matrix"
    )
    assert 'export FIELDSCOPE_READOUT_RUNTIME_PROFILE="$readout_runtime_profile"' in full
    assert (
        ': "${FIELDSCOPE_READOUT_RUNTIME_PROFILE:?Set FIELDSCOPE_READOUT_RUNTIME_PROFILE}"'
        in matrix
    )
    assert '--readout-runtime-profile "$FIELDSCOPE_READOUT_RUNTIME_PROFILE"' in matrix
    assert full.count('--readout-runtime-profile "$readout_runtime_profile"') == 1
    assert full.index("training full readout matrix") < full.index("audit-full-evidence")


def test_full_runbook_reserves_conservative_checkpoint_budget() -> None:
    full = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    assert "checkpoint_and_report_budget_bytes=21474836480" in full
    assert (
        "additional-required-bytes "
        '"$((sparse_projected_bytes + checkpoint_and_report_budget_bytes))"' in full
    )
