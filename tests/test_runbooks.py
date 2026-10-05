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
    assert "imagenet_checkpoint_budget_bytes=8589934592" in extension
    assert (
        '"$((imagenet_projected_bytes + imagenet_checkpoint_budget_bytes))"'
        in extension
    )
    assert "FIELDSCOPE_EXTENSION_RESOURCE_POLL_SECONDS" in extension
    assert "write_combined_extension_budget" in extension
    assert 'if [[ "$fits" == "true" ]]' in extension
    assert "waiting ${resource_poll_seconds}s" in extension
    assert 'tracked_config="${FIELDSCOPE_EXTRACTION_CONFIG:' in extension
    assert 'readout_config="${FIELDSCOPE_READOUT_CONFIG:' in extension
    assert 'export FIELDSCOPE_EXTRACTION_CONFIG="$tracked_config"' in extension
    assert 'export FIELDSCOPE_READOUT_CONFIG="$readout_config"' in extension


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


def test_extract_and_readout_runbooks_have_independent_config_overrides() -> None:
    extraction = (REPOSITORY_ROOT / "scripts" / "eval" / "run_dataset_extract.sh").read_text(
        encoding="utf-8"
    )
    matrix = (REPOSITORY_ROOT / "scripts" / "train" / "run_readout_matrix.sh").read_text(
        encoding="utf-8"
    )
    full = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    assert "FIELDSCOPE_EXTRACTION_CONFIG" in extraction
    assert "FIELDSCOPE_READOUT_CONFIG" in matrix
    assert 'export FIELDSCOPE_EXTRACTION_CONFIG="$tracked_config"' in full
    assert 'export FIELDSCOPE_READOUT_CONFIG="$readout_config"' in full


def test_fixed_revision_recovery_is_fail_closed_and_content_addressed() -> None:
    script = (
        REPOSITORY_ROOT
        / "scripts"
        / "eval"
        / "run_fixed_revision_readout_recovery.sh"
    ).read_text(encoding="utf-8")
    for contract in (
        'source_tree_sha256="6ef1305effef91b29d432c9d2c1505b4726645531adad72f60299168d7119c4d"',
        'config_sha256="0be38fa17ba13c1a9e0b248642c85832cd8cf7e63afe2b799fe89f0f0ea98ed1"',
        'readout_contract_sha256="4422430edf4eb8cdb20a99db8cdf7bac53c4cb934f0b09ba8dd3c21aee3facb8"',
        "fixed-revision recovery requires exactly 0 GiB shared readout cache",
        "symbolic-ref -q --short HEAD",
        "fixed-revision recovery requires a clean worktree",
        "verify_no_formal_worker",
        "refusing duplicate fixed-revision recovery",
        '[[ "$parent_pid" != "$$" ]] || continue',
        "verify_required_artifacts",
        "same-revision prerequisite provenance mismatch",
        "flock -n 9",
        "waiting for genuinely free GPU",
        "readout config SHA-256 mismatch",
        "cache shard SHA-256 mismatch",
        "cache shard gap or overlap",
        "orphaned cache shards present",
        "temporary cache files present",
        'readout_gate_cache_root="${FIELDSCOPE_READOUT_GATE_CACHE_ROOT:-$FIELDSCOPE_DATASETS_ROOT/feature_cache/signal_final512}"',
        'readout_gate_cache_contract_sha256="30c6073ce73f6de74803eb29f497ceedf1ee3cafc488340cf236395824ea42e3"',
        "verify_readout_gate_caches",
        "readout gate manifest SHA-256 mismatch",
        "orphaned readout gate shards present",
        '--train-cache-dir "$readout_gate_cache_root/cifar10_train"',
        '--val-cache-dir "$readout_gate_cache_root/cifar10_val"',
        '--test-cache-dir "$readout_gate_cache_root/cifar10_test"',
        "readout runtime profile used the wrong gate cache",
        "readout runtime profile changed the registered fast-path threshold",
        'manifest_registry="$external_root/recovery_registry_${FIELDSCOPE_EXPECTED_REVISION}_${config_sha256}_${readout_gate_cache_contract_sha256}.json"',
        "cache manifest registry revision mismatch",
        'lock_file="$log_root/full_validation_recovery_${FIELDSCOPE_EXPECTED_REVISION}.lock"',
        "verify_cache_manifests imagenet100",
        "verify_cache_manifests all",
        'export FIELDSCOPE_EXTRACTION_CONFIG="$tracked_config_path"',
        'export FIELDSCOPE_READOUT_CONFIG="$config_path"',
        "readout-runtime-gate",
        "run_final_conclusion_after_main.sh",
    ):
        assert contract in script
    assert "kill " not in script
    assert "git checkout" not in script
    assert "git switch" not in script
    assert "rm -rf" not in script
    assert script.index("wait_for_free_gpu") < script.index("verify_cache_manifests imagenet100")
    assert script.count("wait_for_free_gpu") >= 3


def test_fixed_revision_recovery_uses_config_split_final_wrappers() -> None:
    recovery = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_fixed_revision_readout_recovery.sh"
    ).read_text(encoding="utf-8")
    final = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_fixed_revision_final_recovery.sh"
    ).read_text(encoding="utf-8")
    extension = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_fixed_revision_extension_recovery.sh"
    ).read_text(encoding="utf-8")
    assert 'FIELDSCOPE_CONFIG="$tracked_config_path"' in recovery
    assert 'FIELDSCOPE_CONFIG="$config_path"' in recovery
    assert "run_fixed_revision_final_recovery.sh" in recovery
    assert "run_fixed_revision_extension_recovery.sh" in final
    assert 'FIELDSCOPE_RUNTIME_PROFILE="$main_runtime_profile"' in extension
    assert 'FIELDSCOPE_READOUT_CONFIG="$readout_config"' in extension
    assert 'code_tree_sha256' in final
    assert (
        'env -u FIELDSCOPE_RUNTIME_PROFILE "$python_bin" '
        '-m fieldscope.cli readout-runtime-gate'
    ) in recovery
    assert "audit-final-evidence" in final


def test_full_runbook_reserves_conservative_checkpoint_budget() -> None:
    full = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    assert "checkpoint_and_report_budget_bytes=21474836480" in full
    assert (
        "additional-required-bytes "
        '"$((sparse_projected_bytes + checkpoint_and_report_budget_bytes))"' in full
    )


def test_supervised_error_waiter_runs_all_tasks_only_after_final_decision() -> None:
    script = (
        REPOSITORY_ROOT
        / "scripts"
        / "eval"
        / "run_supervised_error_analysis_after_final.sh"
    ).read_text(encoding="utf-8")
    assert "final_decision_ready" in script
    assert "waiting for complete formal final decision" in script
    assert "run-readout-error-analysis" in script
    assert "FIELDSCOPE_EXPECTED_SOURCE_REVISION" in script
    assert "FIELDSCOPE_EXPECTED_ANALYZER_REVISION" in script
    assert (
        'output_root="${FIELDSCOPE_ERROR_ANALYSIS_OUTPUT_ROOT:-'
        '$FIELDSCOPE_ROOT/recovery/analysis-output/formal-supervised-errors}"'
        in script
    )
    assert "$FIELDSCOPE_ROOT/../" not in script
    assert "flock -n 9" in script
    assert "changes_main_verdict" in script
    assert 'for dataset in imagenet100 voc2012 ade20k nyuv2' in script
    assert 'len(payload.get("input_reports", [])) != 33' in script
    assert 'len(payload.get("comparisons", {})) != 12' in script
    assert 'output_root / "registry.json"' in script
    assert "audit-research-completion" in script
    assert 'completion_audit="$output_root/completion_audit.json"' in script
    assert 'payload.get("execution_complete") is not True' in script
