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
        "--random-flow",
        "--spatially-shuffled-probe",
        "--neutral-prompt",
        "--unrelated-prompt",
    ):
        assert argument in script


def test_full_runbooks_continue_after_complete_negative_signal() -> None:
    waiter = (
        REPOSITORY_ROOT / "scripts" / "eval" / "run_full_validation_when_ready.sh"
    ).read_text(encoding="utf-8")
    full = (
        REPOSITORY_ROOT
        / "scripts"
        / "eval"
        / "run_full_validation_after_signal_gate.sh"
    ).read_text(encoding="utf-8")
    for script in (waiter, full):
        assert '"$verdict" == "incomplete"' in script
        assert '"$verdict" != "stop_or_redesign"' in script
        assert '"$verdict" != "proceed"' in script
