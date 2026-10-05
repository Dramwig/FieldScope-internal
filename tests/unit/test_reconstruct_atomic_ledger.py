import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ops"
        / "fieldscope_reconstruct_atomic_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("fieldscope_reconstruct_atomic_ledger", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_accepted_for_cell_uses_highest_epoch() -> None:
    module = _load_module()
    state = {
        "audited": {
            "nyuv2/z0/seed-1/epoch-2": {"output": "two.json"},
            "nyuv2/z0/seed-1/epoch-10": {"output": "ten.json"},
            "nyuv2/z0/seed-2/epoch-20": {"output": "other.json"},
        }
    }

    key, record = module.latest_accepted_for_cell(state, "nyuv2", "z0", 1)

    assert key.endswith("epoch-10")
    assert record["output"] == "ten.json"


def test_terminal_key_identity_excludes_only_same_cell() -> None:
    terminal_keys = {"nyuv2/random_feature_local/seed-7319/terminal"}

    assert "nyuv2/random_feature_local/seed-7319/terminal" in terminal_keys
    assert "nyuv2/random_feature_local/seed-104729/terminal" not in terminal_keys
