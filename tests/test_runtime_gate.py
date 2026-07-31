import json
from dataclasses import replace

import pytest
import torch

from fieldscope.config import RunConfig, runtime_method_contract_sha256
from fieldscope.contracts import FieldFeatures
from fieldscope.runtime_gate import (
    _compare_features,
    _feature_sha256,
    _synthetic_images,
    load_runtime_gate_report,
)


def _features(value: float) -> FieldFeatures:
    state = torch.full((1, 4, 2), value, dtype=torch.bfloat16)
    response = torch.full((1, 4, 3), value, dtype=torch.bfloat16)
    affinity = torch.eye(4).unsqueeze(0)
    return FieldFeatures(
        state=state,
        response=response,
        affinity=affinity,
        adjacency=affinity.clone(),
        grid_size=(2, 2),
        baselines={"z0": state.clone()},
        graphs={"dit_attention": affinity.clone()},
    )


def test_runtime_gate_hash_and_equivalence_are_tensor_exact() -> None:
    first = _features(1.0)
    repeated = _features(1.0)
    changed = _features(1.0)
    changed.response[0, 0, 0] = torch.tensor(1.5, dtype=torch.bfloat16)
    assert _feature_sha256(first) == _feature_sha256(repeated)
    assert _compare_features(first, repeated)["exact"] is True
    assert _feature_sha256(first) != _feature_sha256(changed)
    comparison = _compare_features(first, changed)
    assert comparison["exact"] is False
    assert comparison["fields"]["response"]["maximum_absolute_error"] == 0.5


def test_runtime_gate_synthetic_images_are_fixed_and_label_free() -> None:
    first = _synthetic_images(32)
    repeated = _synthetic_images(32)
    assert first.shape == (8, 3, 32, 32)
    assert torch.equal(first, repeated)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_runtime_batch_shape_does_not_change_method_contract() -> None:
    config = RunConfig.from_mapping(
        {
            "backend": {
                "name": "auraflow",
                "model_path": "/models/AuraFlow-v0.3",
                "device": "cuda",
            },
            "probe": {"probe_batch_size": 8},
            "runtime": {"batch_size": 2},
        }
    )
    candidate = replace(
        config,
        probe=replace(config.probe, probe_batch_size=64),
        runtime=replace(config.runtime, batch_size=8),
    )
    assert runtime_method_contract_sha256(candidate) == runtime_method_contract_sha256(
        config
    )


def test_existing_runtime_gate_is_reused_only_for_same_clean_revision(
    tmp_path,
    monkeypatch,
) -> None:
    config = RunConfig.from_mapping(
        {
            "backend": {
                "name": "auraflow",
                "model_path": "/models/AuraFlow-v0.3",
                "device": "cuda",
            }
        }
    )
    provenance = {
        "code_revision": "revision",
        "code_tree_sha256": "tree",
        "code_dirty": False,
    }
    monkeypatch.setattr("fieldscope.runtime_gate.code_provenance", lambda: provenance)
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                **provenance,
                "method_runtime_contract_sha256": runtime_method_contract_sha256(
                    config
                ),
                "selected_profile": {
                    "image_batch_size": 2,
                    "probe_batch_size": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_runtime_gate_report(config, path)["status"] == "passed"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["code_revision"] = "stale"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="code_revision"):
        load_runtime_gate_report(config, path)
