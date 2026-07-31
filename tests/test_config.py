import json
from pathlib import Path

import pytest

from fieldscope.config import (
    RunConfig,
    apply_runtime_profile,
    load_config,
    runtime_method_contract_sha256,
    runtime_profile_identity,
)


def test_default_config_is_valid() -> None:
    config = RunConfig()
    config.validate()
    assert config.probe.times == (0.2, 0.5, 0.8)


def test_load_yaml_expands_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIELDSCOPE_TEST_ROOT", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
backend:
  name: toy
  image_size: 32
probe:
  times: [0.25, 0.75]
  graph_grid: [4, 4]
runtime:
  output_dir: ${FIELDSCOPE_TEST_ROOT}/outputs
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.runtime.output_dir == str(tmp_path / "outputs")
    assert config.probe.graph_grid == (4, 4)


def test_invalid_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        RunConfig.from_mapping({"probe": {"times": [0.0, 0.5]}})


def test_auraflow_variant_is_preserved() -> None:
    config = RunConfig.from_mapping(
        {
            "backend": {
                "name": "auraflow",
                "model_path": "/models/AuraFlow-v0.3",
                "variant": "fp16",
            }
        }
    )
    assert config.backend.variant == "fp16"


def test_random_transformer_is_restricted_to_auraflow() -> None:
    with pytest.raises(ValueError, match="only supported by auraflow"):
        RunConfig.from_mapping({"backend": {"random_transformer": True}})


def test_random_transformer_seed_is_preserved() -> None:
    config = RunConfig.from_mapping(
        {
            "backend": {
                "name": "auraflow",
                "model_path": "/models/AuraFlow-v0.3",
                "random_transformer": True,
                "random_transformer_seed": 17,
            }
        }
    )
    assert config.backend.random_transformer
    assert config.backend.random_transformer_seed == 17


def test_registered_causal_configs_change_only_the_registered_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIELDSCOPE_CHECKPOINTS_ROOT", "/checkpoints")
    repository_root = Path(__file__).resolve().parents[1]
    main = load_config(repository_root / "configs" / "model" / "auraflow_v03.yaml")
    variants = {
        "random_flow": load_config(
            repository_root / "configs" / "ablation" / "random_auraflow_v03.yaml"
        ),
        "spatially_shuffled_probe": load_config(
            repository_root
            / "configs"
            / "ablation"
            / "auraflow_spatially_shuffled_probe.yaml"
        ),
        "neutral_prompt": load_config(
            repository_root
            / "configs"
            / "ablation"
            / "auraflow_neutral_prompt.yaml"
        ),
        "unrelated_prompt": load_config(
            repository_root
            / "configs"
            / "ablation"
            / "auraflow_unrelated_prompt.yaml"
        ),
    }
    shared_backend_fields = (
        "name",
        "model_path",
        "variant",
        "device",
        "dtype",
        "image_size",
        "max_sequence_length",
        "local_files_only",
        "offload_text_encoder",
    )
    shared_probe_fields = (
        "times",
        "num_directions",
        "eta",
        "difference",
        "graph_grid",
        "topk",
        "local_radius",
        "probe_batch_size",
        "antithetic_noise",
        "hidden_baseline_dim",
        "seed",
    )
    for variant in variants.values():
        for field in shared_backend_fields:
            assert getattr(variant.backend, field) == getattr(main.backend, field)
        for field in shared_probe_fields:
            assert getattr(variant.probe, field) == getattr(main.probe, field)
        assert variant.tokenizer == main.tokenizer
        assert variant.runtime.batch_size == main.runtime.batch_size
        assert variant.runtime.cache_shard_size == main.runtime.cache_shard_size
        assert variant.runtime.deterministic == main.runtime.deterministic

    assert variants["random_flow"].backend.random_transformer is True
    assert variants["random_flow"].backend.random_transformer_seed == 104729
    assert variants["random_flow"].backend.prompt == main.backend.prompt
    assert variants["random_flow"].probe.probe_type == main.probe.probe_type

    shuffled = variants["spatially_shuffled_probe"]
    assert shuffled.backend == main.backend
    assert shuffled.probe.probe_type == "spatially_shuffled"

    neutral = variants["neutral_prompt"]
    unrelated = variants["unrelated_prompt"]
    assert neutral.backend.random_transformer is False
    assert unrelated.backend.random_transformer is False
    assert neutral.backend.prompt == "a neutral photograph"
    assert unrelated.backend.prompt == "an unrelated scene"
    assert neutral.probe == main.probe
    assert unrelated.probe == main.probe


def test_runtime_profile_changes_only_registered_batch_shapes(tmp_path: Path) -> None:
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
    profile = tmp_path / "runtime.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "method_runtime_contract_sha256": runtime_method_contract_sha256(
                    config
                ),
                "selected_profile": {
                    "image_batch_size": 4,
                    "probe_batch_size": 32,
                },
            }
        ),
        encoding="utf-8",
    )
    selected = apply_runtime_profile(config, profile)
    assert selected.runtime.batch_size == 4
    assert selected.probe.probe_batch_size == 32
    assert selected.backend == config.backend
    assert selected.tokenizer == config.tokenizer
    assert runtime_profile_identity(profile)["sha256"]


def test_runtime_profile_rejects_unregistered_or_mismatched_selection(
    tmp_path: Path,
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
    profile = tmp_path / "runtime.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "method_runtime_contract_sha256": "tampered",
                "selected_profile": {
                    "image_batch_size": 3,
                    "probe_batch_size": 24,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unregistered"):
        runtime_profile_identity(profile)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["selected_profile"] = {
        "image_batch_size": 2,
        "probe_batch_size": 16,
    }
    profile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="method contract"):
        apply_runtime_profile(config, profile)
