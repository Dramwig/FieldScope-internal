from pathlib import Path

import pytest

from fieldscope.config import RunConfig, load_config


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

