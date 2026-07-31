from pathlib import Path
from typing import Any

from fieldscope.assets import audit_backbone_assets
from fieldscope.config import RunConfig
from fieldscope.experiments import file_sha256


def _config(model_root: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "backend": {
                "name": "auraflow",
                "model_path": str(model_root),
                "variant": "fp16",
                "device": "cpu",
            }
        }
    )


def test_backbone_asset_audit_verifies_registered_hashes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"formal weights")
    monkeypatch.setattr(
        "fieldscope.assets.code_provenance",
        lambda: {
            "code_revision": "revision",
            "code_dirty": False,
            "code_tree_sha256": "tree",
        },
    )
    report = audit_backbone_assets(
        _config(tmp_path),
        expected_files={asset.name: file_sha256(asset)},
        expected_hub_revision="hub-revision",
    )
    assert report["status"] == "passed"
    assert report["expected_hub_revision"] == "hub-revision"
    asset.write_bytes(b"changed")
    report = audit_backbone_assets(
        _config(tmp_path),
        expected_files={asset.name: "0" * 64},
    )
    assert report["status"] == "failed"
    assert "SHA-256 mismatch" in report["problems"][0]
