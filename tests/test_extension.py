from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from fieldscope.extension import (
    HIGH_COST_ABLATION_VARIANTS,
    audit_final_evidence,
    audit_imagenet1k_asset,
    build_high_cost_ablation_configs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_build_high_cost_ablation_configs_changes_exactly_one_method_field(
    tmp_path: Path,
) -> None:
    base_path = REPOSITORY_ROOT / "configs" / "model" / "auraflow_v03.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    report = build_high_cost_ablation_configs(base_path, tmp_path)

    assert report["status"] == "passed"
    assert report["variant_count"] == 15
    assert {variant["name"] for variant in report["variants"]} == set(
        HIGH_COST_ABLATION_VARIANTS
    )
    for variant in report["variants"]:
        generated = yaml.safe_load(
            Path(variant["config_path"]).read_text(encoding="utf-8")
        )
        field_name, expected_value = HIGH_COST_ABLATION_VARIANTS[variant["name"]]
        section, key = field_name.split(".", maxsplit=1)
        assert generated[section][key] == expected_value
        for method_section in ("backend", "probe", "tokenizer"):
            for method_key, base_value in base[method_section].items():
                if (method_section, method_key) == (section, key):
                    continue
                assert generated[method_section][method_key] == base_value


def test_audit_imagenet1k_asset_derives_registered_holdout_counts(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    records = []
    label_counts = {0: 6, 1: 4}
    for split, count_per_label in ("train", label_counts), ("val", {0: 1, 1: 1}):
        for label, count in count_per_label.items():
            wnid = f"n{label:08d}"
            directory = extracted / split / wnid
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                relative = f"extracted/{split}/{wnid}/{index}.jpg"
                (tmp_path / relative).touch()
                records.append(
                    {
                        "height": 256,
                        "label": label,
                        "path": relative,
                        "split": split,
                        "width": 256,
                        "wnid": wnid,
                    }
                )
    manifest = tmp_path / "metadata" / "image_manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_bytes = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    summary = tmp_path / "metadata" / "export_summary.json"
    summary.write_text(
        json.dumps(
            {
                "image_size": 256,
                "manifest_sha256": manifest_sha256,
                "split_counts": {"train": 10, "val": 2},
            }
        ),
        encoding="utf-8",
    )

    report = audit_imagenet1k_asset(
        dataset_root=extracted,
        manifest_path=manifest,
        export_summary_path=summary,
        expected_manifest_bytes=len(manifest_bytes),
        expected_manifest_sha256=manifest_sha256,
        expected_source_counts={"train": 10, "val": 2},
        expected_class_count=2,
    )

    assert report["status"] == "passed"
    assert report["fieldscope_split_counts"] == {"train": 8, "val": 2, "test": 2}
    assert report["manifest_unique_paths"] == 12


def test_final_evidence_requires_extension_only_after_positive_main(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provenance = {
        "code_revision": "abc123",
        "code_tree_sha256": "tree123",
        "code_dirty": False,
    }
    monkeypatch.setattr("fieldscope.extension.code_provenance", lambda: provenance)

    def write(name: str, verdict: str) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({**provenance, "verdict": verdict}), encoding="utf-8")
        return path

    negative = write("main-negative", "limited_or_negative")
    negative_causal = write("causal-negative", "limited_or_negative")
    negative_report = audit_final_evidence(
        main_evidence_path=negative,
        causal_evidence_path=negative_causal,
    )
    assert negative_report["verdict"] == "limited_or_negative"

    positive = write("main-positive", "main_tasks_supported_pending_causal_audits")
    positive_causal = write("causal-positive", "supports_core_hypothesis")
    incomplete = audit_final_evidence(
        main_evidence_path=positive,
        causal_evidence_path=positive_causal,
    )
    assert incomplete["verdict"] == "incomplete"

    extension = write("extension", "supported")
    complete = audit_final_evidence(
        main_evidence_path=positive,
        causal_evidence_path=positive_causal,
        extension_evidence_path=extension,
    )
    assert complete["verdict"] == "supports_core_hypothesis_with_scaling_extension"
