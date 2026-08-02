import json
from pathlib import Path

import pytest

from fieldscope.completion import _source_tree_sha256, audit_research_completion
from fieldscope.experiments import file_sha256


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_negative_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path | dict]:
    source = tmp_path / "formal"
    package = source / "src" / "fieldscope"
    package.mkdir(parents=True)
    (package / "formal.py").write_text("FORMAL = True\n", encoding="utf-8")
    revision = "formal-revision"
    tree = _source_tree_sha256(source)
    source_provenance = {
        "code_revision": revision,
        "code_tree_sha256": tree,
        "code_dirty": False,
    }
    analyzer = {
        "code_revision": "analyzer-revision",
        "code_tree_sha256": "analyzer-tree",
        "code_dirty": False,
    }
    monkeypatch.setattr("fieldscope.completion.code_provenance", lambda: analyzer)
    monkeypatch.setattr(
        "fieldscope.completion.audit_source_repository",
        lambda source_revision, *, repository_root: {
            "status": "passed",
            "path": str(repository_root),
            "source_revision": source_revision,
            "actual_revision": source_revision,
            "dirty_paths": [],
        },
    )

    tasks = {}
    for dataset in ("imagenet100", "voc2012", "ade20k", "nyuv2"):
        tasks[dataset] = {
            "run_count": 60,
            "required_optimizer_steps": 10,
            "reported_completed_optimizer_steps": 10,
            "required_training_sample_exposures": 100,
            "reported_completed_training_sample_exposures": 100,
        }
    main = _write_json(
        source / "outputs" / "main.json",
        {
            **source_provenance,
            "status": "failed",
            "verdict": "limited_or_negative",
            "problems": [],
            "tasks": tasks,
            "training_execution": {
                "required_optimizer_steps": 40,
                "reported_completed_optimizer_steps": 40,
                "required_training_sample_exposures": 400,
                "reported_completed_training_sample_exposures": 400,
            },
            "unsupervised": {"num_images": 1449},
        },
    )
    sources = {}
    for name in (
        "empty_prompt",
        "random_flow",
        "spatially_shuffled_probe",
        "neutral_prompt",
        "unrelated_prompt",
    ):
        report = _write_json(source / "outputs" / "causal" / f"{name}.json", {})
        cache = source / "outputs" / "causal" / f"{name}-cache"
        cache.mkdir()
        sources[name] = {
            "path": str(report.relative_to(source)),
            "cache_dir": str(cache.relative_to(source)),
        }
    causal = _write_json(
        source / "outputs" / "causal.json",
        {
            **source_provenance,
            "status": "failed",
            "verdict": "limited_or_negative",
            "problems": [],
            "sources": sources,
        },
    )
    final = _write_json(
        source / "outputs" / "final.json",
        {
            **source_provenance,
            "status": "failed",
            "verdict": "limited_or_negative",
            "problems": [],
            "main_evidence": str(main.relative_to(source)),
            "main_verdict": "limited_or_negative",
            "causal_evidence": str(causal.relative_to(source)),
            "causal_verdict": "limited_or_negative",
            "extension_evidence": None,
            "extension_verdict": None,
        },
    )

    analysis = tmp_path / "analysis"
    task_types = {
        "imagenet100": "classification",
        "voc2012": "segmentation",
        "ade20k": "segmentation",
        "nyuv2": "depth",
    }
    registry_tasks = {}
    summaries = {}
    for dataset, task in task_types.items():
        summary = _write_json(
            analysis / dataset / "summary.json",
            {
                "status": "passed",
                "changes_main_verdict": False,
                "formal_source": {
                    "code_revision": revision,
                    "code_tree_sha256": tree,
                },
                "task": task,
                "input_reports": [{} for _ in range(33)],
                "comparisons": {f"comparison-{index}": {} for index in range(12)},
                "analyzer": analyzer,
            },
        )
        summaries[dataset] = summary
        registry_tasks[dataset] = {
            "path": str(summary),
            "sha256": file_sha256(summary),
        }
    registry = _write_json(
        analysis / "registry.json",
        {
            "schema_version": 1,
            "status": "passed",
            "evidence_scope": "prospective_secondary_supervised_error_analysis",
            "changes_main_verdict": False,
            "formal_source_revision": revision,
            "formal_final_decision": {
                "path": str(final.resolve()),
                "sha256": file_sha256(final),
            },
            "tasks": registry_tasks,
            "analyzer": analyzer,
        },
    )
    return {
        "source": source,
        "main": main,
        "causal": causal,
        "final": final,
        "registry": registry,
        "summaries": summaries,
        "source_provenance": source_provenance,
    }


def test_completion_audit_passes_complete_negative_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _complete_negative_bundle(tmp_path, monkeypatch)

    report = audit_research_completion(
        source_repository_root=bundle["source"],
        final_decision_path=bundle["final"],
        error_registry_path=bundle["registry"],
    )

    assert report["status"] == "passed"
    assert report["execution_complete"] is True
    assert report["scientific_verdict"] == "limited_or_negative"
    assert report["changes_scientific_verdict"] is False
    assert report["problems"] == []
    assert len(report["secondary_error_analysis"]["summaries"]) == 4


def test_completion_audit_rejects_tampered_secondary_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _complete_negative_bundle(tmp_path, monkeypatch)
    summary = bundle["summaries"]["voc2012"]
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = audit_research_completion(
        source_repository_root=bundle["source"],
        final_decision_path=bundle["final"],
        error_registry_path=bundle["registry"],
    )

    assert report["status"] == "incomplete"
    assert report["execution_complete"] is False
    assert "secondary summary SHA-256 mismatch for voc2012" in report["problems"]


def test_completion_audit_requires_extension_after_positive_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _complete_negative_bundle(tmp_path, monkeypatch)
    main = json.loads(bundle["main"].read_text(encoding="utf-8"))
    main.update(status="passed", verdict="main_tasks_supported_pending_causal_audits")
    _write_json(bundle["main"], main)
    causal = json.loads(bundle["causal"].read_text(encoding="utf-8"))
    causal.update(status="passed", verdict="supports_core_hypothesis")
    _write_json(bundle["causal"], causal)
    final = json.loads(bundle["final"].read_text(encoding="utf-8"))
    final.update(
        status="passed",
        verdict="supports_core_hypothesis_limited_scaling",
        main_verdict="main_tasks_supported_pending_causal_audits",
        causal_verdict="supports_core_hypothesis",
    )
    _write_json(bundle["final"], final)
    registry = json.loads(bundle["registry"].read_text(encoding="utf-8"))
    registry["formal_final_decision"]["sha256"] = file_sha256(bundle["final"])
    _write_json(bundle["registry"], registry)

    report = audit_research_completion(
        source_repository_root=bundle["source"],
        final_decision_path=bundle["final"],
        error_registry_path=bundle["registry"],
    )

    assert report["status"] == "incomplete"
    assert report["execution_complete"] is False
    assert "missing extension evidence path" in report["problems"]


def test_completion_audit_passes_complete_positive_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _complete_negative_bundle(tmp_path, monkeypatch)
    main = json.loads(bundle["main"].read_text(encoding="utf-8"))
    main.update(status="passed", verdict="main_tasks_supported_pending_causal_audits")
    _write_json(bundle["main"], main)
    causal = json.loads(bundle["causal"].read_text(encoding="utf-8"))
    causal.update(status="passed", verdict="supports_core_hypothesis")
    _write_json(bundle["causal"], causal)
    source_provenance = bundle["source_provenance"]
    variants = {f"variant-{index}": {} for index in range(15)}
    extension = _write_json(
        bundle["source"] / "outputs" / "extension.json",
        {
            **source_provenance,
            "status": "passed",
            "verdict": "supported",
            "problems": [],
            "imagenet1k": {"matrix": {"run_count": 60}},
            "high_cost_ablations": {
                "required_variant_count": 15,
                "sources": variants,
                "comparisons": variants,
            },
        },
    )
    final = json.loads(bundle["final"].read_text(encoding="utf-8"))
    final.update(
        status="passed",
        verdict="supports_core_hypothesis_with_scaling_extension",
        main_verdict="main_tasks_supported_pending_causal_audits",
        causal_verdict="supports_core_hypothesis",
        extension_evidence=str(extension.relative_to(bundle["source"])),
        extension_verdict="supported",
    )
    _write_json(bundle["final"], final)
    registry = json.loads(bundle["registry"].read_text(encoding="utf-8"))
    registry["formal_final_decision"]["sha256"] = file_sha256(bundle["final"])
    _write_json(bundle["registry"], registry)

    report = audit_research_completion(
        source_repository_root=bundle["source"],
        final_decision_path=bundle["final"],
        error_registry_path=bundle["registry"],
    )

    assert report["status"] == "passed"
    assert report["execution_complete"] is True
    assert (
        report["scientific_verdict"]
        == "supports_core_hypothesis_with_scaling_extension"
    )
