import json
from pathlib import Path

import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.cache import save_features
from fieldscope.config import ProbeConfig
from fieldscope.experiments import code_provenance, file_sha256
from fieldscope.gates import audit_signal_gate
from fieldscope.response import FieldResponseExtractor

SEEDS = (4121, 7319, 104729)
REPRESENTATIONS = (
    "state",
    "response",
    "full",
    "dit_hidden_local",
    "dit_hidden_attention",
    "response_shuffled",
    "full_shuffled",
)


def _write_cache(cache_dir: Path, split: str) -> None:
    features = FieldResponseExtractor(
        ToyFieldBackend(image_size=16),
        ProbeConfig(
            times=(0.5,),
            num_directions=1,
            graph_grid=(2, 2),
            antithetic_noise=False,
        ),
    ).extract(torch.rand(2, 3, 16, 16), noise_seeds=[11, 22])
    shard_path = cache_dir / "shard-00000000-00000002.pt"
    shard_manifest = save_features(
        shard_path,
        features,
        targets={"classification": torch.tensor([0, 1])},
        sample_ids=[f"{split}-0", f"{split}-1"],
    )
    provenance = code_provenance()
    (cache_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "complete": True,
                "dataset": "synthetic",
                "split": split,
                "num_samples": 2,
                "code_revision": provenance["code_revision"],
                "randomness": {
                    "path_noise": "sample_id_sha256_seeded_v1",
                    "probe_basis": "shared_fixed_seed_v1",
                    "base_seed": 4121,
                },
                "shards": [
                    {
                        "path": shard_path.name,
                        "num_samples": 2,
                        "bytes": shard_path.stat().st_size,
                        "sha256": file_sha256(shard_path),
                        "fingerprint": shard_manifest["fingerprint"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _voc_report(path: Path, *, passing: bool) -> None:
    response_ap = 0.70 if passing else 0.40
    response_auc = 0.75 if passing else 0.45
    means = {
        "response": {
            "boundary_average_precision": response_ap,
            "pairwise_auroc": response_auc,
        },
        "response_shuffled": {
            "boundary_average_precision": 0.50,
            "pairwise_auroc": 0.55,
        },
        "state": {
            "boundary_average_precision": 0.52,
            "pairwise_auroc": 0.56,
        },
        "dit_hidden": {
            "boundary_average_precision": 0.55,
            "pairwise_auroc": 0.58,
        },
        "dit_attention": {
            "boundary_average_precision": 0.54,
            "pairwise_auroc": 0.57,
        },
    }
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "code_revision": code_provenance()["code_revision"],
                "representations": {
                    name: {"means": values} for name, values in means.items()
                },
            }
        ),
        encoding="utf-8",
    )


def _cifar_report(path: Path, *, passing: bool) -> None:
    base = {
        "state": 0.60,
        "dit_hidden_local": 0.62,
        "dit_hidden_attention": 0.61,
        "response_shuffled": 0.60,
        "full_shuffled": 0.605,
        "response": 0.64 if passing else 0.61,
        "full": 0.65 if passing else 0.615,
    }
    runs = [
        {
            "representation": representation,
            "seed": seed,
            "test_metric": value + (seed % 3) * 0.001,
        }
        for representation, value in base.items()
        for seed in SEEDS
    ]
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "metric": "top1",
                "code_revision": code_provenance()["code_revision"],
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, *, passing: bool) -> tuple[list[Path], Path, Path]:
    cache_dirs = []
    for split in ("train", "val", "test", "voc-test"):
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        _write_cache(cache_dir, split)
        cache_dirs.append(cache_dir)
    voc = tmp_path / "voc.json"
    cifar = tmp_path / "cifar.json"
    _voc_report(voc, passing=passing)
    _cifar_report(cifar, passing=passing)
    return cache_dirs, voc, cifar


def test_signal_gate_proceeds_only_when_all_checks_pass(tmp_path: Path) -> None:
    cache_dirs, voc, cifar = _inputs(tmp_path, passing=True)
    report = audit_signal_gate(
        cache_dirs=cache_dirs,
        voc_report_path=voc,
        cifar_matrix_path=cifar,
    )
    assert report["status"] == "passed"
    assert report["verdict"] == "proceed"
    assert report["paper_evidence"] is False
    assert all(check["passed"] for check in report["checks"])


def test_signal_gate_stops_on_weak_signal(tmp_path: Path) -> None:
    cache_dirs, voc, cifar = _inputs(tmp_path, passing=False)
    report = audit_signal_gate(
        cache_dirs=cache_dirs,
        voc_report_path=voc,
        cifar_matrix_path=cifar,
    )
    assert report["status"] == "failed"
    assert report["verdict"] == "stop_or_redesign"
    assert any(not check["passed"] for check in report["checks"])


def test_signal_gate_is_incomplete_for_missing_or_stale_inputs(tmp_path: Path) -> None:
    cache_dirs, voc, cifar = _inputs(tmp_path, passing=True)
    manifest_path = cache_dirs[0] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["randomness"]["path_noise"] = "legacy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    voc.unlink()
    report = audit_signal_gate(
        cache_dirs=cache_dirs,
        voc_report_path=voc,
        cifar_matrix_path=cifar,
    )
    assert report["status"] == "incomplete"
    assert report["verdict"] == "incomplete"
    assert any("randomness.path_noise" in value for value in report["input_problems"])
    assert any("missing" in value for value in report["input_problems"])
