"""Immutable asset audits for formal FieldScope experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fieldscope.config import RunConfig
from fieldscope.experiments import code_provenance, file_sha256

AURAFLOW_HUB_REVISION = "2cd8588f04c886002be4571697d84654a50e3af3"
AURAFLOW_FP16_FILES = {
    "text_encoder/model.fp16.safetensors": (
        "decf9b70814ed5e9965bfca9fbd0483462e2bf743790663025b7742f8c014c72"
    ),
    "transformer/diffusion_pytorch_model-00001-of-00002.fp16.safetensors": (
        "6a40b011f287452dbca80face78e667055904c5ad97eb2097ade3200259b2203"
    ),
    "transformer/diffusion_pytorch_model-00002-of-00002.fp16.safetensors": (
        "e46f7a419b9531c293398d8ac012e10fe99f98228bdd71ce0b7dbb51688a36c0"
    ),
    "vae/diffusion_pytorch_model.fp16.safetensors": (
        "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68"
    ),
    "model_index.json": (
        "7ffee10ae3277f136a341bac357405d615961f7c949f7c2e231075f8c7d18541"
    ),
    "scheduler/scheduler_config.json": (
        "484263e607c38c2940d914d53fbf80c6391e180009846689bb2da970d5605f14"
    ),
    "text_encoder/config.json": (
        "4fd2fce0b91e37ddfd44b4afec50a4bb8a433e348d15c30e4c412e9556490edc"
    ),
    "tokenizer/added_tokens.json": (
        "ea5a91a3234f66ea642c8e672d67f0f493759a9bee6910ae304ea9b9492118b5"
    ),
    "tokenizer/special_tokens_map.json": (
        "d8927854091fd9c46871b2e2a077f7509f1231484e5e66edcb3026f9074f3280"
    ),
    "tokenizer/tokenizer.json": (
        "f6648b38637e86d7c227794f71fb036207b4fdacc760c9c03da71a3fcbe14c49"
    ),
    "tokenizer/tokenizer.model": (
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
    ),
    "tokenizer/tokenizer_config.json": (
        "fa2f9e9fd7368f824e2bf44a60761055534b5c8039f2735ad61305bef57d2853"
    ),
    "transformer/config.json": (
        "f80724bd8e8ba2ee0d723c1eb68589bc6d497b2a247af11c23043eb791f7a562"
    ),
    "transformer/diffusion_pytorch_model.safetensors.fp16.index.json": (
        "6791b87e20fd2ae14e7c0f42edb299f232cb941f6363a38de06d60a689067692"
    ),
    "transformer/diffusion_pytorch_model.safetensors.index.fp16.json": (
        "6791b87e20fd2ae14e7c0f42edb299f232cb941f6363a38de06d60a689067692"
    ),
    "vae/config.json": (
        "bf2daa82c70fb8437853a3e9cd901c775a3f7a9af9b49227663ad86e93be3bc7"
    ),
}


def audit_backbone_assets(
    config: RunConfig,
    *,
    expected_files: Mapping[str, str] = AURAFLOW_FP16_FILES,
    expected_hub_revision: str = AURAFLOW_HUB_REVISION,
) -> dict[str, Any]:
    """Verify the exact frozen AuraFlow weights used for formal extraction."""

    problems: list[str] = []
    if config.backend.name != "auraflow":
        problems.append("formal backbone asset audit requires the auraflow backend")
    if config.backend.variant != "fp16":
        problems.append("formal backbone asset audit requires the fp16 variant")
    if config.backend.random_transformer:
        problems.append("formal backbone asset audit rejects a random transformer")
    if not config.backend.local_files_only:
        problems.append("formal backbone asset audit requires local_files_only")
    model_root = Path(config.backend.model_path or "").resolve()
    if not model_root.is_dir():
        problems.append(f"missing model root {model_root}")

    files: dict[str, Any] = {}
    for relative_path, expected_sha256 in expected_files.items():
        path = model_root / relative_path
        if not path.is_file():
            files[relative_path] = {
                "path": str(path),
                "expected_sha256": expected_sha256,
                "actual_sha256": None,
                "bytes": None,
                "matches": False,
            }
            problems.append(f"missing backbone asset {path}")
            continue
        actual_sha256 = file_sha256(path)
        matches = actual_sha256 == expected_sha256
        files[relative_path] = {
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "bytes": path.stat().st_size,
            "matches": matches,
        }
        if not matches:
            problems.append(f"backbone asset SHA-256 mismatch {path}")

    return {
        "status": "passed" if not problems else "failed",
        **code_provenance(),
        "backend": "auraflow",
        "model_root": str(model_root),
        "variant": config.backend.variant,
        "random_transformer": config.backend.random_transformer,
        "local_files_only": config.backend.local_files_only,
        "expected_hub_revision": expected_hub_revision,
        "files": files,
        "problems": problems,
    }
