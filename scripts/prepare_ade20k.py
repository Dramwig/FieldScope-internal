"""Verify and extract the fixed ADEChallengeData2016 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

EXPECTED_SHA256 = "a4a2860390141240c3c05981b5c0084c1d773b2a8c39ac14209da40602cc11ea"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_ade20k(archive: Path, output: Path) -> dict[str, Any]:
    actual_hash = _sha256(archive)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"ADE20K SHA-256 mismatch: {actual_hash}")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(output)
    counts = {
        split: len(list((output / "images" / split).glob("*.jpg")))
        for split in ("training", "validation")
    }
    if counts != {"training": 20210, "validation": 2000}:
        raise ValueError(f"Unexpected ADE20K split counts: {counts}")
    report = {
        "status": "passed",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": actual_hash,
        "output": str(output.resolve()),
        "image_counts": counts,
    }
    (output / "fieldscope_preparation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_ade20k(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
