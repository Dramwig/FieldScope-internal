"""Extract a deterministic full ImageNet-100 subset from ILSVRC2012 archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

from scipy.io import loadmat


def _write_hashed(source: BinaryIO, destination: Path) -> tuple[int, str]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
            digest.update(block)
            size += len(block)
    os.replace(temporary, destination)
    return size, digest.hexdigest()


def _hash_existing(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _devkit_metadata(
    devkit_archive: Path,
) -> tuple[dict[int, str], list[int], dict[str, int]]:
    with tarfile.open(devkit_archive, "r:gz") as bundle:
        metadata_file = bundle.extractfile(
            "ILSVRC2012_devkit_t12/data/meta.mat"
        )
        ground_truth_file = bundle.extractfile(
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        )
        if metadata_file is None or ground_truth_file is None:
            raise ValueError("ImageNet devkit is missing metadata files")
        metadata = loadmat(io.BytesIO(metadata_file.read()), squeeze_me=True)
        rows = metadata["synsets"]
        id_to_wnid = {
            int(row["ILSVRC2012_ID"]): str(row["WNID"])
            for row in rows
            if int(row["num_children"]) == 0
        }
        train_counts = {
            str(row["WNID"]): int(row["num_train_images"])
            for row in rows
            if int(row["num_children"]) == 0
        }
        ground_truth = [
            int(line)
            for line in ground_truth_file.read().decode("utf-8").splitlines()
            if line.strip()
        ]
    if len(id_to_wnid) != 1000 or len(ground_truth) != 50000:
        raise ValueError("Unexpected ILSVRC2012 devkit class or validation count")
    return id_to_wnid, ground_truth, train_counts


def prepare_imagenet100(
    train_archive: Path,
    val_archive: Path,
    devkit_archive: Path,
    output: Path,
) -> dict[str, Any]:
    id_to_wnid, ground_truth, train_counts = _devkit_metadata(devkit_archive)
    selected_classes = sorted(id_to_wnid.values())[:100]
    selected_set = set(selected_classes)
    records: list[dict[str, Any]] = []

    with tarfile.open(train_archive, "r:") as outer:
        class_members = {
            Path(member.name).stem: member
            for member in outer.getmembers()
            if member.isfile() and member.name.endswith(".tar")
        }
        missing = sorted(selected_set - set(class_members))
        if missing:
            raise ValueError(f"Missing selected training class archives: {missing[:5]}")
        for wnid in selected_classes:
            class_root = output / "train" / wnid
            class_root.mkdir(parents=True, exist_ok=True)
            class_file = outer.extractfile(class_members[wnid])
            if class_file is None:
                raise ValueError(f"Cannot read training archive for {wnid}")
            with tarfile.open(fileobj=class_file, mode="r:") as inner:
                for member in inner:
                    if not member.isfile():
                        continue
                    source = inner.extractfile(member)
                    if source is None:
                        raise ValueError(f"Cannot read {member.name}")
                    destination = class_root / Path(member.name).name
                    if destination.is_file():
                        size, sha256 = _hash_existing(destination)
                    else:
                        size, sha256 = _write_hashed(source, destination)
                    records.append(
                        {
                            "split": "train",
                            "class": wnid,
                            "path": destination.relative_to(output).as_posix(),
                            "bytes": size,
                            "sha256": sha256,
                        }
                    )

    with tarfile.open(val_archive, "r:") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            index = int(Path(member.name).stem.rsplit("_", 1)[-1])
            wnid = id_to_wnid[ground_truth[index - 1]]
            if wnid not in selected_set:
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name}")
            class_root = output / "val" / wnid
            class_root.mkdir(parents=True, exist_ok=True)
            destination = class_root / Path(member.name).name
            if destination.is_file():
                size, sha256 = _hash_existing(destination)
            else:
                size, sha256 = _write_hashed(source, destination)
            records.append(
                {
                    "split": "val",
                    "class": wnid,
                    "path": destination.relative_to(output).as_posix(),
                    "bytes": size,
                    "sha256": sha256,
                }
            )

    manifest_path = output / "image_manifest.jsonl"
    manifest_text = "".join(
        json.dumps(record, sort_keys=True) + "\n"
        for record in sorted(records, key=lambda item: item["path"])
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    actual_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val")
    }
    expected_train = sum(train_counts[wnid] for wnid in selected_classes)
    if actual_counts != {"train": expected_train, "val": 5000}:
        raise ValueError(
            f"ImageNet-100 counts {actual_counts} do not match "
            f"expected train={expected_train}, val=5000"
        )
    report = {
        "status": "passed",
        "selection": "lexicographically_first_100_wordnet_ids",
        "classes": selected_classes,
        "counts": actual_counts,
        "total_image_bytes": sum(record["bytes"] for record in records),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
        "sources": {
            "train": str(train_archive.resolve()),
            "val": str(val_archive.resolve()),
            "devkit": str(devkit_archive.resolve()),
        },
    }
    (output / "fieldscope_preparation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "classes.txt").write_text(
        "\n".join(selected_classes) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-archive", required=True, type=Path)
    parser.add_argument("--val-archive", required=True, type=Path)
    parser.add_argument("--devkit-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = prepare_imagenet100(
        args.train_archive,
        args.val_archive,
        args.devkit_archive,
        args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
