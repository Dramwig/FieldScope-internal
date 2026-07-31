"""Convert official NYUv2 labeled MAT files to explicit RGB/depth manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb_image(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3:
        raise ValueError(f"Unexpected NYUv2 image shape: {value.shape}")
    if value.shape[0] == 3:
        value = value.transpose(2, 1, 0)
    elif value.shape[-1] != 3:
        raise ValueError(f"Cannot identify RGB axis in shape {value.shape}")
    return np.asarray(value, dtype=np.uint8)


def _depth_image(value: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != image_shape:
        value = value.T
    if value.shape != image_shape:
        raise ValueError(
            f"NYUv2 depth shape {value.shape} does not match RGB {image_shape}"
        )
    return value


def _split_indices(splits_path: Path, seed: int) -> dict[str, list[int]]:
    splits = loadmat(splits_path)
    training = np.asarray(splits["trainNdxs"]).reshape(-1).astype(np.int64) - 1
    testing = np.asarray(splits["testNdxs"]).reshape(-1).astype(np.int64) - 1
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(training)
    validation_count = max(1, round(len(training) * 0.1))
    return {
        "train": sorted(shuffled[validation_count:].tolist()),
        "val": sorted(shuffled[:validation_count].tolist()),
        "test": sorted(testing.tolist()),
    }


def prepare_nyuv2(
    labeled_mat: Path,
    splits_mat: Path,
    output: Path,
    *,
    seed: int = 4121,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    image_root = output / "images"
    depth_root = output / "depth"
    image_root.mkdir(exist_ok=True)
    depth_root.mkdir(exist_ok=True)
    split_indices = _split_indices(splits_mat, seed)
    manifests: dict[str, list[dict[str, str]]] = {
        split: [] for split in split_indices
    }
    with h5py.File(labeled_mat, "r") as dataset:
        images = dataset["images"]
        depths = dataset["depths"]
        if len(images) != 1449 or len(depths) != 1449:
            raise ValueError("Official NYUv2 labeled file must contain 1449 pairs")
        for split, indices in split_indices.items():
            for source_index in indices:
                sample_id = f"nyuv2-{source_index + 1:04d}"
                image_relative = Path("images") / f"{sample_id}.png"
                depth_relative = Path("depth") / f"{sample_id}.npy"
                image_path = output / image_relative
                depth_path = output / depth_relative
                image = _rgb_image(np.asarray(images[source_index]))
                depth = _depth_image(
                    np.asarray(depths[source_index]),
                    image.shape[:2],
                )
                if not image_path.is_file():
                    Image.fromarray(image).save(image_path)
                if not depth_path.is_file():
                    np.save(depth_path, depth)
                manifests[split].append(
                    {
                        "id": sample_id,
                        "image": image_relative.as_posix(),
                        "depth": depth_relative.as_posix(),
                    }
                )
    for split, samples in manifests.items():
        (output / f"{split}.json").write_text(
            json.dumps(samples, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {
        "status": "passed",
        "labeled_mat": str(labeled_mat.resolve()),
        "labeled_mat_bytes": labeled_mat.stat().st_size,
        "labeled_mat_sha256": _sha256(labeled_mat),
        "splits_mat": str(splits_mat.resolve()),
        "splits_mat_sha256": _sha256(splits_mat),
        "seed": seed,
        "split_counts": {
            split: len(samples) for split, samples in manifests.items()
        },
        "depth_units": "meters",
    }
    (output / "fieldscope_preparation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-mat", required=True, type=Path)
    parser.add_argument("--splits-mat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4121)
    args = parser.parse_args()
    report = prepare_nyuv2(
        args.labeled_mat,
        args.splits_mat,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
