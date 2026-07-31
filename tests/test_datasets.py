import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fieldscope.datasets import (
    ADE20KDirectoryDataset,
    NYUv2DirectoryDataset,
    SyntheticShapesDataset,
    build_vision_dataset,
    fraction_holdout_indices,
    stratified_fraction_holdout_indices,
    stratified_holdout_indices,
)


def test_synthetic_dataset_is_deterministic_and_complete() -> None:
    dataset = SyntheticShapesDataset(length=4, image_size=32, seed=9)
    first = dataset[1]
    second = dataset[1]
    assert set(first) == {
        "image",
        "classification",
        "segmentation",
        "depth",
        "normals",
        "sample_id",
    }
    assert torch.equal(first["image"], second["image"])
    assert first["image"].shape == (3, 32, 32)
    assert first["segmentation"].shape == (32, 32)
    assert first["normals"].shape == (3, 32, 32)


def test_imagenet100_subset_has_contiguous_labels(tmp_path: Path) -> None:
    for class_name, value in (("n0002", 64), ("n0001", 192)):
        directory = tmp_path / "val" / class_name
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(value, 0, 0)).save(directory / "sample.png")

    dataset = build_vision_dataset(
        "imagenet100",
        tmp_path,
        "test",
        image_size=16,
        class_names=["n0002", "n0001"],
    )
    assert dataset.classes == ["n0002", "n0001"]
    assert [dataset[index]["classification"].item() for index in range(2)] == [1, 0]


def test_nyuv2_directory_dataset_uses_explicit_manifest(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "depth").mkdir()
    Image.new("RGB", (8, 6), color=(12, 34, 56)).save(tmp_path / "images" / "a.png")
    np.save(tmp_path / "depth" / "a.npy", np.full((6, 8), 2.5, dtype=np.float32))
    (tmp_path / "train.json").write_text(
        json.dumps(
            [{"id": "nyu-a", "image": "images/a.png", "depth": "depth/a.npy"}]
        ),
        encoding="utf-8",
    )

    sample = NYUv2DirectoryDataset(tmp_path, "train", image_size=16)[0]
    assert sample["sample_id"] == "nyu-a"
    assert sample["image"].shape == (3, 16, 16)
    assert sample["depth"].shape == (1, 16, 16)
    assert torch.allclose(sample["depth"], torch.full((1, 16, 16), 2.5))


def test_stratified_holdout_is_deterministic_and_balanced() -> None:
    targets = [0] * 10 + [1] * 10
    first = stratified_holdout_indices(
        targets,
        holdout_per_class=2,
        seed=4121,
    )
    second = stratified_holdout_indices(
        targets,
        holdout_per_class=2,
        seed=4121,
    )
    assert first == second
    training, validation = first
    assert len(training) == 16
    assert len(validation) == 4
    assert [targets[index] for index in validation].count(0) == 2
    assert [targets[index] for index in validation].count(1) == 2

    fraction_training, fraction_validation = (
        stratified_fraction_holdout_indices(
            targets,
            fraction=0.2,
            seed=4121,
        )
    )
    assert len(fraction_training) == 16
    assert len(fraction_validation) == 4
    unstratified_training, unstratified_validation = fraction_holdout_indices(
        20,
        fraction=0.2,
        seed=4121,
    )
    assert len(unstratified_training) == 16
    assert len(unstratified_validation) == 4
    assert not set(unstratified_training) & set(unstratified_validation)


def test_ade20k_directory_dataset_remaps_labels(tmp_path: Path) -> None:
    image_root = tmp_path / "images" / "training"
    annotation_root = tmp_path / "annotations" / "training"
    image_root.mkdir(parents=True)
    annotation_root.mkdir(parents=True)
    Image.new("RGB", (2, 2), color=(12, 34, 56)).save(
        image_root / "ADE_train_00000001.jpg"
    )
    annotation = Image.fromarray(np.array([[0, 1], [2, 150]], dtype=np.uint8))
    annotation.save(annotation_root / "ADE_train_00000001.png")

    sample = ADE20KDirectoryDataset(tmp_path, "train", image_size=2)[0]
    assert sample["sample_id"] == "ADE_train_00000001"
    assert sample["segmentation"].tolist() == [[255, 0], [1, 149]]
