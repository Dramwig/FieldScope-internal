import torch

from fieldscope.datasets import SyntheticShapesDataset


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

