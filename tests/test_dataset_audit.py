from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from fieldscope.dataset_audit import audit_dataset_splits
from fieldscope.datasets import build_vision_dataset, dataset_sample_ids


class _IdentityDataset(Dataset[dict[str, Any]]):
    def __init__(self, split: str, sample_ids: list[str]) -> None:
        self.split = split
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise AssertionError(f"sample {index} must not be decoded during audit")


def _identity_builder(
    _name: str,
    _root: Path,
    split: str,
    _image_size: int,
    *,
    class_names: list[str] | None,
) -> Dataset[Any]:
    del class_names
    identities = {
        "train": ["source-a", "source-a"],
        "val": ["source-b"],
        "test": ["source-b", "source-c"],
    }
    return _IdentityDataset(split, identities[split])


def _identity_reader(dataset: Dataset[Any]) -> list[str]:
    assert isinstance(dataset, _IdentityDataset)
    return list(dataset.sample_ids)


def test_split_audit_fails_on_duplicates_and_overlap(tmp_path: Path) -> None:
    report = audit_dataset_splits(
        dataset_name="artificial",
        dataset_root=tmp_path,
        image_size=16,
        dataset_builder=_identity_builder,
        sample_id_reader=_identity_reader,
    )
    assert report["status"] == "failed"
    assert report["splits"]["train"]["duplicate_count"] == 1
    assert report["pairwise_overlaps"]["val__test"]["count"] == 1
    assert report["metadata_only"] is True


def test_split_audit_fails_on_expected_count_mismatch(tmp_path: Path) -> None:
    report = audit_dataset_splits(
        dataset_name="artificial",
        dataset_root=tmp_path,
        image_size=16,
        expected_counts={"train": 3, "val": 1, "test": 2},
        dataset_builder=_identity_builder,
        sample_id_reader=_identity_reader,
    )
    assert report["status"] == "failed"
    assert report["splits"]["train"]["count_matches_expected"] is False
    assert any("train count 2 != expected 3" in problem for problem in report["problems"])


def test_cifar_train_and_validation_share_source_namespace_without_overlap(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class FakeCIFAR10(Dataset[tuple[Any, int]]):
        def __init__(
            self,
            *,
            root: str,
            train: bool,
            transform: Any,
            download: bool,
        ) -> None:
            del root, transform, download
            self.train = train
            self.targets = [label for label in range(10) for _ in range(501)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            raise AssertionError(f"sample {index} must not be decoded during ID audit")

    monkeypatch.setattr("torchvision.datasets.CIFAR10", FakeCIFAR10)
    train = build_vision_dataset("cifar10", tmp_path, "train", image_size=16)
    validation = build_vision_dataset("cifar10", tmp_path, "val", image_size=16)
    train_ids = dataset_sample_ids(train)
    validation_ids = dataset_sample_ids(validation)
    assert all(sample_id.startswith("cifar10-official-train-") for sample_id in train_ids)
    assert all(
        sample_id.startswith("cifar10-official-train-")
        for sample_id in validation_ids
    )
    assert not set(train_ids) & set(validation_ids)
    assert len(train_ids) == 10
    assert len(validation_ids) == 5000


def test_imagenet_internal_splits_and_official_test_are_disjoint(
    tmp_path: Path,
) -> None:
    for source_split in ("train", "val"):
        for class_name in ("n0001", "n0002"):
            directory = tmp_path / source_split / class_name
            directory.mkdir(parents=True)
            for index in range(2):
                (directory / f"sample-{index}.jpg").touch()

    datasets = {
        split: build_vision_dataset(
            "imagenet100",
            tmp_path,
            split,
            image_size=16,
            class_names=["n0001", "n0002"],
        )
        for split in ("train", "val", "test")
    }
    identities = {split: dataset_sample_ids(dataset) for split, dataset in datasets.items()}
    assert not set(identities["train"]) & set(identities["val"])
    assert not set(identities["train"]) & set(identities["test"])
    assert not set(identities["val"]) & set(identities["test"])
    assert all(sample_id.startswith("imagenet100-train-") for sample_id in identities["train"])
    assert all(sample_id.startswith("imagenet100-train-") for sample_id in identities["val"])
    assert all(sample_id.startswith("imagenet100-val-") for sample_id in identities["test"])
