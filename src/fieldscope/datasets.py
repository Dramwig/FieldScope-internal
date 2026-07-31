"""Synthetic data and auditable adapters for real vision benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


def _path_sample_id(sample_prefix: str, path: str | Path) -> str:
    source = Path(path)
    source_name = (Path(source.parent.name) / source.name).as_posix()
    return f"{sample_prefix}-{source_name}"


def _dataset_source_sample_id(dataset: Dataset[Any], index: int, sample_prefix: str) -> str:
    images = getattr(dataset, "images", None)
    if images is not None:
        return _path_sample_id(sample_prefix, images[index])
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        source = samples[index][0]
        if isinstance(source, (str, Path)):
            return _path_sample_id(sample_prefix, source)
    return f"{sample_prefix}-{index:08d}"


class SyntheticShapesDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic classification + segmentation + depth + normal targets."""

    def __init__(self, length: int = 32, image_size: int = 64, seed: int = 4121):
        self.length = length
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        size = self.image_size
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        shape_class = index % 2
        center_y = int(torch.randint(size // 3, 2 * size // 3, (1,), generator=generator))
        center_x = int(torch.randint(size // 3, 2 * size // 3, (1,), generator=generator))
        radius = int(torch.randint(max(3, size // 8), max(4, size // 4), (1,), generator=generator))
        if shape_class == 0:
            foreground = (yy - center_y).abs().maximum((xx - center_x).abs()) <= radius
        else:
            foreground = (yy - center_y).square() + (xx - center_x).square() <= radius**2

        image = torch.rand((3, size, size), generator=generator) * 0.08
        color = torch.rand((3, 1, 1), generator=generator) * 0.6 + 0.35
        image = torch.where(foreground.unsqueeze(0), color, image)
        segmentation = foreground.long() * (shape_class + 1)
        depth = torch.ones((1, size, size))
        depth[:, foreground] = 0.35 + 0.15 * shape_class

        delta_x = (xx - center_x).float() / max(1, radius)
        delta_y = (yy - center_y).float() / max(1, radius)
        normal_z = torch.sqrt((1.0 - delta_x.square() - delta_y.square()).clamp_min(0))
        normals = torch.stack([delta_x, delta_y, normal_z], dim=0)
        background_normal = torch.tensor([0.0, 0.0, 1.0]).reshape(3, 1, 1)
        normals = torch.where(foreground.unsqueeze(0), normals, background_normal)
        normals = torch.nn.functional.normalize(normals, dim=0)
        return {
            "image": image,
            "classification": torch.tensor(shape_class, dtype=torch.long),
            "segmentation": segmentation,
            "depth": depth,
            "normals": normals,
            "sample_id": f"synthetic-{index:06d}",
        }


def build_torchvision_dataset(
    name: str,
    root: str | Path,
    split: str,
    image_size: int,
    *,
    download: bool = False,
) -> Dataset[Any]:
    """Build supported real datasets without hiding download behavior."""

    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    root = str(root)
    image_transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    if name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=split == "train",
            transform=image_transform,
            download=download,
        )
    if name == "voc2012":
        target_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(image_size),
                transforms.PILToTensor(),
                transforms.Lambda(lambda tensor: tensor.squeeze(0).long()),
            ]
        )
        return datasets.VOCSegmentation(
            root=root,
            year="2012",
            image_set=split,
            download=download,
            transform=image_transform,
            target_transform=target_transform,
        )
    raise ValueError(f"Unsupported dataset: {name}")


class TaskDataset(Dataset[dict[str, Any]]):
    """Normalize tuple-style datasets to the FieldScope sample contract."""

    def __init__(self, dataset: Dataset[Any], task: str, sample_prefix: str):
        self.dataset = dataset
        self.task = task
        self.sample_prefix = sample_prefix

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, target = self.dataset[index]
        return {
            "image": image,
            self.task: target,
            "sample_id": _dataset_source_sample_id(
                self.dataset,
                index,
                self.sample_prefix,
            ),
        }


class IndexedTaskDataset(Dataset[dict[str, Any]]):
    """Task adapter over an explicit source-index split."""

    def __init__(
        self,
        dataset: Dataset[Any],
        indices: list[int],
        task: str,
        sample_prefix: str,
    ):
        self.dataset = dataset
        self.indices = indices
        self.task = task
        self.sample_prefix = sample_prefix

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        image, target = self.dataset[source_index]
        return {
            "image": image,
            self.task: target,
            "sample_id": _dataset_source_sample_id(
                self.dataset,
                source_index,
                self.sample_prefix,
            ),
        }


def stratified_holdout_indices(
    targets: list[int],
    *,
    holdout_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return sorted train/holdout indices from a deterministic class split."""

    generator = np.random.default_rng(seed)
    target_array = np.asarray(targets)
    training: list[int] = []
    holdout: list[int] = []
    for label in sorted(np.unique(target_array).tolist()):
        indices = np.flatnonzero(target_array == label)
        if len(indices) <= holdout_per_class:
            raise ValueError(f"Class {label} has too few samples for the holdout")
        shuffled = generator.permutation(indices)
        holdout.extend(shuffled[:holdout_per_class].tolist())
        training.extend(shuffled[holdout_per_class:].tolist())
    return sorted(training), sorted(holdout)


def stratified_fraction_holdout_indices(
    targets: list[int],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministically hold out the same fraction within every class."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between zero and one")
    generator = np.random.default_rng(seed)
    target_array = np.asarray(targets)
    training: list[int] = []
    holdout: list[int] = []
    for label in sorted(np.unique(target_array).tolist()):
        indices = np.flatnonzero(target_array == label)
        count = max(1, round(len(indices) * fraction))
        if len(indices) <= count:
            raise ValueError(f"Class {label} has too few samples for the holdout")
        shuffled = generator.permutation(indices)
        holdout.extend(shuffled[:count].tolist())
        training.extend(shuffled[count:].tolist())
    return sorted(training), sorted(holdout)


def fraction_holdout_indices(
    length: int,
    *,
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministically split an unstratified dataset into train/holdout."""

    if length < 2 or not 0.0 < fraction < 1.0:
        raise ValueError("length and fraction do not define a valid holdout")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(length)
    holdout_count = max(1, round(length * fraction))
    return (
        sorted(shuffled[holdout_count:].tolist()),
        sorted(shuffled[:holdout_count].tolist()),
    )


class ClassSubsetDataset(Dataset[dict[str, Any]]):
    """Deterministic class subset with contiguous labels."""

    def __init__(
        self,
        dataset: Any,
        class_names: list[str],
        *,
        sample_prefix: str,
    ):
        available = {name: index for index, name in enumerate(dataset.classes)}
        missing = sorted(set(class_names) - set(available))
        if missing:
            raise ValueError(f"Unknown ImageFolder classes: {missing[:5]}")
        source_to_target = {
            available[name]: target for target, name in enumerate(class_names)
        }
        self.samples = [
            (path, source_to_target[label])
            for path, label in dataset.samples
            if label in source_to_target
        ]
        self.loader = dataset.loader
        self.transform = dataset.transform
        self.classes = list(class_names)
        self.sample_prefix = sample_prefix

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "classification": torch.tensor(target, dtype=torch.long),
            "sample_id": _path_sample_id(self.sample_prefix, path),
        }


class NYUv2DirectoryDataset(Dataset[dict[str, Any]]):
    """Prepared NYUv2 RGB/depth pairs with an explicit split manifest.

    The manifest is a JSON list with ``id``, ``image`` and ``depth`` paths,
    relative to ``root``. Depth arrays are stored in meters as ``.npy``.
    """

    def __init__(self, root: str | Path, split: str, image_size: int):
        import json

        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self.root = Path(root)
        manifest_path = self.root / f"{split}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.image_size = image_size
        self._interpolation = InterpolationMode.BILINEAR
        self._depth_interpolation = InterpolationMode.NEAREST
        self._tf = TF

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(self.root / sample["image"]).convert("RGB")
        image = self._tf.resize(
            image,
            self.image_size,
            interpolation=self._interpolation,
            antialias=True,
        )
        image = self._tf.center_crop(image, [self.image_size, self.image_size])
        image_tensor = self._tf.to_tensor(image)
        depth = torch.from_numpy(
            np.load(self.root / sample["depth"]).astype(np.float32, copy=False)
        ).unsqueeze(0)
        depth = self._tf.resize(
            depth,
            self.image_size,
            interpolation=self._depth_interpolation,
        )
        depth = self._tf.center_crop(depth, [self.image_size, self.image_size])
        return {
            "image": image_tensor,
            "depth": depth,
            "sample_id": str(sample["id"]),
        }


class ADE20KDirectoryDataset(Dataset[dict[str, Any]]):
    """ADE20K semantic segmentation with official 150-class label remapping."""

    def __init__(self, root: str | Path, split: str, image_size: int):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        split_name = {"train": "training", "val": "validation"}.get(split, split)
        if split_name not in {"training", "validation"}:
            raise ValueError("ADE20K split must be train/training or val/validation")
        self.root = Path(root)
        image_root = self.root / "images" / split_name
        annotation_root = self.root / "annotations" / split_name
        self.samples = [
            (path, annotation_root / f"{path.stem}.png")
            for path in sorted(image_root.glob("*.jpg"))
        ]
        if not self.samples or any(not annotation.is_file() for _, annotation in self.samples):
            raise FileNotFoundError(
                f"Incomplete ADE20K image/annotation pairs under {self.root}"
            )
        self.image_size = image_size
        self._tf = TF
        self._image_interpolation = InterpolationMode.BILINEAR
        self._target_interpolation = InterpolationMode.NEAREST

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, annotation_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        annotation = Image.open(annotation_path)
        image = self._tf.resize(
            image,
            self.image_size,
            interpolation=self._image_interpolation,
            antialias=True,
        )
        annotation = self._tf.resize(
            annotation,
            self.image_size,
            interpolation=self._target_interpolation,
        )
        image = self._tf.center_crop(image, [self.image_size, self.image_size])
        annotation = self._tf.center_crop(
            annotation,
            [self.image_size, self.image_size],
        )
        target = self._tf.pil_to_tensor(annotation).squeeze(0).long()
        target = torch.where(target == 0, 255, target - 1)
        return {
            "image": self._tf.to_tensor(image),
            "segmentation": target,
            "sample_id": image_path.stem,
        }


def build_vision_dataset(
    name: str,
    root: str | Path,
    split: str,
    image_size: int,
    *,
    class_names: list[str] | None = None,
) -> Dataset[dict[str, Any]]:
    """Build a benchmark dataset with explicit task names and sample IDs."""

    if name == "cifar10":
        from torchvision import datasets, transforms

        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        if split == "test":
            return TaskDataset(
                datasets.CIFAR10(
                    root=str(root),
                    train=False,
                    transform=image_transform,
                    download=False,
                ),
                "classification",
                "cifar10-official-test",
            )
        if split not in {"train", "val"}:
            raise ValueError("CIFAR-10 split must be train, val, or test")
        dataset = datasets.CIFAR10(
            root=str(root),
            train=True,
            transform=image_transform,
            download=False,
        )
        training, validation = stratified_holdout_indices(
            dataset.targets,
            holdout_per_class=500,
            seed=4121,
        )
        indices = training if split == "train" else validation
        return IndexedTaskDataset(
            dataset,
            indices,
            "classification",
            "cifar10-official-train",
        )
    if name == "voc2012":
        if split not in {"train", "val", "test"}:
            raise ValueError("VOC 2012 split must be train, val, or test")
        source_split = "val" if split == "test" else "train"
        field_dataset = TaskDataset(
            build_torchvision_dataset(
                name,
                root,
                source_split,
                image_size,
                download=False,
            ),
            "segmentation",
            f"{name}-{source_split}",
        )
        if split in {"train", "val"}:
            training, validation = fraction_holdout_indices(
                len(field_dataset),
                fraction=0.1,
                seed=4121,
            )
            return Subset(
                field_dataset,
                training if split == "train" else validation,
            )
        return field_dataset
    if name in {"imagenet", "imagenet100"}:
        from torchvision import datasets, transforms

        if split not in {"train", "val", "test"}:
            raise ValueError("ImageNet split must be train, val, or test")
        source_split = "val" if split == "test" else "train"
        split_root = Path(root) / source_split
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(split_root, transform=image_transform)
        selected = class_names
        if selected is None and name == "imagenet100":
            selected = sorted(dataset.classes)[:100]
        if selected is not None:
            field_dataset: Dataset[dict[str, Any]] = ClassSubsetDataset(
                dataset,
                selected,
                sample_prefix=f"{name}-{source_split}",
            )
            targets = [target for _, target in field_dataset.samples]
        else:
            field_dataset = TaskDataset(
                dataset,
                "classification",
                f"{name}-{source_split}",
            )
            targets = list(dataset.targets)
        if split in {"train", "val"}:
            training, validation = stratified_fraction_holdout_indices(
                targets,
                fraction=0.1,
                seed=4121,
            )
            return Subset(
                field_dataset,
                training if split == "train" else validation,
            )
        return field_dataset
    if name == "nyuv2":
        return NYUv2DirectoryDataset(root, split, image_size)
    if name == "ade20k":
        if split not in {"train", "val", "test"}:
            raise ValueError("ADE20K split must be train, val, or test")
        source_split = "validation" if split == "test" else "training"
        field_dataset = ADE20KDirectoryDataset(root, source_split, image_size)
        if split in {"train", "val"}:
            training, validation = fraction_holdout_indices(
                len(field_dataset),
                fraction=0.1,
                seed=4121,
            )
            return Subset(
                field_dataset,
                training if split == "train" else validation,
            )
        return field_dataset
    raise ValueError(f"Unsupported dataset: {name}")


def dataset_sample_ids(dataset: Dataset[Any]) -> list[str]:
    """Return source-stable sample IDs without decoding images or targets."""

    if isinstance(dataset, Subset):
        source_ids = dataset_sample_ids(dataset.dataset)
        return [source_ids[int(index)] for index in dataset.indices]
    if isinstance(dataset, SyntheticShapesDataset):
        return [f"synthetic-{index:06d}" for index in range(len(dataset))]
    if isinstance(dataset, IndexedTaskDataset):
        return [
            _dataset_source_sample_id(
                dataset.dataset,
                source_index,
                dataset.sample_prefix,
            )
            for source_index in dataset.indices
        ]
    if isinstance(dataset, TaskDataset):
        return [
            _dataset_source_sample_id(
                dataset.dataset,
                index,
                dataset.sample_prefix,
            )
            for index in range(len(dataset))
        ]
    if isinstance(dataset, ClassSubsetDataset):
        return [
            _path_sample_id(dataset.sample_prefix, path)
            for path, _ in dataset.samples
        ]
    if isinstance(dataset, NYUv2DirectoryDataset):
        return [str(sample["id"]) for sample in dataset.samples]
    if isinstance(dataset, ADE20KDirectoryDataset):
        return [image_path.stem for image_path, _ in dataset.samples]
    raise TypeError(f"Cannot derive source-stable IDs for {type(dataset).__name__}")
