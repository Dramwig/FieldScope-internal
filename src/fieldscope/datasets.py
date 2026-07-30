"""Synthetic smoke data and thin torchvision dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


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
            transforms.Resize((image_size, image_size), antialias=True),
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
                    (image_size, image_size), interpolation=InterpolationMode.NEAREST
                ),
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

