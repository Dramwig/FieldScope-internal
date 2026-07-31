"""Source-identity audits for benchmark train/validation/test splits."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from fieldscope.datasets import build_vision_dataset, dataset_sample_ids


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Hash an ID multiset without ambiguous string concatenation."""

    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _split_summary(sample_ids: list[str]) -> tuple[dict[str, Any], list[str]]:
    counts: dict[str, int] = {}
    for sample_id in sample_ids:
        counts[sample_id] = counts.get(sample_id, 0) + 1
    duplicate_ids = sorted(
        sample_id for sample_id, count in counts.items() if count > 1
    )
    summary = {
        "count": len(sample_ids),
        "unique_count": len(counts),
        "duplicate_count": len(sample_ids) - len(counts),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_examples": duplicate_ids[:10],
        "sample_ids_sha256": sample_ids_sha256(sample_ids),
    }
    return summary, duplicate_ids


def audit_dataset_splits(
    *,
    dataset_name: str,
    dataset_root: Path,
    image_size: int,
    class_names: list[str] | None = None,
    splits: Sequence[str] = ("train", "val", "test"),
    expected_counts: dict[str, int] | None = None,
    dataset_builder: Callable[..., Dataset[Any]] = build_vision_dataset,
    sample_id_reader: Callable[[Dataset[Any]], list[str]] = dataset_sample_ids,
) -> dict[str, Any]:
    """Build benchmark splits and verify source IDs are unique and disjoint.

    The default ID reader inspects dataset metadata only. It never calls
    ``Dataset.__getitem__`` and therefore does not decode images or targets.
    """

    if image_size < 1:
        raise ValueError("image_size must be positive")
    split_names = list(splits)
    if not split_names or len(split_names) != len(set(split_names)):
        raise ValueError("splits must contain unique names")
    expected_counts = expected_counts or {}
    unknown_expected = sorted(set(expected_counts) - set(split_names))
    if unknown_expected or any(count < 1 for count in expected_counts.values()):
        raise ValueError("expected_counts must contain positive counts for audited splits")

    split_ids: dict[str, list[str]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for split in split_names:
        dataset = dataset_builder(
            dataset_name,
            dataset_root,
            split,
            image_size,
            class_names=class_names,
        )
        sample_ids = sample_id_reader(dataset)
        if len(sample_ids) != len(dataset):
            raise ValueError(
                f"ID count for {split} does not match dataset length: "
                f"{len(sample_ids)} != {len(dataset)}"
            )
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
            raise ValueError(f"Split {split} contains an empty or non-string sample ID")
        split_ids[split] = sample_ids
        summary, duplicate_ids = _split_summary(sample_ids)
        summary["expected_count"] = expected_counts.get(split)
        summary["count_matches_expected"] = (
            len(sample_ids) == expected_counts[split]
            if split in expected_counts
            else None
        )
        split_reports[split] = summary
        if split in expected_counts and len(sample_ids) != expected_counts[split]:
            problems.append(
                f"{split} count {len(sample_ids)} != expected {expected_counts[split]}"
            )
        if duplicate_ids:
            problems.append(
                f"{split} contains {summary['duplicate_count']} duplicate occurrences "
                f"across {len(duplicate_ids)} source IDs"
            )

    pairwise_overlaps: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
            pair_name = f"{left}__{right}"
            pairwise_overlaps[pair_name] = {
                "count": len(overlap),
                "examples": overlap[:10],
            }
            if overlap:
                problems.append(
                    f"{left} and {right} overlap on {len(overlap)} source IDs"
                )

    from fieldscope.experiments import code_provenance

    return {
        "status": "passed" if not problems else "failed",
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "image_size": image_size,
        "class_names": class_names,
        "split_order": split_names,
        "expected_counts": expected_counts,
        "splits": split_reports,
        "pairwise_overlaps": pairwise_overlaps,
        "problems": problems,
        "metadata_only": True,
        **code_provenance(),
    }
