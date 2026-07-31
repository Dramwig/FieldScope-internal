"""Patch pooling, affinity construction, sparsification, and diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_pool(tensor: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Adaptive-average pool [B,C,H,W] to [B,P,C]."""

    if tensor.ndim != 4:
        raise ValueError("tensor must have shape [B,C,H,W]")
    pooled = F.adaptive_avg_pool2d(tensor, grid_size)
    return pooled.flatten(2).transpose(1, 2).contiguous()


def cosine_affinity(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine Gram matrix for [B,P,D] features."""

    if features.ndim != 3:
        raise ValueError("features must have shape [B,P,D]")
    normalized = F.normalize(features.float(), dim=-1, eps=eps)
    return torch.bmm(normalized, normalized.transpose(1, 2)).to(dtype=features.dtype)


def pooled_attention_affinity(
    query: torch.Tensor,
    key: torch.Tensor,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Pool native multi-head Q/K maps and return a symmetric attention graph."""

    if query.ndim != 5 or query.shape != key.shape:
        raise ValueError("query and key must have shape [B,heads,D,H,W]")
    batch, heads, head_dim, height, width = query.shape
    pooled_query = F.adaptive_avg_pool2d(
        query.reshape(batch, heads * head_dim, height, width),
        grid_size,
    )
    pooled_key = F.adaptive_avg_pool2d(
        key.reshape(batch, heads * head_dim, height, width),
        grid_size,
    )
    patches = grid_size[0] * grid_size[1]
    pooled_query = pooled_query.reshape(batch, heads, head_dim, patches).transpose(
        2, 3
    )
    pooled_key = pooled_key.reshape(batch, heads, head_dim, patches).transpose(2, 3)
    logits = torch.einsum(
        "bhpd,bhqd->bhpq",
        pooled_query.float(),
        pooled_key.float(),
    ) / head_dim**0.5
    attention = logits.softmax(dim=-1).mean(dim=1)
    return 0.5 * (attention + attention.transpose(1, 2))


def _local_mask(
    grid_size: tuple[int, int],
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    height, width = grid_size
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
    distance = (positions[:, None] - positions[None, :]).abs()
    return (distance[..., 0] <= radius) & (distance[..., 1] <= radius)


def sparsify_affinity(
    affinity: torch.Tensor,
    grid_size: tuple[int, int],
    topk: int,
    local_radius: int,
) -> torch.Tensor:
    """Keep positive local and global top-k weighted edges plus self loops."""

    if affinity.ndim != 3 or affinity.shape[-1] != affinity.shape[-2]:
        raise ValueError("affinity must have shape [B,P,P]")
    batch, patches, _ = affinity.shape
    if patches != grid_size[0] * grid_size[1]:
        raise ValueError("grid_size does not match affinity")

    positive = affinity.clamp_min(0)
    eye = torch.eye(patches, device=affinity.device, dtype=torch.bool)
    local = _local_mask(grid_size, local_radius, affinity.device) if local_radius else eye
    keep = local.unsqueeze(0).expand(batch, -1, -1).clone()

    if topk:
        candidates = affinity.masked_fill(eye.unsqueeze(0), float("-inf"))
        count = min(topk, max(0, patches - 1))
        if count:
            indices = candidates.topk(count, dim=-1).indices
            top_mask = torch.zeros_like(affinity, dtype=torch.bool)
            top_mask.scatter_(-1, indices, True)
            keep |= top_mask

    keep = keep | keep.transpose(1, 2)
    keep |= eye.unsqueeze(0)
    adjacency = positive * keep.to(dtype=positive.dtype)
    adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
    adjacency = adjacency + eye.unsqueeze(0).to(adjacency.dtype)
    return adjacency


def normalize_adjacency(adjacency: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric D^-1/2 A D^-1/2 normalization."""

    degree = adjacency.sum(dim=-1).clamp_min(eps)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt.unsqueeze(-1) * adjacency * inverse_sqrt.unsqueeze(-2)


def fixed_grid_adjacency(
    grid_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    radius: int = 1,
) -> torch.Tensor:
    """Content-independent local graph for matched state-only baselines."""

    mask = _local_mask(grid_size, radius, device)
    adjacency = mask.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
    return adjacency


def spectral_binary_partition(adjacency: torch.Tensor) -> torch.Tensor:
    """Return a deterministic two-way normalized-cut partition [B,P]."""

    normalized = normalize_adjacency(adjacency.float())
    patches = adjacency.shape[-1]
    identity = torch.eye(patches, device=adjacency.device).unsqueeze(0)
    laplacian = identity - normalized
    _, eigenvectors = torch.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, :, 1] if patches > 1 else eigenvectors[:, :, 0]
    threshold = fiedler.median(dim=-1, keepdim=True).values
    return (fiedler > threshold).long()


def boundary_strength(
    affinity: torch.Tensor, grid_size: tuple[int, int]
) -> torch.Tensor:
    """Estimate per-patch boundary strength from 4-neighbour dissimilarity."""

    batch, patches, _ = affinity.shape
    height, width = grid_size
    if patches != height * width:
        raise ValueError("grid_size does not match affinity")
    index = torch.arange(patches, device=affinity.device).reshape(height, width)
    score = torch.zeros((batch, patches), device=affinity.device, dtype=affinity.dtype)
    count = torch.zeros((patches,), device=affinity.device, dtype=affinity.dtype)
    for first, second in (
        (index[:, :-1], index[:, 1:]),
        (index[:-1, :], index[1:, :]),
    ):
        first_flat = first.flatten()
        second_flat = second.flatten()
        difference = 1.0 - affinity[:, first_flat, second_flat]
        score[:, first_flat] += difference
        score[:, second_flat] += difference
        count[first_flat] += 1
        count[second_flat] += 1
    return score / count.clamp_min(1).unsqueeze(0)
