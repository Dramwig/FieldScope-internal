import torch

from fieldscope.graph import (
    boundary_strength,
    cosine_affinity,
    normalize_adjacency,
    patch_pool,
    sparsify_affinity,
    spectral_binary_partition,
)


def test_patch_pool_and_affinity_shapes() -> None:
    feature_map = torch.randn(2, 6, 8, 8)
    patches = patch_pool(feature_map, (4, 4))
    affinity = cosine_affinity(patches)
    assert patches.shape == (2, 16, 6)
    assert affinity.shape == (2, 16, 16)
    assert torch.allclose(affinity, affinity.transpose(1, 2), atol=1e-5)


def test_sparse_graph_is_symmetric_and_diagnostic_ready() -> None:
    features = torch.randn(2, 16, 12)
    affinity = cosine_affinity(features)
    adjacency = sparsify_affinity(affinity, (4, 4), topk=3, local_radius=1)
    normalized = normalize_adjacency(adjacency)
    partition = spectral_binary_partition(adjacency)
    boundaries = boundary_strength(affinity, (4, 4))
    assert torch.allclose(adjacency, adjacency.transpose(1, 2))
    assert torch.isfinite(normalized).all()
    assert partition.shape == (2, 16)
    assert boundaries.shape == (2, 16)
    assert torch.all(adjacency.diagonal(dim1=-2, dim2=-1) >= 1)

