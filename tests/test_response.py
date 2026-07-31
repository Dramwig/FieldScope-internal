import pytest
import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.config import ProbeConfig
from fieldscope.feature_ops import slice_features
from fieldscope.response import FieldResponseExtractor


def test_extractor_produces_complete_feature_contract() -> None:
    backend = ToyFieldBackend(image_size=32)
    config = ProbeConfig(
        times=(0.25, 0.75),
        num_directions=4,
        eta=0.02,
        graph_grid=(4, 4),
        topk=3,
        local_radius=1,
        probe_batch_size=8,
        antithetic_noise=True,
        seed=11,
    )
    images = torch.rand(2, 3, 32, 32)
    features = FieldResponseExtractor(backend, config).extract(images)
    assert features.state.shape == (2, 16, 14)
    assert features.response.shape == (2, 16, 64)
    assert features.affinity.shape == (2, 16, 16)
    assert set(features.baselines) == {
        "z0",
        "zt",
        "trajectory",
        "velocity",
        "mismatch",
        "endpoint",
        "dit_hidden",
    }
    assert set(features.graphs) == {
        "dit_attention",
        "dit_attention_adjacency",
    }
    assert features.baselines["dit_hidden"].shape == (2, 16, 768)
    assert features.baselines["trajectory"].shape == (2, 16, 8)
    assert features.metadata["noise_views"] == 2
    features.validate()


def test_forward_difference_path() -> None:
    backend = ToyFieldBackend(image_size=32)
    config = ProbeConfig(
        times=(0.5,),
        num_directions=2,
        difference="forward",
        graph_grid=(2, 2),
        probe_batch_size=4,
        antithetic_noise=False,
    )
    features = FieldResponseExtractor(backend, config).extract(torch.rand(1, 3, 32, 32))
    assert torch.isfinite(features.response).all()


def test_sample_seeded_extraction_is_batch_invariant() -> None:
    backend = ToyFieldBackend(image_size=32)
    config = ProbeConfig(
        times=(0.25, 0.75),
        num_directions=4,
        graph_grid=(4, 4),
        topk=3,
        probe_batch_size=8,
        antithetic_noise=True,
        seed=11,
    )
    extractor = FieldResponseExtractor(backend, config)
    images = torch.rand(2, 3, 32, 32)
    seeds = [101, 202]
    together = extractor.extract(images, noise_seeds=seeds)
    separately = [
        extractor.extract(images[index : index + 1], noise_seeds=[seeds[index]])
        for index in range(2)
    ]
    for index, expected in enumerate(separately):
        actual = slice_features(together, index)
        assert torch.equal(actual.state, expected.state)
        assert torch.equal(actual.response, expected.response)
        assert torch.equal(actual.affinity, expected.affinity)
        assert torch.equal(actual.adjacency, expected.adjacency)
        for name in actual.baselines:
            assert torch.equal(actual.baselines[name], expected.baselines[name])
        for name in actual.graphs:
            assert torch.equal(actual.graphs[name], expected.graphs[name])
    assert together.metadata["noise_policy"] == "sample_id_sha256_seeded_v1"
    assert together.metadata["probe_basis"] == "shared_fixed_seed_v1"


def test_noise_seed_validation() -> None:
    extractor = FieldResponseExtractor(
        ToyFieldBackend(image_size=32),
        ProbeConfig(times=(0.5,), graph_grid=(2, 2)),
    )
    images = torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError, match="length"):
        extractor.extract(images, noise_seeds=[1])
