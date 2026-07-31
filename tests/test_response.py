import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.config import ProbeConfig
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
        "velocity",
        "mismatch",
        "endpoint",
        "dit_hidden",
    }
    assert set(features.graphs) == {
        "dit_attention",
        "dit_attention_adjacency",
    }
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
