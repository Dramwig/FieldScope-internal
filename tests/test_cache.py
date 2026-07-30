from pathlib import Path

import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.cache import load_features, save_features
from fieldscope.config import ProbeConfig
from fieldscope.response import FieldResponseExtractor


def test_feature_cache_round_trip(tmp_path: Path) -> None:
    config = ProbeConfig(
        times=(0.5,),
        num_directions=2,
        graph_grid=(2, 2),
        probe_batch_size=4,
        antithetic_noise=False,
    )
    features = FieldResponseExtractor(ToyFieldBackend(image_size=32), config).extract(
        torch.rand(1, 3, 32, 32)
    )
    path = tmp_path / "features.pt"
    manifest = save_features(
        path,
        features,
        targets={"classification": torch.tensor([1])},
        sample_ids=["sample"],
    )
    restored, targets, restored_manifest = load_features(path)
    assert torch.equal(restored.state, features.state.cpu())
    assert targets["classification"].item() == 1
    assert restored_manifest["fingerprint"] == manifest["fingerprint"]

