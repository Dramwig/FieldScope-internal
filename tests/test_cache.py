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
        targets={
            "classification": torch.tensor([1]),
            "segmentation": torch.tensor([[[0, 1], [2, 255]]]),
        },
        sample_ids=["sample"],
    )
    restored, targets, restored_manifest = load_features(path)
    assert torch.equal(restored.state, features.state.cpu())
    assert torch.equal(
        restored.graphs["dit_attention"],
        features.graphs["dit_attention"].cpu(),
    )
    assert targets["classification"].item() == 1
    assert restored_manifest["fingerprint"] == manifest["fingerprint"]
    assert restored_manifest["format_version"] == 3
    raw = torch.load(path, weights_only=False)
    assert "adjacency" not in raw["features"]
    assert raw["targets"]["classification"].dtype == torch.int32
    assert raw["targets"]["segmentation"].dtype == torch.uint8
