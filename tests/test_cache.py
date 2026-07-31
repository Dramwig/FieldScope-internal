from pathlib import Path

import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.cache import load_features, save_features
from fieldscope.config import ProbeConfig, TokenizerConfig
from fieldscope.feature_ops import select_representation
from fieldscope.model import FieldScopeModel
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
    assert restored_manifest["format_version"] == 4
    assert restored_manifest["storage_policy"] == "dense"
    raw = torch.load(path, weights_only=False)
    assert "adjacency" not in raw["features"]
    assert raw["targets"]["classification"].dtype == torch.int32
    assert raw["targets"]["segmentation"].dtype == torch.uint8


def test_readout_sparse_cache_is_lossless_and_smaller(tmp_path: Path) -> None:
    config = ProbeConfig(
        times=(0.2, 0.5, 0.8),
        num_directions=4,
        graph_grid=(16, 16),
        topk=2,
        local_radius=0,
        probe_batch_size=8,
        antithetic_noise=True,
    )
    features = FieldResponseExtractor(ToyFieldBackend(image_size=32), config).extract(
        torch.rand(2, 3, 32, 32), noise_seeds=[101, 202]
    )
    targets = {"classification": torch.tensor([1, 0])}
    dense_path = tmp_path / "dense.pt"
    sparse_path = tmp_path / "sparse.pt"
    save_features(
        dense_path,
        features,
        targets=targets,
        sample_ids=["a", "b"],
        storage_policy="dense",
    )
    sparse_manifest = save_features(
        sparse_path,
        features,
        targets=targets,
        sample_ids=["a", "b"],
        storage_policy="readout_sparse",
    )
    restored, restored_targets, restored_manifest = load_features(sparse_path)
    assert sparse_manifest == restored_manifest
    assert restored_manifest["storage_policy"] == "readout_sparse"
    assert restored_manifest["dense_affinity_available"] is False
    assert sparse_path.stat().st_size < dense_path.stat().st_size * 0.7
    assert torch.equal(restored.state, features.state.cpu())
    assert torch.equal(restored.response, features.response.cpu())
    assert torch.equal(restored.adjacency, features.adjacency.cpu())
    assert torch.equal(restored.affinity, features.adjacency.cpu())
    assert torch.equal(
        restored.graphs["dit_attention_adjacency"],
        features.graphs["dit_attention_adjacency"].cpu(),
    )
    assert "dit_attention" not in restored.graphs
    for name in features.baselines:
        assert torch.equal(restored.baselines[name], features.baselines[name].cpu())
    assert torch.equal(restored_targets["classification"], targets["classification"].int())
    raw = torch.load(sparse_path, weights_only=False)
    assert raw["features"]["adjacency"]["format"] == (
        "packed_weighted_adjacency_v1"
    )
    assert raw["features"]["adjacency"]["indices"].dtype == torch.uint8

    dense_features, _, _ = load_features(dense_path)
    tokenizer_config = TokenizerConfig(
        hidden_dim=16,
        input_dim=32,
        num_layers=1,
        dropout=0.0,
        num_classes=2,
        segmentation_classes=3,
    )
    for representation in (
        "state",
        "response",
        "full",
        "dit_hidden_local",
        "dit_hidden_attention",
    ):
        dense_selected, mode = select_representation(
            dense_features, representation
        )
        sparse_selected, sparse_mode = select_representation(
            restored, representation
        )
        assert sparse_mode == mode
        model = FieldScopeModel(
            dense_selected.state.shape[-1],
            dense_selected.response.shape[-1],
            tokenizer_config,
            mode=mode,
        ).eval()
        with torch.no_grad():
            dense_output = model(dense_selected, output_size=(16, 16))
            sparse_output = model(sparse_selected, output_size=(16, 16))
        for name in dense_output:
            assert torch.equal(dense_output[name], sparse_output[name])
