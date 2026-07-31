import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.config import ProbeConfig, TokenizerConfig
from fieldscope.feature_ops import select_representation
from fieldscope.losses import (
    graph_stability_loss,
    multitask_loss,
    representation_consistency_loss,
)
from fieldscope.model import FieldScopeModel
from fieldscope.response import FieldResponseExtractor


def _features():
    config = ProbeConfig(
        times=(0.5,),
        num_directions=2,
        graph_grid=(4, 4),
        topk=3,
        probe_batch_size=4,
        antithetic_noise=False,
    )
    return FieldResponseExtractor(ToyFieldBackend(image_size=32), config).extract(
        torch.rand(2, 3, 32, 32)
    )


def test_multitask_model_shapes_and_backward() -> None:
    features = _features()
    model = FieldScopeModel(
        features.state.shape[-1],
        features.response.shape[-1],
        TokenizerConfig(
            hidden_dim=32,
            num_layers=2,
            num_classes=2,
            segmentation_classes=3,
        ),
    )
    predictions = model(features, output_size=(16, 16))
    assert predictions["classification"].shape == (2, 2)
    assert predictions["segmentation"].shape == (2, 3, 16, 16)
    assert predictions["depth"].shape == (2, 1, 16, 16)
    assert torch.all(predictions["depth"] > 0)
    assert predictions["normals"].shape == (2, 3, 16, 16)

    targets = {
        "classification": torch.tensor([0, 1]),
        "segmentation": torch.zeros(2, 16, 16, dtype=torch.long),
        "depth": torch.ones(2, 1, 16, 16),
        "normals": torch.nn.functional.normalize(torch.randn(2, 3, 16, 16), dim=1),
    }
    total, pieces = multitask_loss(predictions, targets)
    total.backward()
    assert set(pieces) == {"classification", "segmentation", "depth", "normals"}
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_consistency_losses() -> None:
    value = torch.randn(2, 4, 4)
    assert graph_stability_loss(value, value).item() == 0
    assert representation_consistency_loss(value, value).item() == 0


def test_feature_transfer_can_normalize_readout_dtype() -> None:
    features = _features()
    features.state = features.state.to(torch.bfloat16)
    normalized = features.to("cpu", dtype=torch.float32)
    assert normalized.state.dtype == torch.float32
    assert normalized.response.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in normalized.baselines.values())
    assert all(value.dtype == torch.float32 for value in normalized.graphs.values())


def test_readout_parameter_count_is_representation_matched() -> None:
    features = _features()
    config = TokenizerConfig(
        hidden_dim=32,
        input_dim=64,
        num_layers=1,
        num_classes=2,
        segmentation_classes=3,
    )
    counts = []
    for representation in ("z0", "response", "full", "dit_hidden_attention"):
        selected, mode = select_representation(features, representation)
        model = FieldScopeModel(
            selected.state.shape[-1],
            selected.response.shape[-1],
            config,
            mode=mode,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1
