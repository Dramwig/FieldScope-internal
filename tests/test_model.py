import torch

from fieldscope.backends.toy import ToyFieldBackend
from fieldscope.config import ProbeConfig, TokenizerConfig
from fieldscope.feature_ops import select_representation
from fieldscope.graph import (
    fixed_grid_adjacency,
    fixed_normalized_grid_adjacency,
    normalize_adjacency,
)
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


def test_model_can_execute_only_the_requested_task_head() -> None:
    features = _features()
    model = FieldScopeModel(
        features.state.shape[-1],
        features.response.shape[-1],
        TokenizerConfig(
            hidden_dim=16,
            input_dim=16,
            num_layers=1,
            num_classes=4,
            segmentation_classes=3,
        ),
    )
    predictions = model(features, output_size=(16, 16), task="classification")
    assert set(predictions) == {"classification"}
    assert predictions["classification"].shape == (2, 4)


def test_cached_local_adjacency_matches_direct_normalization_exactly() -> None:
    device = torch.device("cpu")
    direct = normalize_adjacency(fixed_grid_adjacency((4, 4), 3, device, torch.float32))
    cached = fixed_normalized_grid_adjacency(
        (4, 4),
        3,
        device,
        torch.float32,
    )
    assert torch.equal(direct, cached)
    assert cached.stride(0) == 0


def test_model_fast_path_matches_validated_forward_exactly() -> None:
    features = _features()
    selected, mode = select_representation(features, "full_local")
    model = FieldScopeModel(
        selected.state.shape[-1],
        selected.response.shape[-1],
        TokenizerConfig(
            hidden_dim=16,
            input_dim=16,
            num_layers=1,
            num_classes=4,
            segmentation_classes=3,
        ),
        mode=mode,
    ).eval()
    validated = model(
        selected,
        output_size=(4, 4),
        task="classification",
        validate_features=True,
    )
    fast = model(
        selected,
        output_size=(4, 4),
        task="classification",
        validate_features=False,
    )
    assert torch.equal(validated["classification"], fast["classification"])


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
    for representation in (
        "random_feature_local",
        "z0",
        "zt",
        "trajectory",
        "velocity",
        "mismatch",
        "endpoint",
        "state",
        "state_nograph",
        "state_graph",
        "response_nograph",
        "response_local",
        "response",
        "full_nograph",
        "full_local",
        "full",
        "dit_hidden_local",
        "dit_hidden_attention",
        "response_shuffled",
        "full_shuffled",
    ):
        selected, mode = select_representation(features, representation)
        model = FieldScopeModel(
            selected.state.shape[-1],
            selected.response.shape[-1],
            config,
            mode=mode,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1
