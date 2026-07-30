import torch

from fieldscope.path import endpoint_estimate, rectified_state, rectified_tangent


def test_rectified_path_endpoints_and_tangent() -> None:
    z0 = torch.tensor([[[[3.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    at_noise = rectified_state(z0, noise, torch.tensor(0.0))
    at_image = rectified_state(z0, noise, torch.tensor(1.0))
    tangent = rectified_tangent(z0, noise)
    assert torch.equal(at_noise, noise)
    assert torch.equal(at_image, z0)
    assert torch.equal(tangent, torch.tensor([[[[4.0]]]]))


def test_endpoint_estimate_recovers_linear_endpoint() -> None:
    z0 = torch.randn(2, 4, 3, 3)
    noise = torch.randn_like(z0)
    time = torch.tensor([0.2, 0.7])
    state = rectified_state(z0, noise, time)
    estimate = endpoint_estimate(state, rectified_tangent(z0, noise), time)
    assert torch.allclose(estimate, z0)

