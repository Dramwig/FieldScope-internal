import torch

from fieldscope.probes import generate_structured_probes


def test_probes_are_reproducible_and_normalized() -> None:
    reference = torch.zeros(2, 4, 8, 8)
    first = generate_structured_probes(reference, 8, seed=17)
    second = generate_structured_probes(reference, 8, seed=17)
    assert first.shape == (2, 8, 4, 8, 8)
    assert torch.equal(first, second)
    rms = first.square().mean(dim=(-3, -2, -1)).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

