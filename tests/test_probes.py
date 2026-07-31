import torch

from fieldscope.probes import generate_probes, generate_structured_probes


def test_probes_are_reproducible_and_normalized() -> None:
    reference = torch.zeros(2, 4, 8, 8)
    first = generate_structured_probes(reference, 8, seed=17)
    second = generate_structured_probes(reference, 8, seed=17)
    assert first.shape == (2, 8, 4, 8, 8)
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])
    rms = first.square().mean(dim=(-3, -2, -1)).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_probe_controls_are_reproducible_and_distinct() -> None:
    reference = torch.zeros(1, 4, 8, 8)
    structured = generate_probes(reference, 4, 23, "structured")
    gaussian = generate_probes(reference, 4, 23, "gaussian")
    shuffled = generate_probes(reference, 4, 23, "spatially_shuffled")
    assert torch.equal(gaussian, generate_probes(reference, 4, 23, "gaussian"))
    assert torch.equal(
        shuffled,
        generate_probes(reference, 4, 23, "spatially_shuffled"),
    )
    assert not torch.equal(structured, gaussian)
    assert not torch.equal(structured, shuffled)
    for probes in (gaussian, shuffled):
        rms = probes.square().mean(dim=(-3, -2, -1)).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_all_probe_controls_share_one_basis_across_batch() -> None:
    reference = torch.zeros(3, 4, 8, 8)
    for probe_type in ("structured", "gaussian", "spatially_shuffled"):
        probes = generate_probes(reference, 4, 29, probe_type)
        assert torch.equal(probes[0], probes[1])
        assert torch.equal(probes[0], probes[2])
