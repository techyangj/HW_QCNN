import pytest

torch = pytest.importorskip("torch")

from src.QCNN_layers.ClassPOVM_layer import FullSpaceClassPOVM


def test_full_space_povm_sums_all_probability_mass():
    diagonal = torch.tensor([[0.05, 0.10, 0.20, 0.15, 0.25, 0.25]], dtype=torch.float32)
    rho = torch.diag_embed(diagonal).to(torch.complex64)
    readout = FullSpaceClassPOVM(num_basis=6, num_classes=3)

    probs = readout(rho)

    assert probs.shape == (1, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1))
    assert torch.allclose(probs, torch.tensor([[0.20, 0.35, 0.45]]))
