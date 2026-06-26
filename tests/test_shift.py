import pytest

torch = pytest.importorskip("torch")

from src.QCNN_layers.ShiftPermutation_layer import ShiftPermutationDensity3D


def test_shift_inverse_restores_density():
    layer = ShiftPermutationDensity3D(I=4, J=3, shift_r=2, shift_c=2)
    rho = torch.randn(2, 4 * 4 * 3, 4 * 4 * 3, dtype=torch.complex64)
    shifted = layer(rho)
    restored = layer.inverse(shifted)
    assert torch.allclose(restored, rho)
