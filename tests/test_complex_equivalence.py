import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D


def test_rbs_conv_real_and_complex_backends_match_with_zero_imaginary_part():
    layer = Conv_RBS_density_I2_3D(I=4, K=2, J=3, kernel_layout="all_connection", device=torch.device("cpu"))
    vector = torch.randn(1, 4 * 4 * 3)
    vector = vector / vector.norm(dim=1, keepdim=True)
    rho = vector[:, :, None] * vector[:, None, :]

    out_real = layer(rho)
    out_complex = layer(rho.to(torch.complex64))

    assert torch.allclose(out_complex.real, out_real, atol=1e-6)
    assert torch.allclose(out_complex.imag, torch.zeros_like(out_complex.imag), atol=1e-6)
