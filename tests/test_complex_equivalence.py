import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D
from src.QCNN_layers.IndexedConv_layer import IndexedConv_RBS_density_I2_3D


def test_rbs_conv_real_and_complex_backends_match_with_zero_imaginary_part():
    layer = Conv_RBS_density_I2_3D(I=4, K=2, J=3, kernel_layout="all_connection", device=torch.device("cpu"))
    vector = torch.randn(1, 4 * 4 * 3)
    vector = vector / vector.norm(dim=1, keepdim=True)
    rho = vector[:, :, None] * vector[:, None, :]

    out_real = layer(rho)
    out_complex = layer(rho.to(torch.complex64))

    assert torch.allclose(out_complex.real, out_real, atol=1e-6)
    assert torch.allclose(out_complex.imag, torch.zeros_like(out_complex.imag), atol=1e-6)


def test_indexed_rbs_conv_matches_dense_backend_forward_and_gradients():
    dense = Conv_RBS_density_I2_3D(I=4, K=2, J=3, kernel_layout="all_connection", device=torch.device("cpu"))
    indexed = IndexedConv_RBS_density_I2_3D(I=4, K=2, J=3, kernel_layout="all_connection", device=torch.device("cpu"))
    for dense_param, indexed_param in zip(dense.Parameters, indexed.Parameters):
        indexed_param.data.copy_(dense_param.data)

    vector = torch.randn(2, 4 * 4 * 3, dtype=torch.complex64)
    vector = vector / vector.norm(dim=1, keepdim=True)
    rho_dense = (vector[:, :, None] * vector.conj()[:, None, :]).requires_grad_(True)
    rho_indexed = rho_dense.detach().clone().requires_grad_(True)

    out_dense = dense(rho_dense)
    out_indexed = indexed(rho_indexed)
    assert torch.allclose(out_indexed, out_dense, atol=1e-5)

    loss_dense = out_dense.real.square().sum() + out_dense.imag.square().sum()
    loss_indexed = out_indexed.real.square().sum() + out_indexed.imag.square().sum()
    loss_dense.backward()
    loss_indexed.backward()

    assert torch.allclose(rho_indexed.grad, rho_dense.grad, atol=1e-5)
    for dense_param, indexed_param in zip(dense.Parameters, indexed.Parameters):
        assert torch.allclose(indexed_param.grad, dense_param.grad, atol=1e-5)
