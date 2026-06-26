import pytest

torch = pytest.importorskip("torch")

from src.QCNN_layers.PhaseCoupling_layer import PhaseCouplingDensity3D


def pure_density(batch, dim):
    vector = torch.randn(batch, dim, dtype=torch.complex64)
    vector = vector / vector.norm(dim=1, keepdim=True)
    return vector[:, :, None] * vector.conj()[:, None, :]


def test_phase_layer_preserves_trace_hermiticity_and_diagonal():
    layer = PhaseCouplingDensity3D(I=4, K=2, J=3, rank=2, mode="full")
    with torch.no_grad():
        layer.rf_feat.normal_(0.0, 0.2)
        layer.cf_feat.normal_(0.0, 0.2)
        layer.rc_col.normal_(0.0, 0.2)

    rho = pure_density(batch=2, dim=4 * 4 * 3)
    out = layer(rho)

    assert torch.allclose(
        out.diagonal(dim1=-2, dim2=-1).real,
        rho.diagonal(dim1=-2, dim2=-1).real,
        atol=1e-6,
    )
    assert torch.allclose(out, out.mH, atol=1e-6)
    assert torch.allclose(
        out.diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        torch.ones(2, dtype=out.dtype),
        atol=1e-6,
    )
    assert torch.linalg.eigvalsh(out).amin().real > -1e-5


def test_zero_phase_is_identity():
    layer = PhaseCouplingDensity3D(I=4, K=2, J=3, rank=2, mode="lite")
    rho = pure_density(batch=1, dim=4 * 4 * 3)
    assert torch.allclose(layer(rho), rho, atol=1e-6)
