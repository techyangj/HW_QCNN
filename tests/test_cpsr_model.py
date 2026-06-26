import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from models.CPSR_HW_QCNN import CPSRHWQCNN


def pure_density(batch, dim):
    vector = torch.randn(batch, dim, dtype=torch.complex64)
    vector = vector / vector.norm(dim=1, keepdim=True)
    return vector[:, :, None] * vector.conj()[:, None, :]


def test_small_cpsr_full_space_forward_backward():
    model = CPSRHWQCNN(
        I=4,
        J=2,
        K=2,
        k=3,
        class_count=3,
        phase_rank=1,
        readout="full_space",
        device=torch.device("cpu"),
    )
    rho = pure_density(batch=2, dim=4 * 4 * 2)
    phase_maps = [torch.randn(2, 4, 4, 2), torch.randn(2, 2, 2, 2)]

    output = model(rho, phase_maps)
    loss = -output[:, 0].clamp_min(1e-8).log().mean()
    loss.backward()

    assert output.shape == (2, 3)
    assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-5)
    assert not torch.isnan(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
