import torch


def complex_dtype_for(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


def real_dtype_for(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def ensure_complex_density(rho):
    if torch.is_complex(rho):
        return rho
    return rho.to(dtype=complex_dtype_for(rho.dtype))


def conjugate_by_diagonal_phase(rho, phase):
    rho = ensure_complex_density(rho)
    phase = phase.to(device=rho.device, dtype=rho.dtype)
    return rho * phase[:, :, None] * phase.conj()[:, None, :]


def normalize_density_matrix(rho, eps=1e-12):
    traces = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return rho / (traces[:, None, None] + eps)


def density_physicality(rho):
    hermiticity_error = (rho - rho.mH).abs().amax()
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_error = (trace - 1).abs().amax()
    eigvals = torch.linalg.eigvalsh(rho)
    return {
        "trace_error": trace_error.real,
        "hermiticity_error": hermiticity_error.real,
        "min_eigenvalue": eigvals.amin().real,
    }
