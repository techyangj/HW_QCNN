import math

import torch
from torch import nn

from src.complex_backend import complex_dtype_for, ensure_complex_density


def downsample_phase_map(phase_map):
    """Signed 2x2 average used for the multiscale fixed phase map."""
    if phase_map.dim() != 4:
        raise ValueError("phase_map must have shape (batch, I, I, J)")
    return 0.25 * (
        phase_map[:, 0::2, 0::2, :]
        + phase_map[:, 1::2, 0::2, :]
        + phase_map[:, 0::2, 1::2, :]
        + phase_map[:, 1::2, 1::2, :]
    )


def normalize_phase_map(phase_map, eps=1e-12):
    scale = phase_map.abs().flatten(1).amax(dim=1).clamp_min(eps)
    return phase_map / scale[:, None, None, None]


class TensorPhaseReuploadDensity3D(nn.Module):
    """
    Input-dependent diagonal phase D_x for the 3D image tensor basis.

    Each channel has a trainable bounded scale alpha_c = pi * tanh(eta_c).
    Passing eta = 0 makes this layer exactly identity while keeping useful
    gradients through eta.
    """

    def __init__(self, I, J, init_eta=0.0, eps=1e-12, device=None):
        super().__init__()
        self.I = I
        self.J = J
        self.eps = eps
        self.eta = nn.Parameter(torch.full((J,), float(init_eta), device=device))
        basis = torch.arange(I * I * J, device=device)
        self.register_buffer("channel", (basis % J).long())

    def _coerce_phase_map(self, phase_map):
        if phase_map.dim() == 2:
            expected = self.I * self.I * self.J
            if phase_map.shape[1] != expected:
                raise ValueError(f"flattened phase_map must have {expected} columns")
            return phase_map

        if phase_map.dim() != 4:
            raise ValueError("phase_map must be flattened or have shape (batch, I, I, J)")

        if phase_map.shape[1:] == (self.I, self.I, self.J):
            ordered = phase_map
        elif phase_map.shape[1:] == (self.J, self.I, self.I):
            ordered = phase_map.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"phase_map shape must be (batch, {self.I}, {self.I}, {self.J}) "
                f"or (batch, {self.J}, {self.I}, {self.I})"
            )
        return normalize_phase_map(ordered, self.eps).reshape(phase_map.shape[0], -1)

    def phase_vector(self, phase_map, dtype, device):
        flat_map = self._coerce_phase_map(phase_map).to(device=device)
        if phase_map.dim() == 2:
            scale = flat_map.abs().amax(dim=1).clamp_min(self.eps)
            flat_map = flat_map / scale[:, None]
        alpha = (math.pi * torch.tanh(self.eta)).to(device=device)
        angles = flat_map * alpha[self.channel].unsqueeze(0)
        ones = torch.ones_like(angles)
        return torch.polar(ones, angles).to(dtype=complex_dtype_for(dtype))

    def forward(self, rho, phase_map=None):
        if phase_map is None:
            return rho
        rho = ensure_complex_density(rho)
        phase = self.phase_vector(phase_map, rho.dtype, rho.device)
        return rho * phase[:, :, None] * phase.conj()[:, None, :]
