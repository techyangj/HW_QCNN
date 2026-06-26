import torch
from torch import nn

from src.complex_backend import complex_dtype_for, ensure_complex_density


class PhaseCouplingDensity3D(nn.Module):
    """
    Low-rank cross-register diagonal phase layer for the 3D image tensor basis.

    The flattened basis is q = (row * I + column) * J + channel. The layer applies
    rho[q, r] <- exp(i phi[q]) rho[q, r] exp(-i phi[r]) while preserving the
    row/column/channel Hamming-weight-one tensor subspace.
    """

    def __init__(self, I, K, J, rank=2, mode="lite", init_scale=1e-2, device=None):
        super().__init__()
        if mode not in {"lite", "full"}:
            raise ValueError("mode must be 'lite' or 'full'")
        if I % K != 0:
            raise ValueError("I must be divisible by K for shared local windows")

        self.I = I
        self.K = K
        self.J = J
        self.rank = rank
        self.mode = mode

        self.rf_row = nn.Parameter(torch.empty(K, rank, device=device))
        self.rf_feat = nn.Parameter(torch.zeros(J, rank, device=device))
        self.cf_col = nn.Parameter(torch.empty(K, rank, device=device))
        self.cf_feat = nn.Parameter(torch.zeros(J, rank, device=device))
        if mode == "full":
            self.rc_row = nn.Parameter(torch.empty(K, rank, device=device))
            self.rc_col = nn.Parameter(torch.zeros(K, rank, device=device))
        else:
            self.register_parameter("rc_row", None)
            self.register_parameter("rc_col", None)

        self.reset_parameters(init_scale)
        self._register_basis_tables(device)

    def reset_parameters(self, init_scale):
        nn.init.normal_(self.rf_row, mean=0.0, std=init_scale)
        nn.init.normal_(self.cf_col, mean=0.0, std=init_scale)
        if self.rc_row is not None:
            nn.init.normal_(self.rc_row, mean=0.0, std=init_scale)

    def _register_basis_tables(self, device):
        basis = torch.arange(self.I * self.I * self.J, device=device)
        row = basis // (self.I * self.J)
        column = (basis // self.J) % self.I
        channel = basis % self.J
        self.register_buffer("local_row", (row % self.K).long())
        self.register_buffer("local_col", (column % self.K).long())
        self.register_buffer("channel", channel.long())

    def phase_tables(self):
        theta_rf = self.rf_row @ self.rf_feat.T
        theta_cf = self.cf_col @ self.cf_feat.T
        theta_rc = None
        if self.mode == "full":
            theta_rc = self.rc_row @ self.rc_col.T
        return theta_rc, theta_rf, theta_cf

    def phase_angles(self):
        theta_rc, theta_rf, theta_cf = self.phase_tables()
        phi = theta_rf[self.local_row, self.channel] + theta_cf[self.local_col, self.channel]
        if theta_rc is not None:
            phi = phi + theta_rc[self.local_row, self.local_col]
        return phi

    def phase_vector(self, batch_size, dtype, device):
        complex_dtype = complex_dtype_for(dtype)
        phi = self.phase_angles().to(device=device)
        ones = torch.ones_like(phi, device=device)
        phase = torch.polar(ones, phi).to(dtype=complex_dtype)
        return phase.unsqueeze(0).expand(batch_size, -1)

    def forward(self, rho):
        rho = ensure_complex_density(rho)
        phase = self.phase_vector(rho.shape[0], rho.dtype, rho.device)
        return rho * phase[:, :, None] * phase.conj()[:, None, :]


class PhaseCoupledConvBlock3D(nn.Module):
    """
    M_out D_theta M_in block built from the official separable RBS convolution.
    """

    def __init__(self, I, K, J, kernel_layout, device, rank=2, mode="lite", conv_cls=None):
        super().__init__()
        if conv_cls is None:
            from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D

            conv_cls = Conv_RBS_density_I2_3D
        self.mixer_in = conv_cls(I, K, J, kernel_layout, device)
        self.phase = PhaseCouplingDensity3D(I, K, J, rank=rank, mode=mode, device=device)
        self.mixer_out = conv_cls(I, K, J, kernel_layout, device)

    def forward(self, rho):
        return self.mixer_out(self.phase(self.mixer_in(rho)))


def phase_parameter_count(K, J, rank=2, mode="lite"):
    if mode == "lite":
        return 2 * (K + J) * rank
    if mode == "full":
        return (4 * K + 2 * J) * rank
    raise ValueError("mode must be 'lite' or 'full'")
