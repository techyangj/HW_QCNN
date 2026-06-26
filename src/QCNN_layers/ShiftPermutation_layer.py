import torch
from torch import nn


def shifted_source_permutation(I, J, shift_r, shift_c, device=None):
    """
    Return source indices for output[new] = input[source].

    With forward shift (row, col) -> (row + shift_r, col + shift_c), the source
    of an output coordinate is the inverse-shifted coordinate.
    """
    sources = []
    for row in range(I):
        for col in range(I):
            src_row = (row - shift_r) % I
            src_col = (col - shift_c) % I
            for channel in range(J):
                sources.append((src_row * I + src_col) * J + channel)
    return torch.tensor(sources, dtype=torch.long, device=device)


class ShiftPermutationDensity3D(nn.Module):
    """Cyclic spatial shift on the 3D image tensor density matrix basis."""

    def __init__(self, I, J, shift_r, shift_c=None, device=None):
        super().__init__()
        if shift_c is None:
            shift_c = shift_r
        self.I = I
        self.J = J
        self.shift_r = shift_r
        self.shift_c = shift_c
        self.register_buffer("perm", shifted_source_permutation(I, J, shift_r, shift_c, device))
        self.register_buffer("inverse_perm", shifted_source_permutation(I, J, -shift_r, -shift_c, device))

    def forward(self, rho):
        return rho[:, self.perm][:, :, self.perm]

    def inverse(self, rho):
        return rho[:, self.inverse_perm][:, :, self.inverse_perm]


class ShiftedBlock3D(nn.Module):
    """Apply S^-1 Q S for a density-matrix block Q."""

    def __init__(self, block, I, J, shift_r, shift_c=None, device=None):
        super().__init__()
        self.shift = ShiftPermutationDensity3D(I, J, shift_r, shift_c, device)
        self.block = block

    def forward(self, rho):
        return self.shift.inverse(self.block(self.shift(rho)))
