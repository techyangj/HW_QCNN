import os
import sys

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.toolbox import QCNN_RBS_based_VQC_3D, RBS_generalized_I2_3D_bottom_channel


class IndexedRBSConvDensity(nn.Module):
    """
    Exact density-matrix RBS conjugation using the affected basis pairs only.

    This is mathematically equivalent to U @ rho @ U^H for the real RBS unitary,
    but avoids materializing and multiplying by the full dense N x N unitary.
    """

    def __init__(self, gate_impact, theta, device=None):
        super().__init__()
        self.angle = theta
        if gate_impact:
            left, right = zip(*gate_impact)
        else:
            left, right = (), ()
        self.register_buffer("left_indices", torch.tensor(left, dtype=torch.long, device=device))
        self.register_buffer("right_indices", torch.tensor(right, dtype=torch.long, device=device))

    def _indices(self, device):
        if self.left_indices.device == device:
            return self.left_indices, self.right_indices
        return self.left_indices.to(device), self.right_indices.to(device)

    def forward(self, rho):
        if self.left_indices.numel() == 0:
            return rho

        left, right = self._indices(rho.device)
        cos_theta = torch.cos(self.angle)
        sin_theta = torch.sin(self.angle)

        out = rho.clone()

        left_rows = out.index_select(1, left)
        right_rows = out.index_select(1, right)
        out[:, left, :] = cos_theta * left_rows + sin_theta * right_rows
        out[:, right, :] = -sin_theta * left_rows + cos_theta * right_rows

        left_cols = out.index_select(2, left)
        right_cols = out.index_select(2, right)
        out[:, :, left] = cos_theta * left_cols + sin_theta * right_cols
        out[:, :, right] = -sin_theta * left_cols + cos_theta * right_cols

        return out


class IndexedConv_RBS_density_I2_3D(nn.Module):
    """
    Indexed backend for the 3D image-basis RBS convolution.

    It preserves the public parameter layout of Conv_RBS_density_I2_3D so weights
    can be copied between dense and indexed backends for equivalence tests or
    warm starts.
    """

    def __init__(self, I, K, J, kernel_layout, device):
        super().__init__()
        list_gates = []
        _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC_3D(I, K, J, kernel_layout)
        for key in RBS_dictionary:
            list_gates.append(RBS_dictionary[key])

        self.Parameters = nn.ParameterList(
            [
                nn.Parameter(torch.rand((), device=device), requires_grad=True)
                for _ in range(int(K * (K - 1)) + J * (J - 1) // 2)
            ]
        )
        self.RBS_gates = nn.ModuleList(
            [
                IndexedRBSConvDensity(
                    RBS_generalized_I2_3D_bottom_channel(*list_gates[i], I, J),
                    self.Parameters[Param_dictionary[i]],
                    device,
                )
                for i in range(len(list_gates))
            ]
        )

    def forward(self, input_state):
        for rbs in self.RBS_gates:
            input_state = rbs(input_state)
        return input_state
