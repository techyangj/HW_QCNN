import math
import os
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.QCNN_layers.ClassPOVM_layer import FullSpaceClassPOVM
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D
from src.QCNN_layers.DataReupload_layer import TensorPhaseReuploadDensity3D
from src.QCNN_layers.Dense_layer import (
    Basis_Change_I_to_HW_density_3D,
    Dense_RBS_density_3D,
    Trace_out_dimension,
)
from src.QCNN_layers.Measurement_layer import measurement
from src.QCNN_layers.PhaseCoupling_layer import PhaseCoupledConvBlock3D
from src.QCNN_layers.Pooling_layer import Pooling_3D_density
from src.QCNN_layers.ShiftPermutation_layer import ShiftedBlock3D
from src.list_gates import full_connection_circuit, half_connection_circuit, slide_circuit


def official_dense_gates(I, J, reduced_qubit=5):
    final_spatial = I // 4
    dense_full_qubits = 2 * final_spatial + J
    dense_full_gates = (
        half_connection_circuit(dense_full_qubits)
        + full_connection_circuit(dense_full_qubits)
        + half_connection_circuit(dense_full_qubits)
        + full_connection_circuit(dense_full_qubits)
        + slide_circuit(dense_full_qubits - 1)
    )
    dense_reduce_gates = (
        half_connection_circuit(reduced_qubit)
        + full_connection_circuit(reduced_qubit)
        + half_connection_circuit(reduced_qubit)
        + slide_circuit(reduced_qubit)
    )
    return dense_full_gates, dense_reduce_gates


class OfficialHWQCNN3D(nn.Module):
    """
    Paper/official 3D HW-QCNN topology, packaged as a reusable module.
    """

    def __init__(
        self,
        I=16,
        J=7,
        K=4,
        k=3,
        kernel_layout="all_connection",
        class_count=10,
        reduced_qubit=5,
        device=torch.device("cpu"),
    ):
        super().__init__()
        if I % 4 != 0:
            raise ValueError("I must be divisible by 4 for two pooling layers")
        O = I // 2
        final_I = I // 4
        dense_full_gates, dense_reduce_gates = official_dense_gates(I, J, reduced_qubit)

        self.device = device
        self.conv1 = Conv_RBS_density_I2_3D(I, K, J, kernel_layout, device)
        self.pool1 = Pooling_3D_density(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O, K, J, kernel_layout, device)
        self.pool2 = Pooling_3D_density(O, final_I, J, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(final_I, J, k, device)
        self.dense_full1 = Dense_RBS_density_3D(final_I, J, k, dense_full_gates, device)
        self.reduce_dim = Trace_out_dimension(class_count, device)
        self.dense_reduced = Dense_RBS_density_3D(0, reduced_qubit, k, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.basis_map(x)
        x = self.dense_full1(x)
        x = self.dense_reduced(self.reduce_dim(x))
        return measurement(x, self.device)


class CPSRHWQCNN(nn.Module):
    """
    Coupled-phase shifted-window data-reuploading HW-QCNN.

    phase_maps, when supplied, should be [Z_stage1, Z_stage2] with shapes
    (batch, I, I, J) and (batch, I/2, I/2, J).
    """

    def __init__(
        self,
        I=16,
        J=7,
        K=4,
        k=3,
        kernel_layout="all_connection",
        class_count=10,
        reduced_qubit=5,
        device=torch.device("cpu"),
        phase_mode="lite",
        phase_rank=2,
        use_phase=True,
        use_shift=True,
        use_reupload=True,
        readout="truncated",
        blocks_per_stage=2,
        conv_backend="dense",
        checkpoint_blocks=False,
    ):
        super().__init__()
        if I % 4 != 0:
            raise ValueError("I must be divisible by 4 for two pooling layers")
        if blocks_per_stage not in {1, 2}:
            raise ValueError("blocks_per_stage must be 1 or 2")
        if readout not in {"truncated", "full_space"}:
            raise ValueError("readout must be 'truncated' or 'full_space'")
        if conv_backend not in {"dense", "indexed"}:
            raise ValueError("conv_backend must be 'dense' or 'indexed'")

        self.I = I
        self.J = J
        self.K = K
        self.k = k
        self.device = device
        self.use_reupload = use_reupload
        self.use_shift = use_shift
        self.readout = readout
        self.blocks_per_stage = blocks_per_stage
        self.conv_backend = conv_backend
        self.checkpoint_blocks = checkpoint_blocks
        conv_cls = self._resolve_conv_cls(conv_backend)

        O = I // 2
        final_I = I // 4
        dense_full_gates, dense_reduce_gates = official_dense_gates(I, J, reduced_qubit)

        self.reupload1 = TensorPhaseReuploadDensity3D(I, J, device=device) if use_reupload else None
        self.reupload2 = TensorPhaseReuploadDensity3D(O, J, device=device) if use_reupload else None

        self.stage1_block1 = self._make_block(I, K, J, kernel_layout, device, phase_mode, phase_rank, use_phase, conv_cls)
        self.stage2_block1 = self._make_block(O, K, J, kernel_layout, device, phase_mode, phase_rank, use_phase, conv_cls)
        if blocks_per_stage == 2:
            stage1_block2 = self._make_block(I, K, J, kernel_layout, device, phase_mode, phase_rank, use_phase, conv_cls)
            stage2_block2 = self._make_block(O, K, J, kernel_layout, device, phase_mode, phase_rank, use_phase, conv_cls)
            shift = K // 2
            self.stage1_block2 = (
                ShiftedBlock3D(stage1_block2, I, J, shift, shift, device) if use_shift else stage1_block2
            )
            self.stage2_block2 = (
                ShiftedBlock3D(stage2_block2, O, J, shift, shift, device) if use_shift else stage2_block2
            )
        else:
            self.stage1_block2 = None
            self.stage2_block2 = None

        self.pool1 = Pooling_3D_density(I, O, J, device)
        self.pool2 = Pooling_3D_density(O, final_I, J, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(final_I, J, k, device)
        self.dense_full1 = Dense_RBS_density_3D(final_I, J, k, dense_full_gates, device)

        if readout == "full_space":
            dense_basis = math.comb(2 * final_I + J, k)
            self.class_readout = FullSpaceClassPOVM(dense_basis, class_count, device=device)
            self.reduce_dim = None
            self.dense_reduced = None
        else:
            self.class_readout = None
            self.reduce_dim = Trace_out_dimension(class_count, device)
            self.dense_reduced = Dense_RBS_density_3D(0, reduced_qubit, k, dense_reduce_gates, device)

    def _resolve_conv_cls(self, conv_backend):
        if conv_backend == "dense":
            return Conv_RBS_density_I2_3D
        from src.QCNN_layers.IndexedConv_layer import IndexedConv_RBS_density_I2_3D

        return IndexedConv_RBS_density_I2_3D

    def _make_block(self, I, K, J, kernel_layout, device, phase_mode, phase_rank, use_phase, conv_cls):
        if use_phase:
            return PhaseCoupledConvBlock3D(
                I,
                K,
                J,
                kernel_layout,
                device,
                rank=phase_rank,
                mode=phase_mode,
                conv_cls=conv_cls,
            )
        return conv_cls(I, K, J, kernel_layout, device)

    def _apply_block(self, block, x):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _run_stage(self, x, reupload, phase_map, block1, block2, pool):
        if self.use_reupload:
            x = reupload(x, phase_map)
        x = self._apply_block(block1, x)
        if block2 is not None:
            x = self._apply_block(block2, x)
        return pool(x)

    def forward(self, x, phase_maps=None):
        if phase_maps is None:
            phase_maps = [None, None]
        x = self._run_stage(x, self.reupload1, phase_maps[0], self.stage1_block1, self.stage1_block2, self.pool1)
        x = self._run_stage(x, self.reupload2, phase_maps[1], self.stage2_block1, self.stage2_block2, self.pool2)
        x = self.basis_map(x)
        x = self.dense_full1(x)
        if self.readout == "full_space":
            return self.class_readout(x)
        x = self.dense_reduced(self.reduce_dim(x))
        return measurement(x, self.device)
