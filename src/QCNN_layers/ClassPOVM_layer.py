import torch
from torch import nn


def balanced_class_index(num_basis, num_classes=10, device=None):
    """Deterministically partition basis states into balanced class groups."""
    return torch.arange(num_basis, device=device, dtype=torch.long) % num_classes


class FullSpaceClassPOVM(nn.Module):
    """
    Trace-preserving multiclass readout over the complete output basis.

    Unlike Trace_out_dimension, this layer does not crop and renormalize a 10x10
    submatrix. It sums the full diagonal probability mass into num_classes bins.
    """

    def __init__(self, num_basis, num_classes=10, class_index=None, eps=1e-12, device=None):
        super().__init__()
        self.num_basis = num_basis
        self.num_classes = num_classes
        self.eps = eps
        if class_index is None:
            class_index = balanced_class_index(num_basis, num_classes, device)
        if class_index.numel() != num_basis:
            raise ValueError("class_index must have one entry per basis state")
        if class_index.min() < 0 or class_index.max() >= num_classes:
            raise ValueError("class_index values must be in [0, num_classes)")
        self.register_buffer("class_index", class_index.long())

    def forward(self, rho):
        prob_basis = torch.diagonal(rho, dim1=-2, dim2=-1).real
        prob_class = torch.zeros(
            prob_basis.shape[0],
            self.num_classes,
            device=prob_basis.device,
            dtype=prob_basis.dtype,
        )
        class_index = self.class_index.to(prob_basis.device).unsqueeze(0).expand_as(prob_basis)
        prob_class.scatter_add_(1, class_index, prob_basis)
        normalizer = prob_class.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return prob_class / normalizer


class QuantumNLLLoss(nn.Module):
    """Negative log likelihood for already-normalized quantum class probabilities."""

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, probabilities, target):
        selected = probabilities.gather(1, target.view(-1, 1)).clamp_min(self.eps)
        return -selected.log().mean()
