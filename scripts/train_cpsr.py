import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d
from torch.optim.lr_scheduler import ExponentialLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.CPSR_HW_QCNN import CPSRHWQCNN
from scripts.reproduce_official import reproducible_cifar10_loaders, resolve_device, set_seed
from src.QCNN_layers.ClassPOVM_layer import QuantumNLLLoss
from src.QCNN_layers.DataReupload_layer import downsample_phase_map
from src.load_dataset import copy_images_bottom_channel_stride, to_density_matrix
from src.toolbox import normalize_DM


def copy_phase_channels_from_grayscale(gray, J, stride):
    """Build a signed I x I x J tensor phase map from the fixed grayscale input."""
    batch, I, _ = gray.shape
    flat = gray.reshape(batch, I * I)
    channels = []
    for channel in range(J):
        channels.append(torch.roll(flat, shifts=channel * stride, dims=1))
    return torch.stack(channels, dim=-1).reshape(batch, I, I, J)


def preprocess_batch(data, target, I, J, stride, device):
    adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
    data = adaptive_avg_pool(data).to(device)
    gray = data.sum(dim=1, keepdim=True)
    target = target.squeeze().to(device)

    flat = gray.reshape(gray.shape[0], I * I)
    vectors = F.normalize(flat, p=2, dim=1)
    init_density_matrix = to_density_matrix(vectors, device)
    channel_density = normalize_DM(copy_images_bottom_channel_stride(init_density_matrix, J, stride)).to(device)

    phase_stage1 = copy_phase_channels_from_grayscale(gray.squeeze(1), J, stride).to(device)
    phase_stage2 = downsample_phase_map(phase_stage1).to(device)
    return channel_density, [phase_stage1, phase_stage2], target


def compute_loss(output, target, readout, output_scale, ce_loss, nll_loss):
    if readout == "full_space":
        return nll_loss(output, target)
    return ce_loss(output * output_scale, target)


def run_epoch(model, loader, optimizer, args, device, train=True):
    model.train(train)
    ce_loss = torch.nn.CrossEntropyLoss()
    nll_loss = QuantumNLLLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for data, target in loader:
        if train:
            optimizer.zero_grad()
        density, phase_maps, target = preprocess_batch(data, target, args.I, args.J, args.stride, device)
        output = model(density, phase_maps)
        loss = compute_loss(output, target, args.readout, args.output_scale, ce_loss, nll_loss)
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_seen += target.numel()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1), total_correct / max(total_seen, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CPSR-HW-QCNN on the CIFAR-10 protocol.")
    parser.add_argument("--I", type=int, default=16)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--kernel-layout", default="all_connection")
    parser.add_argument("--phase-mode", choices=["lite", "full"], default="lite")
    parser.add_argument("--phase-rank", type=int, default=2)
    parser.add_argument("--blocks-per-stage", type=int, choices=[1, 2], default=2)
    parser.add_argument("--no-phase", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--no-reupload", action="store_true")
    parser.add_argument("--readout", choices=["truncated", "full_space"], default="truncated")
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2 * 0.66)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--output-scale", type=float, default=50.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "cpsr_cifar10")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = reproducible_cifar10_loaders(
        args.train_count,
        args.test_count,
        args.batch_size,
        args.seed,
        args.output_dir,
    )

    model = CPSRHWQCNN(
        I=args.I,
        J=args.J,
        K=args.K,
        k=args.k,
        kernel_layout=args.kernel_layout,
        device=device,
        phase_mode=args.phase_mode,
        phase_rank=args.phase_rank,
        use_phase=not args.no_phase,
        use_shift=not args.no_shift,
        use_reupload=not args.no_reupload,
        readout=args.readout,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    print(f"Start training! Number of network total parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)

    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, args, device, train=True)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc * 100)
        print(
            f"Epoch {epoch}: Loss = {train_loss:.6f}, "
            f"accuracy = {train_acc * 100:.4f} %, time={(time.time() - start):.4f}s"
        )
        if (epoch + 1) % args.test_interval == 0:
            with torch.no_grad():
                test_loss, test_acc = run_epoch(model, test_loader, optimizer, args, device, train=False)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc * 100)
            print(f"Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_acc * 100:.4f} %")
        scheduler.step()

    torch.save(model.state_dict(), args.output_dir / f"cpsr_seed{args.seed}.pt")
    np.save(
        args.output_dir / f"cpsr_seed{args.seed}.npy",
        {
            "train_loss": train_loss_list,
            "train_accuracy": train_acc_list,
            "test_loss": test_loss_list,
            "test_accuracy": test_acc_list,
            "args": vars(args),
        },
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()
