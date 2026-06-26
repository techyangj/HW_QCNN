import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.CPSR_HW_QCNN import OfficialHWQCNN3D
from src.load_dataset import load_cifar10
from src.training import train_globally


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def stratified_indices(dataset, labels, count, seed):
    rng = torch.Generator().manual_seed(seed)
    by_class = {label: [] for label in labels}
    for idx, (_, label) in enumerate(dataset):
        if label in by_class:
            by_class[label].append(idx)
    per_class = count // len(labels)
    remainder = count % len(labels)
    selected = []
    for order, label in enumerate(labels):
        indices = torch.tensor(by_class[label])
        perm = indices[torch.randperm(indices.numel(), generator=rng)]
        take = per_class + (1 if order < remainder else 0)
        selected.extend(perm[:take].tolist())
    return selected


def reproducible_cifar10_loaders(train_count, test_count, batch_size, seed, output_dir):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train = datasets.CIFAR10(root=str(ROOT / "data"), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=True, transform=transform)
    labels = list(range(10))
    train_idx = stratified_indices(train, labels, train_count, seed)
    test_idx = stratified_indices(test, labels, test_count, seed + 10_000)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"indices_seed{seed}.json", "w") as f:
        json.dump({"train": train_idx, "test": test_idx}, f, indent=2)
    return (
        DataLoader(Subset(train, train_idx), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(test, test_idx), batch_size=batch_size, shuffle=False),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the official HW-QCNN CIFAR-10 baseline.")
    parser.add_argument("--I", type=int, default=16)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--kernel-layout", default="all_connection")
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2 * 0.66)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--output-scale", type=float, default=50.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--faithful-official-loader", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "official_cifar10")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = OfficialHWQCNN3D(
        I=args.I,
        J=args.J,
        K=args.K,
        k=args.k,
        kernel_layout=args.kernel_layout,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    criterion = torch.nn.CrossEntropyLoss()

    if args.faithful_official_loader:
        train_loader, test_loader = load_cifar10(
            list(range(10)),
            args.train_count,
            args.test_count,
            args.batch_size,
        )
    else:
        train_loader, test_loader = reproducible_cifar10_loaders(
            args.train_count,
            args.test_count,
            args.batch_size,
            args.seed,
            args.output_dir,
        )

    state, train_loss, train_acc, test_loss, test_acc = train_globally(
        args.batch_size,
        args.I,
        args.J,
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        args.output_scale,
        args.epochs,
        args.test_interval,
        args.stride,
        device,
    )

    torch.save(state, args.output_dir / f"official_seed{args.seed}.pt")
    np.save(
        args.output_dir / f"official_seed{args.seed}.npy",
        {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "args": vars(args),
        },
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()
