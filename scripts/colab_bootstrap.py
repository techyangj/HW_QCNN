import argparse
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/ptitbroussou/HW_QCNN.git"
DEFAULT_BRANCH = "feature/cpsr-hw-qcnn"


def run(command, cwd=None):
    print("+ " + " ".join(shlex.quote(str(part)) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone HW_QCNN into a Colab runtime, install dependencies, and run training."
    )
    parser.add_argument("--repo-url", default=os.environ.get("HW_QCNN_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--branch", default=os.environ.get("HW_QCNN_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--workdir", type=Path, default=Path("/content/HW_QCNN"))
    parser.add_argument("--requirements", default="requirements-colab.txt")
    parser.add_argument("--train-script", default="scripts/train_cpsr.py")
    parser.add_argument("--archive", type=Path, default=Path("/content/hw_qcnn_runs.tar.gz"))
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def sync_repo(args):
    if args.workdir.exists():
        if (args.workdir / ".git").exists():
            run(["git", "fetch", "origin", args.branch], cwd=args.workdir)
            run(["git", "checkout", args.branch], cwd=args.workdir)
            run(["git", "reset", "--hard", f"origin/{args.branch}"], cwd=args.workdir)
            return
        raise RuntimeError(f"{args.workdir} exists but is not a git checkout")

    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            args.branch,
            args.repo_url,
            str(args.workdir),
        ]
    )


def install_dependencies(args):
    if args.skip_install:
        return
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    requirements = args.workdir / args.requirements
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])


def print_runtime_info():
    code = (
        "import torch, sys; "
        "print('python', sys.version.split()[0]); "
        "print('torch', torch.__version__); "
        "print('cuda_available', torch.cuda.is_available()); "
        "print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
    )
    run([sys.executable, "-c", code])


def archive_runs(args):
    runs_dir = args.workdir / "runs"
    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}", flush=True)
        return
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(runs_dir, arcname="runs")
    print(f"Archived runs to {args.archive}", flush=True)


def main():
    args = parse_args()
    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    sync_repo(args)
    install_dependencies(args)
    print_runtime_info()
    run([sys.executable, args.train_script, *train_args], cwd=args.workdir)
    archive_runs(args)


if __name__ == "__main__":
    main()

