import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ARCHIVE = Path(os.environ.get("HW_QCNN_UPLOAD_ARCHIVE", "/content/HW_QCNN_upload.tar.gz"))
WORKDIR = Path(os.environ.get("HW_QCNN_WORKDIR", "/content/HW_QCNN"))
RUNS_ARCHIVE = Path(os.environ.get("HW_QCNN_RUNS_ARCHIVE", "/content/hw_qcnn_runs.tar.gz"))

DEFAULT_TRAIN_ARGS = [
    "--I",
    "4",
    "--J",
    "2",
    "--K",
    "2",
    "--k",
    "3",
    "--phase-rank",
    "1",
    "--blocks-per-stage",
    "1",
    "--readout",
    "full_space",
    "--epochs",
    "1",
    "--train-count",
    "10",
    "--test-count",
    "10",
    "--batch-size",
    "2",
    "--test-interval",
    "1",
    "--device",
    "auto",
    "--output-dir",
    "runs/colab_upload_smoke",
]


def run(command, cwd=None):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def unpack_repo():
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Expected uploaded archive at {ARCHIVE}")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(WORKDIR)


def install_dependencies():
    requirements = WORKDIR / "requirements-colab.txt"
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
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


def train():
    extra = os.environ.get("HW_QCNN_TRAIN_ARGS")
    train_args = shlex.split(extra) if extra else DEFAULT_TRAIN_ARGS
    run([sys.executable, "scripts/train_cpsr.py", *train_args], cwd=WORKDIR)


def archive_runs():
    runs_dir = WORKDIR / "runs"
    if runs_dir.exists():
        with tarfile.open(RUNS_ARCHIVE, "w:gz") as archive:
            archive.add(runs_dir, arcname="runs")
        print(f"Archived runs to {RUNS_ARCHIVE}", flush=True)


def main():
    unpack_repo()
    install_dependencies()
    print_runtime_info()
    train()
    archive_runs()


if __name__ == "__main__":
    main()
