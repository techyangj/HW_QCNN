# Handoff: CPSR-HW-QCNN Fork and Colab Workflow

Date verified: June 25, 2026 EDT / June 26, 2026 UTC

## Current Status

The CPSR-HW-QCNN work is pushed to a fork because the current GitHub user does
not have write permission to the upstream repository.

- Upstream repository: https://github.com/ptitbroussou/HW_QCNN
- Fork repository: https://github.com/techyangj/HW_QCNN
- Working branch: `feature/cpsr-hw-qcnn`
- Verified code commit before this handoff note: `d06117c`
- Local branch tracking: `fork/feature/cpsr-hw-qcnn`
- Fork PR URL suggested by GitHub:
  https://github.com/techyangj/HW_QCNN/pull/new/feature/cpsr-hw-qcnn
- Upstream PR compare URL:
  https://github.com/ptitbroussou/HW_QCNN/compare/main...techyangj:HW_QCNN:feature/cpsr-hw-qcnn

Remote setup in the local checkout:

```bash
origin  https://github.com/ptitbroussou/HW_QCNN.git
fork    https://github.com/techyangj/HW_QCNN.git
```

Important: pushing directly to `origin` fails with HTTP 403 because
`techyangj` does not have write access to `ptitbroussou/HW_QCNN`.

## What Was Added

The branch contains a reusable CPSR training workflow:

- `models/CPSR_HW_QCNN.py`
  - Defines `OfficialHWQCNN3D` and `CPSRHWQCNN`.
- `scripts/reproduce_official.py`
  - Runs the pinned official CIFAR-10 baseline protocol.
- `scripts/train_cpsr.py`
  - Runs the CPSR model with phase coupling, shifted windows, data reuploading,
    ablations, and `full_space` readout support.
- `scripts/colab_bootstrap.py`
  - Colab CLI runner that clones the fork branch into Colab, installs
    dependencies, runs training, and archives `runs/`.
- `scripts/colab_upload_runner.py`
  - Fallback Colab runner for cases where GitHub is unavailable. It expects an
    uploaded tarball at `/content/HW_QCNN_upload.tar.gz`.
- `requirements-colab.txt`
  - Extra Colab dependencies beyond the default Colab PyTorch stack.
- `configs/cifar10_official.yaml` and `configs/cifar10_cpsr.yaml`
  - Recorded experimental configuration defaults.
- `tests/`
  - Physicality, complex backend, shifted permutation, full-space POVM readout,
    and small CPSR forward/backward tests.

The `.gitignore` excludes local-only artifacts such as `.venv/`, `data/`,
`runs/`, caches, `.npy`, and `.pt` files.

## Validation Already Done

Local tests:

```bash
./.venv/bin/python -m pytest -q
```

Result:

```text
6 passed
```

Local micro smoke run:

```bash
./.venv/bin/python scripts/train_cpsr.py \
  --I 4 --J 2 --K 2 --k 3 \
  --phase-rank 1 \
  --blocks-per-stage 1 \
  --readout full_space \
  --epochs 1 \
  --train-count 10 \
  --test-count 10 \
  --batch-size 2 \
  --test-interval 1 \
  --device cpu \
  --output-dir runs/local_colab_micro_smoke
```

Colab fork-clone smoke run was also completed successfully on a T4 runtime.
The session was stopped afterward. There were no active Colab sessions at the
end of the handoff.

Colab fork smoke output:

```text
train_loss: [17.254804134368896]
train_accuracy: [10.0]
test_loss: [17.188949489593504]
test_accuracy: [10.0]
train_indices: 10
test_indices: 10
```

This was only a tiny pipeline check with `I=4,J=2`. It is not a scientific
result.

Downloaded local artifacts from the fork-clone smoke run:

- `runs/colab_fork_smoke_d06117c/hw_qcnn_runs.tar.gz`
- `runs/colab_fork_smoke_d06117c/cpsr-fork-smoke-log.md`
- `runs/colab_fork_smoke_d06117c/runs/colab_fork_smoke/cpsr_seed0.npy`
- `runs/colab_fork_smoke_d06117c/runs/colab_fork_smoke/cpsr_seed0.pt`
- `runs/colab_fork_smoke_d06117c/runs/colab_fork_smoke/indices_seed0.json`

There is also an earlier upload-mode smoke run at:

- `runs/colab_smoke_6c9da3f/`

## Colab CLI Notes

The Colab CLI was installed locally with:

```bash
uv tool install google-colab-cli
```

The binary is:

```bash
/Users/a1/.local/bin/colab
```

On this machine, `/Users/a1/.config` is owned by `root`, so Colab CLI commands
were run with a temporary HOME:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab sessions
```

If another person runs this on their own machine, they should authenticate with
their own Google account. On first use, the CLI prints a Google OAuth URL. Open
that URL in a browser, approve access, and paste the authorization code back
into the terminal. Do not share OAuth tokens or authorization codes.

Check for active sessions:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab sessions
```

At handoff time, this returned:

```text
[colab] No active sessions found on server.
```

## Re-run the Verified Colab Smoke Test

From the local repository root:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab run \
  --keep \
  -s cpsr-fork-smoke \
  --gpu T4 \
  --timeout 1800 \
  scripts/colab_bootstrap.py \
  --I 4 --J 2 --K 2 --k 3 \
  --phase-rank 1 \
  --blocks-per-stage 1 \
  --readout full_space \
  --epochs 1 \
  --train-count 10 \
  --test-count 10 \
  --batch-size 2 \
  --test-interval 1 \
  --device auto \
  --output-dir runs/colab_fork_smoke
```

Download outputs:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab download \
  -s cpsr-fork-smoke \
  /content/hw_qcnn_runs.tar.gz \
  ./runs/cpsr-fork-smoke.tar.gz

HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab log \
  -s cpsr-fork-smoke \
  -o ./runs/cpsr-fork-smoke-log.md
```

Stop the session:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab stop -s cpsr-fork-smoke
```

## Run a Full CPSR Training Job

Use this when ready to spend Colab compute. This uses the default CPSR protocol
from `scripts/train_cpsr.py` unless overridden.

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab run \
  --keep \
  -s cpsr-full-seed0 \
  --gpu T4 \
  --timeout 86400 \
  scripts/colab_bootstrap.py \
  --epochs 40 \
  --seed 0 \
  --device auto \
  --output-dir runs/cpsr_cifar10_seed0
```

After completion:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab download \
  -s cpsr-full-seed0 \
  /content/hw_qcnn_runs.tar.gz \
  ./runs/cpsr-full-seed0.tar.gz

HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab log \
  -s cpsr-full-seed0 \
  -o ./runs/cpsr-full-seed0-log.md

HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab stop -s cpsr-full-seed0
```

Always stop the session after downloading artifacts to avoid wasting compute.

## Run the Official Baseline on Colab

The bootstrap defaults to `scripts/train_cpsr.py`, but it can run another script:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab run \
  --keep \
  -s official-full-seed0 \
  --gpu T4 \
  --timeout 86400 \
  scripts/colab_bootstrap.py \
  --train-script scripts/reproduce_official.py \
  --epochs 40 \
  --seed 0 \
  --device auto \
  --output-dir runs/official_cifar10_seed0
```

Then download `/content/hw_qcnn_runs.tar.gz`, export the log, and stop the
session as shown above.

## Fallback Upload Mode

If GitHub is not reachable or the branch is not available, create a tarball from
the current commit:

```bash
git archive --format=tar.gz -o /tmp/HW_QCNN-current.tar.gz HEAD
```

Create a Colab session and upload:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab new -s cpsr-upload --gpu T4
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab upload \
  -s cpsr-upload \
  /tmp/HW_QCNN-current.tar.gz \
  /content/HW_QCNN_upload.tar.gz
```

Run the upload runner:

```bash
HOME=/tmp/codex-colab-home /Users/a1/.local/bin/colab exec \
  -s cpsr-upload \
  -f scripts/colab_upload_runner.py \
  --timeout 1800
```

Then download `/content/hw_qcnn_runs.tar.gz`, export the log, and stop the
session.

## Known Issues and Caveats

- The fork branch is ready, but no pull request has been opened yet.
- The upstream repository is not writable by `techyangj`.
- Colab resource availability is dynamic. T4 was available during the smoke
  tests, but future availability may differ.
- The smoke run is intentionally tiny and only validates the pipeline.
- Full default training may take substantially longer and will consume Colab
  compute units.
- Local result directories under `runs/` are ignored by git and are not pushed.

## Suggested Next Steps

1. Open a PR from `techyangj/HW_QCNN:feature/cpsr-hw-qcnn`.
2. Run full CPSR seed 0 on Colab and download `runs/cpsr_cifar10_seed0`.
3. Run the official baseline seed 0 for comparison.
4. Repeat for the planned seed set if the first full run is healthy.
5. Keep the downloaded logs and `indices_seed*.json` files with the results for
   reproducibility.
