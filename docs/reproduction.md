# Reproduction

## Installation

Use Python 3.12 or 3.13. `uv sync --python 3.12 --locked --extra dev` installs the locked
development environment. For a standard pip installation, select a supported Python first:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
spanish-sentiment demo
```

On Windows, `py -3.12 -m venv .venv` selects Python 3.12 explicitly. The core package needs
only NumPy. Optional extras are `dev`, `train` and `report`. The pip route does not enforce
the uv transitive dependency lock.

## Inputs

```text
assets/private/
  logits_seed42.npy
  logits_cardiff.npy
  logits_beto.npy
  logits_cascade_robertuito_gpu.npy
  pln-uja-practica-final/
    train.csv                          # ID,TEXT,TARGET
    test.csv                           # ID,TEXT; only ID is needed for replay
    submission_v22_02_cascade010.csv   # ID,TARGET; optional reference
```

Each logit matrix is `(1706, 4)` in P/N/NEU/NONE order. Matrices do not carry IDs, so shape
validation alone cannot prove row alignment. Use original artifacts with the recorded hashes.
The data and logits are excluded from Git. Obtain the corpus through the competition subject
to its terms; logits are outputs of the author's training, not files provided by Kaggle.

## Replay

```bash
uv run spanish-sentiment check --logits-dir assets/private --test-csv assets/private/pln-uja-practica-final/test.csv
uv run spanish-sentiment replay --logits-dir assets/private --test-csv assets/private/pln-uja-practica-final/test.csv --output artifacts/submission.csv --reference assets/private/pln-uja-practica-final/submission_v22_02_cascade010.csv
```

IDs remain strings, including leading zeros. Replay rejects missing or duplicate test IDs,
invalid matrix shapes and non-finite values. A reference must have exactly ID,TARGET columns
in the same order and match every label before output is created. CSV line endings remain
CRLF to reproduce the archived Windows artifact consistently across platforms.

## Training

Use `--dry-run` to validate both CSVs and inspect the frozen recipe. It needs no training
dependencies and creates no files. Repeated training IDs are accepted because they occur in
the original corpus; prediction IDs must be unique.

Install a hardware-compatible PyTorch build and the optional dependencies. The original report
used Python 3.12, PyTorch 2.11.0+cu126, Transformers 4.57.1 and a 4 GB RTX 3050 Laptop GPU.
The locked extra selects the ordinary PyPI PyTorch build; check its hardware support before
a long run. GPU-specific wheels may require a separate environment.

```bash
uv sync --locked --extra train
uv run spanish-sentiment train --component rob42 --train-csv assets/private/pln-uja-practica-final/train.csv --test-csv assets/private/pln-uja-practica-final/test.csv --output-dir artifacts/training-001
```

Repeat with `cardiff`, `beto` and `cascade`. The four components can share the same output root;
each reserves its own new subdirectory and writes its historical logit filename at the root.
Existing component directories or logits are rejected. Saved models, component predictions and
`training.json` stay in their component subdirectory. That record includes the recipe, seed,
class order, dependency versions and input/output hashes. A failed training run may leave partial artifacts: inspect
them and choose a new output directory rather than treating them as a completed result.

Full training downloads models on first use and can be slow on CPU. No duration or GPU
determinism claim is made. Backbone revisions were not pinned in the original experiment;
cached-logit replay is the precise reproduction path for the archived labels.

For a hardware-specific environment, uv can select a compatible PyTorch backend:

```bash
uv pip install -e ".[train]" --torch-backend=auto
uv run --no-sync spanish-sentiment train --component rob42 --train-csv assets/private/pln-uja-practica-final/train.csv --test-csv assets/private/pln-uja-practica-final/test.csv --output-dir artifacts/training-001
```

This alternative does not enforce the PyPI-only lock for the selected PyTorch wheel.
`--no-sync` keeps that hardware-specific installation; the completed run records its versions.
See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for backend selection.
