"""Default project templates for `ars init`."""

# ---------------------------------------------------------------------------
# CLAUDE.md — context for Claude Code when configuring/running experiments
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_MD = r'''# Autoresearch Studio project

This project uses **autoresearchstudio** (`ars`) — an autonomous ML experiment framework.
The `ars` CLI handles experiment execution, metric extraction, keep/discard decisions, and dashboard sync automatically.

## Project files

| File | Role | Can you modify it? |
|------|------|--------------------|
| `train.py` | Model, hyperparameters, training loop | **Yes** — this is the main file you optimize |
| `prepare.py` | Data loading, evaluation function, constants | **No** during experiments (read-only). Edit only when adapting to a new dataset **before** `ars setup` |
| `autoresearch.yaml` | Configuration: metric, timeout, file roles, API key | Edit when configuring, not during the experiment loop |
| `program.md` | Auto-generated instructions for the experiment loop | Do not edit manually — regenerate with `ars generate` |

## Two modes of operation

### 1. Configure mode (before experiments start)
If the user asks you to adapt the project to their dataset/task:
- Modify `prepare.py`: replace data loading, evaluation, constants
- Modify `train.py`: adapt the model architecture to the new data shape
- Update `autoresearch.yaml`: set metric name, pattern, direction, timeout, constraints
- Run `ars generate` to update `program.md`

### 2. Experiment mode (autonomous loop)
Once experiments begin, follow the instructions in `program.md`:
- Only modify files listed under `files.editable` in `autoresearch.yaml`
- Use `ars` commands for the loop: `ars run` → `ars log` → `ars judge`
- Never stop — run indefinitely until the user interrupts

## Key ars commands

```
ars setup --tag <tag>    # create branch + init tracking + download data
ars run -d "description" # run experiment with timeout
ars log -d "description" # extract metrics from output
ars judge                # keep if improved, discard (auto-revert) if not
ars status               # current state, best metric
ars results              # full results table
```

## Output format contract

`train.py` must print metrics in this exact format (matched by regex in `autoresearch.yaml`):
```
val_accuracy:     0.990625
val_loss:         0.083412
num_params:       421642
```
The metric names and format must match the `metric.pattern` fields in `autoresearch.yaml`.
'''

# ---------------------------------------------------------------------------
# prepare.py — data loading & evaluation (READ-ONLY during experiments)
# ---------------------------------------------------------------------------

DEFAULT_PREPARE_PY = r'''"""
Data preparation and evaluation harness.

THIS FILE IS READ-ONLY DURING EXPERIMENTS.
The AI agent cannot modify it once experiments begin.
Modify it BEFORE running `ars setup` to adapt to your dataset.

== What to customize ==
1. TIME_BUDGET        — how long training runs (seconds)
2. download_data()    — download / locate your dataset
3. load_data()        — return (train_X, train_y, test_X, test_y) tensors
4. evaluate_accuracy  — fixed evaluation metric (accuracy + loss)
5. make_dataloader    — simple infinite batch iterator

The default is a working MNIST example. Replace the data loading
and evaluation to use your own dataset.
"""

import os
import struct
import gzip
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────
# Adjust these for your task.

TIME_BUDGET = 20          # training time budget in seconds
EVAL_BATCH_SIZE = 128     # batch size used during evaluation
EVAL_BATCHES = 50         # number of batches for validation

DATA_DIR = Path("./data")

# ── Data download ──────────────────────────────────────────────────
# Replace this section with your own data download / preparation.

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_data():
    """Download dataset files. Called once by `ars setup`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"  {filename} already exists")
            continue
        url = f"{MNIST_URL}/{filename}"
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Downloaded {filename}")


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(num, 1, rows, cols).float() / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


# ── load_data() ────────────────────────────────────────────────────
# Must return: (train_images, train_labels, test_images, test_labels)
# All as torch tensors.

def load_data():
    """Load train and test data as tensors."""
    train_images = _read_images(DATA_DIR / MNIST_FILES["train_images"])
    train_labels = _read_labels(DATA_DIR / MNIST_FILES["train_labels"])
    test_images = _read_images(DATA_DIR / MNIST_FILES["test_images"])
    test_labels = _read_labels(DATA_DIR / MNIST_FILES["test_labels"])
    return train_images, train_labels, test_images, test_labels


# ── make_dataloader() ──────────────────────────────────────────────
# Simple infinite batch iterator. Usually no need to change this.

def make_dataloader(images, labels, batch_size, shuffle=True):
    """Infinite dataloader yielding (batch_x, batch_y) tuples."""
    n = len(images)
    while True:
        if shuffle:
            perm = torch.randperm(n)
            images_shuffled = images[perm]
            labels_shuffled = labels[perm]
        else:
            images_shuffled = images
            labels_shuffled = labels
        for i in range(0, n - batch_size + 1, batch_size):
            yield images_shuffled[i:i + batch_size], labels_shuffled[i:i + batch_size]


# ── evaluate_accuracy() ───────────────────────────────────────────
# Fixed evaluation — DO NOT MODIFY during experiments.
# This is the ground truth metric used by `ars judge`.

@torch.no_grad()
def evaluate_accuracy(model, images, labels, batch_size=EVAL_BATCH_SIZE, num_batches=EVAL_BATCHES):
    """
    Evaluate model accuracy on a dataset.
    Returns (accuracy, avg_loss).
    """
    model.eval()
    device = next(model.parameters()).device
    loader = make_dataloader(images, labels, batch_size, shuffle=False)

    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * len(y)
        total_samples += len(y)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    model.train()
    return accuracy, avg_loss


# ── Main (run once to download data) ──────────────────────────────

if __name__ == "__main__":
    print("Preparing data...")
    download_data()
    train_images, train_labels, test_images, test_labels = load_data()
    print(f"\nTrain: {train_images.shape} images, {train_labels.shape} labels")
    print(f"Test:  {test_images.shape} images, {test_labels.shape} labels")
    print("\nDone! Ready to train.")
'''


# ---------------------------------------------------------------------------
# train.py — model + training loop (EDITABLE by the AI agent)
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_PY = r'''"""
Training script — the AI agent modifies this file.

== What gets optimized ==
- Model architecture (Net class)
- Hyperparameters (LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, ...)
- Training loop (optimizer, scheduler, augmentation, ...)

== What must stay ==
- Import from prepare.py (load_data, make_dataloader, evaluate_accuracy, TIME_BUDGET)
- The output format at the end (val_accuracy, val_loss, num_params lines)
- Time-budgeted training (must respect TIME_BUDGET)

Usage: python train.py
"""

import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, load_data, make_dataloader, evaluate_accuracy

# ── Model (modify this) ───────────────────────────────────────────

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── Hyperparameters (modify these) ────────────────────────────────

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0

# ── Training loop (modify this) ───────────────────────────────────

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_images, train_labels, test_images, test_labels = load_data()
    train_images, train_labels = train_images.to(device), train_labels.to(device)
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    # Model
    model = Net().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_dataloader(train_images, train_labels, BATCH_SIZE)

    # Training loop (time-budgeted)
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0

    while True:
        t0 = time.time()

        x, y = next(train_loader)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0
        total_training_time += dt
        step += 1

        loss_val = loss.item()
        if math.isnan(loss_val) or loss_val > 100:
            print("FAIL")
            exit(1)

        ema = 0.9
        smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
        debiased = smooth_loss / (1 - ema ** step)

        if step % 50 == 0:
            remaining = max(0, TIME_BUDGET - total_training_time)
            print(f"step {step:05d} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s")

        if total_training_time >= TIME_BUDGET:
            break

    print()

    # ── Evaluation (do not change output format) ──────────────────
    accuracy, val_loss = evaluate_accuracy(model, test_images, test_labels)

    t_end = time.time()
    print("---")
    print(f"val_accuracy:     {accuracy:.6f}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params}")


if __name__ == "__main__":
    main()
'''
