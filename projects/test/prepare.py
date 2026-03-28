"""
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
