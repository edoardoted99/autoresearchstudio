"""
One-time data preparation for MNIST autoresearch experiments.
Downloads MNIST and creates train/val splits.

Usage:
    python prepare.py

Data is stored in ./data/
"""

import os
import struct
import gzip
import urllib.request
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 120       # training time budget in seconds (2 minutes)
EVAL_BATCHES = 50       # number of batches for validation eval
BATCH_SIZE = 128        # fixed batch size for evaluation

DATA_DIR = Path("./data")
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_mnist():
    """Download MNIST dataset files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, filename in FILES.items():
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


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_data():
    """Load MNIST train and test data as tensors."""
    train_images = _read_images(DATA_DIR / FILES["train_images"])
    train_labels = _read_labels(DATA_DIR / FILES["train_labels"])
    test_images = _read_images(DATA_DIR / FILES["test_images"])
    test_labels = _read_labels(DATA_DIR / FILES["test_labels"])
    return train_images, train_labels, test_images, test_labels


def make_dataloader(images, labels, batch_size, shuffle=True):
    """Simple infinite dataloader."""
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


@torch.no_grad()
def evaluate_accuracy(model, images, labels, batch_size=BATCH_SIZE, num_batches=EVAL_BATCHES):
    """
    Evaluate model accuracy on a dataset.
    Returns (accuracy, avg_loss).
    This is the fixed evaluation metric — DO NOT MODIFY.
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
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * len(y)
        total_samples += len(y)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    model.train()
    return accuracy, avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Preparing MNIST data...")
    download_mnist()

    # Verify
    train_images, train_labels, test_images, test_labels = load_data()
    print(f"\nTrain: {train_images.shape} images, {train_labels.shape} labels")
    print(f"Test:  {test_images.shape} images, {test_labels.shape} labels")
    print("\nDone! Ready to train.")
