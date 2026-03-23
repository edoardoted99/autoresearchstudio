"""
One-time data preparation for CIFAR-10 autoresearch experiments.
Downloads CIFAR-10 and creates train/val splits.

Usage:
    python prepare.py

Data is stored in ./data/
"""

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 30        # training time budget in seconds
EVAL_BATCHES = 50       # number of batches for validation eval
BATCH_SIZE = 128        # fixed batch size for evaluation

DATA_DIR = Path("./data")
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_FILE = "cifar-10-python.tar.gz"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_cifar10():
    """Download and extract CIFAR-10 dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = DATA_DIR / CIFAR_FILE
    extracted_dir = DATA_DIR / "cifar-10-batches-py"

    if extracted_dir.exists():
        print("  CIFAR-10 already extracted")
        return

    if not archive_path.exists():
        print("  Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR_URL, archive_path)
        print("  Downloaded")

    print("  Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("  Extracted")


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = torch.tensor(d[b"data"], dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
    labels = torch.tensor(d[b"labels"], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_data():
    """Load CIFAR-10 train and test data as tensors."""
    batch_dir = DATA_DIR / "cifar-10-batches-py"

    train_images, train_labels = [], []
    for i in range(1, 6):
        imgs, lbls = _load_batch(batch_dir / f"data_batch_{i}")
        train_images.append(imgs)
        train_labels.append(lbls)

    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    test_images, test_labels = _load_batch(batch_dir / "test_batch")

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
    print("Preparing CIFAR-10 data...")
    download_cifar10()

    train_images, train_labels, test_images, test_labels = load_data()
    print(f"\nTrain: {train_images.shape} images, {train_labels.shape} labels")
    print(f"Test:  {test_images.shape} images, {test_labels.shape} labels")
    print("\nDone! Ready to train.")
