"""
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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, load_data, make_dataloader, evaluate_accuracy

# ── Model (modify this) ───────────────────────────────────────────

def augment_batch(x):
    """Random shift augmentation for MNIST."""
    shift = 3
    n, c, h, w = x.shape
    dx = random.randint(-shift, shift)
    dy = random.randint(-shift, shift)
    x_aug = torch.zeros_like(x)
    src_x1 = max(0, dx)
    src_x2 = min(w, w + dx)
    src_y1 = max(0, dy)
    src_y2 = min(h, h + dy)
    dst_x1 = max(0, -dx)
    dst_y1 = max(0, -dy)
    x_aug[:, :, dst_y1:dst_y1+(src_y2-src_y1), dst_x1:dst_x1+(src_x2-src_x1)] = x[:, :, src_y1:src_y2, src_x1:src_x2]
    return x_aug


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28->14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14->7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 7->3
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── Hyperparameters (modify these) ────────────────────────────────

LEARNING_RATE = 2e-3
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4

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
        x = augment_batch(x)
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
