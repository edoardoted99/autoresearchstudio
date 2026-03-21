"""
MNIST classification — the file the agent modifies.
Single-file, simple CNN baseline.

Usage: python train.py
"""

import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, load_data, make_dataloader, evaluate_accuracy

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------------
# Hyperparameters (edit these)
# ---------------------------------------------------------------------------

LEARNING_RATE = 3e-3
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)
    train_loader = make_dataloader(train_images, train_labels, BATCH_SIZE)

    # Training loop (time-budgeted)
    t_start_training = time.time()
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
        scheduler.step()

        dt = time.time() - t0
        total_training_time += dt
        step += 1

        # Logging
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

    # Evaluation
    accuracy, val_loss = evaluate_accuracy(model, test_images, test_labels)

    # Summary
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
