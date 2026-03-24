"""Create a demo API key with sample MNIST experiments."""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from dashboard.models import ApiKey, Experiment

DEMO_KEY = "ars_demo_mnist_example_000000000000"

DEMO_CONFIG = {
    "project": {
        "name": "mnist_example",
        "description": "MNIST digit classification demo",
        "goal": "Get the highest val_accuracy.",
    },
    "files": {
        "editable": ["train.py"],
        "readonly": ["prepare.py"],
    },
    "experiment": {
        "run_command": "python train.py",
        "timeout": 60,
    },
    "metric": {
        "name": "val_accuracy",
        "pattern": r"^val_accuracy:\s+([\d.]+)",
        "direction": "maximize",
    },
}

EXPERIMENTS = [
    {
        "experiment_number": 1,
        "status": "keep",
        "metric_value": 0.9506,
        "metric_name": "val_accuracy",
        "description": "baseline CNN",
        "commit_hash": "a1b2c3d",
        "parent_commit_hash": "",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -1,0 +1,30 @@\n+# Baseline CNN for MNIST\n+class Net(nn.Module):\n+    def __init__(self):\n+        self.conv1 = nn.Conv2d(1, 32, 3)\n+        self.conv2 = nn.Conv2d(32, 64, 3)\n+        self.fc1 = nn.Linear(64*7*7, 128)\n+        self.fc2 = nn.Linear(128, 10)\n",
        "duration_seconds": 18.3,
    },
    {
        "experiment_number": 2,
        "status": "keep",
        "metric_value": 0.9720,
        "metric_name": "val_accuracy",
        "description": "add dropout + cosine lr schedule",
        "commit_hash": "e4f5g6h",
        "parent_commit_hash": "a1b2c3d",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -5,6 +5,8 @@\n         self.conv1 = nn.Conv2d(1, 32, 3)\n         self.conv2 = nn.Conv2d(32, 64, 3)\n+        self.dropout = nn.Dropout(0.25)\n         self.fc1 = nn.Linear(64*7*7, 128)\n+        self.fc_drop = nn.Dropout(0.5)\n         self.fc2 = nn.Linear(128, 10)\n@@ -20,1 +22,2 @@\n-optimizer = optim.Adam(model.parameters(), lr=1e-3)\n+optimizer = optim.Adam(model.parameters(), lr=1e-3)\n+scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)\n",
        "duration_seconds": 19.1,
    },
    {
        "experiment_number": 3,
        "status": "discard",
        "metric_value": 0.9680,
        "metric_name": "val_accuracy",
        "description": "replace relu with gelu",
        "commit_hash": "i7j8k9l",
        "parent_commit_hash": "e4f5g6h",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -12,3 +12,3 @@\n-        x = F.relu(self.conv1(x))\n-        x = F.relu(self.conv2(x))\n+        x = F.gelu(self.conv1(x))\n+        x = F.gelu(self.conv2(x))\n",
        "duration_seconds": 20.5,
    },
    {
        "experiment_number": 4,
        "status": "keep",
        "metric_value": 0.9785,
        "metric_name": "val_accuracy",
        "description": "batch norm after each conv layer",
        "commit_hash": "m0n1o2p",
        "parent_commit_hash": "e4f5g6h",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -4,6 +4,8 @@\n         self.conv1 = nn.Conv2d(1, 32, 3)\n+        self.bn1 = nn.BatchNorm2d(32)\n         self.conv2 = nn.Conv2d(32, 64, 3)\n+        self.bn2 = nn.BatchNorm2d(64)\n",
        "duration_seconds": 21.7,
    },
    {
        "experiment_number": 5,
        "status": "discard",
        "metric_value": 0.9601,
        "metric_name": "val_accuracy",
        "description": "aggressive data augmentation (rotation + scale)",
        "commit_hash": "q3r4s5t",
        "parent_commit_hash": "m0n1o2p",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -30,1 +30,5 @@\n-transform = transforms.ToTensor()\n+transform = transforms.Compose([\n+    transforms.RandomRotation(15),\n+    transforms.RandomAffine(0, scale=(0.8, 1.2)),\n+    transforms.ToTensor(),\n+])\n",
        "duration_seconds": 24.2,
    },
    {
        "experiment_number": 6,
        "status": "keep",
        "metric_value": 0.9812,
        "metric_name": "val_accuracy",
        "description": "increase conv channels 32->64, 64->128",
        "commit_hash": "u6v7w8x",
        "parent_commit_hash": "m0n1o2p",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -4,4 +4,4 @@\n-        self.conv1 = nn.Conv2d(1, 32, 3)\n-        self.bn1 = nn.BatchNorm2d(32)\n-        self.conv2 = nn.Conv2d(32, 64, 3)\n-        self.bn2 = nn.BatchNorm2d(64)\n+        self.conv1 = nn.Conv2d(1, 64, 3)\n+        self.bn1 = nn.BatchNorm2d(64)\n+        self.conv2 = nn.Conv2d(64, 128, 3)\n+        self.bn2 = nn.BatchNorm2d(128)\n",
        "duration_seconds": 26.8,
    },
    {
        "experiment_number": 7,
        "status": "discard",
        "metric_value": 0.9754,
        "metric_name": "val_accuracy",
        "description": "reduce learning rate to 5e-4",
        "commit_hash": "y9z0a1b",
        "parent_commit_hash": "u6v7w8x",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -22,1 +22,1 @@\n-optimizer = optim.Adam(model.parameters(), lr=1e-3)\n+optimizer = optim.Adam(model.parameters(), lr=5e-4)\n",
        "duration_seconds": 27.1,
    },
    {
        "experiment_number": 8,
        "status": "keep",
        "metric_value": 0.9841,
        "metric_name": "val_accuracy",
        "description": "add third conv layer + global avg pool",
        "commit_hash": "c2d3e4f",
        "parent_commit_hash": "u6v7w8x",
        "diff": "--- a/train.py\n+++ b/train.py\n@@ -6,6 +6,8 @@\n         self.conv2 = nn.Conv2d(64, 128, 3)\n         self.bn2 = nn.BatchNorm2d(128)\n+        self.conv3 = nn.Conv2d(128, 256, 3)\n+        self.bn3 = nn.BatchNorm2d(256)\n+        self.gap = nn.AdaptiveAvgPool2d(1)\n-        self.fc1 = nn.Linear(128*5*5, 128)\n+        self.fc1 = nn.Linear(256, 128)\n",
        "duration_seconds": 30.4,
    },
]


class Command(BaseCommand):
    help = "Create demo API key and sample MNIST experiments"

    def handle(self, *args, **options):
        api_key, created = ApiKey.objects.get_or_create(
            key=DEMO_KEY,
            defaults={"name": "mnist_example", "config": DEMO_CONFIG},
        )
        if not created:
            api_key.name = "mnist_example"
            api_key.config = DEMO_CONFIG
            api_key.save()
            api_key.experiments.all().delete()
            self.stdout.write("Reset existing demo key.")

        base_time = timezone.now() - timedelta(hours=2)

        for i, exp_data in enumerate(EXPERIMENTS):
            started = base_time + timedelta(minutes=i * 5)
            finished = started + timedelta(seconds=exp_data["duration_seconds"])
            Experiment.objects.create(
                api_key=api_key,
                run_tag="demo",
                experiment_number=exp_data["experiment_number"],
                commit_hash=exp_data["commit_hash"],
                parent_commit_hash=exp_data["parent_commit_hash"],
                metric_name=exp_data["metric_name"],
                metric_value=exp_data["metric_value"],
                status=exp_data["status"],
                description=exp_data["description"],
                diff=exp_data["diff"],
                timeout_seconds=60,
                duration_seconds=exp_data["duration_seconds"],
                started_at=started,
                finished_at=finished,
            )

        self.stdout.write(self.style.SUCCESS(
            f"Demo ready! Key: {DEMO_KEY}\n"
            f"Dashboard: /dashboard/?key={DEMO_KEY}"
        ))
