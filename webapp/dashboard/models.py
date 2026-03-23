import secrets
from django.db import models


def generate_api_key():
    return f"ars_{secrets.token_hex(24)}"


class ApiKey(models.Model):
    key = models.CharField(max_length=64, unique=True, default=generate_api_key)
    name = models.CharField(max_length=255, help_text="Project name for this key")
    config = models.JSONField(default=dict, blank=True, help_text="Project configuration from autoresearch.yaml")
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.key[:12]}...)"


class Experiment(models.Model):
    STATUS_CHOICES = [
        ("running", "Running"),
        ("keep", "Keep"),
        ("discard", "Discard"),
        ("crash", "Crash"),
    ]

    api_key = models.ForeignKey(ApiKey, on_delete=models.CASCADE, related_name="experiments")
    run_tag = models.CharField(max_length=100)
    experiment_number = models.IntegerField()
    commit_hash = models.CharField(max_length=40, blank=True, default="")
    parent_commit_hash = models.CharField(max_length=40, blank=True, default="")
    metric_name = models.CharField(max_length=100, blank=True, default="")
    metric_value = models.FloatField(null=True, blank=True)
    secondary_metrics = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="running")
    description = models.TextField(blank=True, default="")
    diff = models.TextField(blank=True, default="")
    timeout_seconds = models.FloatField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"#{self.experiment_number} {self.status} ({self.api_key.name}/{self.run_tag})"
