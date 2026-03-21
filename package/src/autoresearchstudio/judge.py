"""Keep/discard logic: compare metrics, apply git operations."""

import subprocess
from dataclasses import dataclass
from typing import Optional

from .config import Config
from .tracker import Tracker, Experiment


@dataclass
class JudgeResult:
    decision: str  # "keep", "discard", "crash"
    current_value: Optional[float]
    best_value: Optional[float]
    delta: Optional[float]
    reason: str
    is_baseline: bool = False


class Judge:
    def __init__(self, config: Config, tracker: Tracker):
        self.config = config
        self.tracker = tracker

    def evaluate(self, experiment: Experiment) -> JudgeResult:
        """Compare experiment against current best and return a decision."""
        # Crash: no metric value
        if experiment.metric_value is None:
            return JudgeResult(
                decision="crash",
                current_value=None,
                best_value=None,
                delta=None,
                reason="No metric value (crash or timeout)",
            )

        best = self.tracker.get_best()

        # First successful run: auto-keep as baseline
        if best is None:
            return JudgeResult(
                decision="keep",
                current_value=experiment.metric_value,
                best_value=None,
                delta=None,
                reason=f"Baseline: {self.config.metric.name} = {experiment.metric_value:.6f}",
                is_baseline=True,
            )

        current = experiment.metric_value
        best_val = best.metric_value
        delta = current - best_val
        threshold = self.config.judge.threshold

        if self.config.metric.direction == "minimize":
            improved = delta < -threshold
            delta_display = delta
        else:
            improved = delta > threshold
            delta_display = delta

        if improved:
            return JudgeResult(
                decision="keep",
                current_value=current,
                best_value=best_val,
                delta=delta_display,
                reason=(
                    f"{self.config.metric.name} improved "
                    f"{best_val:.6f} -> {current:.6f} (delta: {delta_display:+.6f})"
                ),
            )
        else:
            return JudgeResult(
                decision="discard",
                current_value=current,
                best_value=best_val,
                delta=delta_display,
                reason=(
                    f"{self.config.metric.name} did not improve "
                    f"{best_val:.6f} -> {current:.6f} (delta: {delta_display:+.6f})"
                ),
            )

    def apply(self, experiment: Experiment, result: JudgeResult):
        """Apply the decision: update DB status, git reset if discard/crash."""
        self.tracker.update_experiment(experiment.id, status=result.decision)

        if result.decision in ("discard", "crash"):
            # Revert the last commit
            try:
                subprocess.run(
                    ["git", "reset", "--hard", "HEAD~1"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass  # If reset fails, don't crash the framework

        self.tracker.sync(experiment.id)
