"""Experiment execution: subprocess with timeout and output capture."""

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunResult:
    exit_code: Optional[int]
    duration_seconds: float
    timed_out: bool
    log_file: str


def run_experiment(command: str, log_file: str = "run.log", timeout: int = 600) -> RunResult:
    """Execute command, capture stdout+stderr to log_file, enforce timeout.

    Returns RunResult with exit code, duration, timeout flag, and log path.
    """
    t0 = time.time()

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Kill the entire process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Grace period
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    duration = time.time() - t0

    return RunResult(
        exit_code=proc.returncode,
        duration_seconds=duration,
        timed_out=timed_out,
        log_file=log_file,
    )


def read_log_tail(log_file: str, lines: int = 50) -> str:
    """Read last N lines of a log file."""
    try:
        with open(log_file) as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except FileNotFoundError:
        return ""
