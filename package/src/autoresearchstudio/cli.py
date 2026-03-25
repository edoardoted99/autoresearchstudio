"""CLI entry point: all `ars` subcommands."""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from . import __version__
from .config import (
    Config, load_config, validate_config, config_to_yaml,
    CONFIG_FILENAME, ProjectConfig, FilesConfig, ExperimentConfig,
    MetricConfig, JudgeConfig, ApiConfig, SecondaryMetric,
)
from .judge import Judge
from .prompt import generate_program_md
from .runner import run_experiment, read_log_tail
from .templates import DEFAULT_TRAIN_PY, DEFAULT_PREPARE_PY, DEFAULT_CLAUDE_MD
from .tracker import Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(*args, check=True, capture=True) -> str:
    """Run a git command. Returns stdout as string."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=capture,
        text=True,
        check=check,
    )
    return result.stdout.strip() if capture else ""


def _git_commit_hash() -> str:
    try:
        return _git("rev-parse", "--short=7", "HEAD")
    except subprocess.CalledProcessError:
        return ""


def _git_branch() -> str:
    try:
        return _git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return ""


def _git_diff_from_parent() -> str:
    try:
        return _git("diff", "HEAD~1", "HEAD")
    except subprocess.CalledProcessError:
        return ""


def _extract_metric(log_file: str, pattern: str) -> float | None:
    """Extract a metric value from a log file using a regex pattern."""
    try:
        with open(log_file) as f:
            for line in f:
                m = re.search(pattern, line)
                if m:
                    return float(m.group(1))
    except (FileNotFoundError, ValueError):
        pass
    return None


def _load_config_or_exit() -> Config:
    if not os.path.exists(CONFIG_FILENAME):
        print(f"Error: {CONFIG_FILENAME} not found. Run `ars init` first.")
        sys.exit(1)
    return load_config()


def _load_tracker(config: Config) -> Tracker:
    return Tracker(config)


def _get_run_tag(tracker: Tracker) -> str:
    tag = tracker.local.get_meta("run_tag")
    if not tag:
        print("Error: no active run. Run `ars setup --tag <tag>` first.")
        sys.exit(1)
    return tag


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _generate_api_key(config: Config) -> str | None:
    """Try to generate an API key from the remote server. Returns key or None."""
    import requests as req

    endpoint = config.api.endpoint.rstrip("/")
    base_url = endpoint.rsplit("/api", 1)[0]
    url = f"{base_url}/api/keys/"
    name = config.project.name or "cli"

    try:
        resp = req.post(url, json={"name": name}, timeout=10)
        resp.raise_for_status()
        return resp.json()["key"]
    except Exception:
        return None


def _write_file_if_needed(path: str, content: str, force: bool) -> bool:
    """Write file if it doesn't exist or force is set. Returns True if written."""
    if os.path.exists(path) and not force:
        print(f"  {path} already exists (skipped, use --force to overwrite)")
        return False
    with open(path, "w") as f:
        f.write(content)
    print(f"  {path}")
    return True


def cmd_init(args):
    """Initialize a new autoresearch project with working defaults."""
    if os.path.exists(CONFIG_FILENAME) and not args.force:
        print(f"{CONFIG_FILENAME} already exists. Use --force to overwrite.")
        sys.exit(1)

    if args.from_template == "karpathy":
        config = Config(
            project=ProjectConfig(
                name="autoresearch",
                description="Autonomous pretraining research — generalized from Karpathy's autoresearch.",
                goal="Get the lowest val_bpb.",
            ),
            files=FilesConfig(
                editable=["train.py"],
                readonly=["prepare.py"],
                context=["README.md"],
            ),
            experiment=ExperimentConfig(
                run_command="uv run train.py",
                timeout=600,
                setup_command="uv run prepare.py",
                constraints=(
                    "VRAM is a soft constraint. Some increase is acceptable for "
                    "meaningful val_bpb gains, but it should not blow up dramatically."
                ),
            ),
            metric=MetricConfig(
                name="val_bpb",
                pattern=r"^val_bpb:\s+([\d.]+)",
                direction="minimize",
                secondary=[
                    SecondaryMetric(name="peak_vram_mb", pattern=r"^peak_vram_mb:\s+([\d.]+)"),
                    SecondaryMetric(name="mfu_percent", pattern=r"^mfu_percent:\s+([\d.]+)"),
                ],
            ),
            judge=JudgeConfig(
                threshold=0.0,
                simplicity_note=(
                    "**Simplicity criterion**: All else being equal, simpler is better. "
                    "A small improvement that adds ugly complexity is not worth it. "
                    "Conversely, removing something and getting equal or better results "
                    "is a great outcome."
                ),
            ),
        )
    else:
        # Default: working MNIST example
        config = Config(
            project=ProjectConfig(
                name="my-autoresearch",
                description="Autonomous ML optimization experiment.",
                goal="Get the highest val_accuracy.",
            ),
            files=FilesConfig(
                editable=["train.py"],
                readonly=["prepare.py"],
            ),
            experiment=ExperimentConfig(
                run_command="python train.py",
                timeout=60,
                setup_command="python prepare.py",
                log_file="run.log",
                constraints=(
                    "Keep the model small enough to train on CPU in the time budget. "
                    "A GPU is not required for this example."
                ),
            ),
            metric=MetricConfig(
                name="val_accuracy",
                pattern=r"^val_accuracy:\s+([\d.]+)",
                direction="maximize",
                secondary=[
                    SecondaryMetric(name="val_loss", pattern=r"^val_loss:\s+([\d.]+)"),
                    SecondaryMetric(name="num_params", pattern=r"^num_params:\s+([\d]+)"),
                ],
            ),
            judge=JudgeConfig(
                threshold=0.0,
                simplicity_note=(
                    "**Simplicity criterion**: All else being equal, simpler is better. "
                    "A tiny accuracy gain that adds ugly complexity is not worth it. "
                    "Removing complexity while keeping accuracy is a great outcome."
                ),
            ),
        )

    # --- 1. Generate template files ---
    print("Creating project files:")
    _write_file_if_needed("train.py", DEFAULT_TRAIN_PY, args.force)
    _write_file_if_needed("prepare.py", DEFAULT_PREPARE_PY, args.force)
    _write_file_if_needed("CLAUDE.md", DEFAULT_CLAUDE_MD, args.force)

    # --- 2. Generate API key ---
    print("\nConnecting to dashboard...")
    api_key = _generate_api_key(config)
    if api_key:
        config.api.key = api_key
        print(f"  API key: {api_key}")
    else:
        print("  Could not reach server (you can run `ars key` later)")

    # --- 3. Write autoresearch.yaml ---
    yaml_str = config_to_yaml(config)
    with open(CONFIG_FILENAME, "w") as f:
        f.write(yaml_str)
    print(f"\n  {CONFIG_FILENAME}")

    # --- 4. Generate program.md ---
    program = generate_program_md(config)
    with open("program.md", "w") as f:
        f.write(program)
    print("  program.md")

    # --- 5. Print summary ---
    if api_key:
        base_url = config.api.endpoint.rsplit("/api", 1)[0]
        dashboard_url = f"{base_url}/dashboard/?key={api_key}"
        print(f"\n  Dashboard: {dashboard_url}")

    print("\nDone! Launch Claude and tell it:")
    print("  \"configure and start the experiments\"")


def cmd_setup(args):
    """Set up a fresh experiment run."""
    config = _load_config_or_exit()
    errors = validate_config(config)
    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    tag = args.tag
    branch = f"autoresearch/{tag}"

    # Check branch doesn't exist
    if not args.skip_branch:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Error: branch {branch} already exists. Pick a different tag.")
            sys.exit(1)

        _git("checkout", "-b", branch)
        print(f"Created branch: {branch}")

    # Init tracker
    tracker = _load_tracker(config)
    tracker.local.set_meta("run_tag", tag)
    tracker.local.set_meta("project_name", config.project.name)

    # Run setup command if configured
    if config.experiment.setup_command and not args.skip_setup:
        print(f"Running setup: {config.experiment.setup_command}")
        result = subprocess.run(config.experiment.setup_command, shell=True)
        if result.returncode != 0:
            print("Warning: setup command exited with non-zero status.")

    # Regenerate program.md with current config
    program = generate_program_md(config)
    with open("program.md", "w") as f:
        f.write(program)

    tracker.close()

    print(f"\nSetup complete!")
    print(f"  Branch: {branch}")
    print(f"  Metric: {config.metric.name} ({config.metric.direction})")
    print(f"  Run command: {config.experiment.run_command}")
    print(f"  Timeout: {config.experiment.timeout}s")
    print(f"\nReady. Run baseline with: ars run --description \"baseline\"")


def cmd_run(args):
    """Execute an experiment with timeout and output capture."""
    config = _load_config_or_exit()
    tracker = _load_tracker(config)
    run_tag = _get_run_tag(tracker)

    command = args.run_command or config.experiment.run_command
    timeout = args.timeout or config.experiment.timeout
    log_file = config.experiment.log_file
    description = args.description or ""

    # Create experiment record
    exp_num = tracker.local.next_experiment_number(run_tag)
    commit_hash = _git_commit_hash()
    parent_hash = ""
    try:
        parent_hash = _git("rev-parse", "--short=7", "HEAD~1")
    except subprocess.CalledProcessError:
        pass

    exp_id = tracker.create_experiment(
        run_tag=run_tag,
        experiment_number=exp_num,
        commit_hash=commit_hash,
        parent_commit_hash=parent_hash,
        status="running",
        description=description,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    tracker.sync(exp_id)

    print(f"Run #{exp_num} started | commit: {commit_hash} | timeout: {timeout}s")
    print(f"Command: {command}")
    print(f"Log: {log_file}")

    # Execute
    result = run_experiment(command, log_file, timeout)

    # Update record
    now = datetime.now(timezone.utc).isoformat()
    if result.timed_out:
        status = "crash"
        reason = "timeout"
        tail = read_log_tail(log_file)
        tracker.update_experiment(exp_id,
                                  status=status,
                                  duration_seconds=result.duration_seconds,
                                  finished_at=now,
                                  stdout_tail=tail)
        print(f"\nRun #{exp_num} TIMEOUT after {result.duration_seconds:.0f}s")
    elif result.exit_code != 0:
        status = "crash"
        tail = read_log_tail(log_file)
        tracker.update_experiment(exp_id,
                                  status=status,
                                  duration_seconds=result.duration_seconds,
                                  finished_at=now,
                                  stdout_tail=tail)
        print(f"\nRun #{exp_num} CRASHED (exit code {result.exit_code}) after {result.duration_seconds:.0f}s")
        print(f"Tail of log:\n{tail[-500:]}")
    else:
        tracker.update_experiment(exp_id,
                                  duration_seconds=result.duration_seconds,
                                  finished_at=now)
        print(f"\nRun #{exp_num} finished in {result.duration_seconds:.0f}s | exit code: 0")

    tracker.sync(exp_id)
    tracker.close()


def cmd_log(args):
    """Extract metrics from run output and record results."""
    config = _load_config_or_exit()
    tracker = _load_tracker(config)

    latest = tracker.get_latest()
    if not latest:
        print("Error: no experiments found. Run `ars run` first.")
        sys.exit(1)

    log_file = args.file or config.experiment.log_file
    description = args.description

    # Extract primary metric
    if args.metric is not None:
        metric_value = args.metric
    else:
        metric_value = _extract_metric(log_file, config.metric.pattern)

    # Extract secondary metrics
    secondary = {}
    for sm in config.metric.secondary:
        val = _extract_metric(log_file, sm.pattern)
        if val is not None:
            secondary[sm.name] = val

    # Extract diff
    diff = _git_diff_from_parent()

    # Update record
    updates = {
        "metric_value": metric_value,
        "secondary_metrics": secondary,
        "diff": diff,
    }
    if description:
        updates["description"] = description

    tracker.update_experiment(latest.id, **updates)
    tracker.sync(latest.id)

    # Print results
    print(f"Experiment #{latest.experiment_number}:")
    if metric_value is not None:
        print(f"  {config.metric.name}: {metric_value:.6f}")
    else:
        print(f"  {config.metric.name}: NOT FOUND (crash?)")
    for name, val in secondary.items():
        print(f"  {name}: {val:.1f}")
    if description:
        print(f"  Description: {description}")

    tracker.close()


def cmd_judge(args):
    """Decide keep/discard based on metric comparison with current best."""
    config = _load_config_or_exit()
    tracker = _load_tracker(config)

    latest = tracker.get_latest()
    if not latest:
        print("Error: no experiments found.")
        sys.exit(1)

    judge = Judge(config, tracker)

    # Force overrides
    if args.force_keep:
        tracker.update_experiment(latest.id, status="keep")
        tracker.sync(latest.id)
        print(f"KEEP (forced): experiment #{latest.experiment_number}")
        tracker.close()
        return

    if args.force_discard:
        tracker.update_experiment(latest.id, status="discard")
        try:
            _git("reset", "--hard", "HEAD~1")
            reverted_to = _git_commit_hash()
            print(f"DISCARD (forced): experiment #{latest.experiment_number}")
            print(f"Reverted to {reverted_to}")
        except subprocess.CalledProcessError:
            print(f"DISCARD (forced): experiment #{latest.experiment_number}")
            print("Warning: git reset failed")
        tracker.sync(latest.id)
        tracker.close()
        return

    # Reload latest with metric values
    latest = tracker.local.get_experiment(latest.id)

    result = judge.evaluate(latest)
    judge.apply(latest, result)

    # Print result
    label = result.decision.upper()
    if result.is_baseline:
        print(f"{label} (baseline): {result.reason}")
    elif result.decision == "keep":
        print(f"{label}: {result.reason}")
    elif result.decision == "discard":
        reverted_to = _git_commit_hash()
        print(f"{label}: {result.reason}")
        print(f"Reverted to {reverted_to}")
    elif result.decision == "crash":
        reverted_to = _git_commit_hash()
        print(f"{label}: {result.reason}")
        print(f"Reverted to {reverted_to}")

    tracker.close()


def cmd_results(args):
    """Show results table."""
    config = _load_config_or_exit()
    tracker = _load_tracker(config)

    experiments = tracker.get_all(status=args.status, last_n=args.last)

    if not experiments:
        print("No experiments recorded yet.")
        tracker.close()
        return

    fmt = args.format

    if fmt == "tsv":
        print("commit\t{}\tmemory_gb\tstatus\tdescription".format(config.metric.name))
        for e in experiments:
            metric_str = f"{e.metric_value:.6f}" if e.metric_value is not None else "0.000000"
            mem = e.secondary_metrics.get("peak_vram_mb", 0)
            mem_gb = mem / 1024 if mem else 0.0
            print(f"{e.commit_hash}\t{metric_str}\t{mem_gb:.1f}\t{e.status}\t{e.description}")

    elif fmt == "json":
        import json
        data = []
        for e in experiments:
            data.append({
                "experiment": e.experiment_number,
                "commit": e.commit_hash,
                "metric": e.metric_value,
                "status": e.status,
                "description": e.description,
                "duration": e.duration_seconds,
                "secondary": e.secondary_metrics,
            })
        print(json.dumps(data, indent=2))

    else:  # table
        # Header
        metric_col = config.metric.name
        print(f"{'#':>4}  {'commit':>7}  {metric_col:>12}  {'status':>8}  {'dur':>5}  description")
        print("-" * 80)

        best = tracker.get_best()
        best_id = best.id if best else None

        for e in experiments:
            num = e.experiment_number
            commit = e.commit_hash[:7] if e.commit_hash else "-------"
            metric_str = f"{e.metric_value:.6f}" if e.metric_value is not None else "   ---   "
            status = e.status
            dur = f"{e.duration_seconds:.0f}s" if e.duration_seconds else "  -"
            desc = e.description or ""
            marker = " *" if e.id == best_id else ""
            print(f"{num:4d}  {commit}  {metric_str:>12}  {status:>8}  {dur:>5}  {desc}{marker}")

        if best:
            print(f"\n* Best: {config.metric.name} = {best.metric_value:.6f} (experiment #{best.experiment_number})")

    tracker.close()


def cmd_status(args):
    """Show current project state."""
    config = _load_config_or_exit()
    tracker = _load_tracker(config)

    branch = _git_branch()
    commit = _git_commit_hash()
    run_tag = tracker.local.get_meta("run_tag") or "(none)"
    stats = tracker.get_stats()
    best = tracker.get_best()
    latest = tracker.get_latest()

    print(f"Project:     {config.project.name}")
    print(f"Branch:      {branch}")
    print(f"Commit:      {commit}")
    print(f"Run tag:     {run_tag}")
    print(f"Metric:      {config.metric.name} ({config.metric.direction})")
    print(f"Run command: {config.experiment.run_command}")

    if best:
        print(f"Best:        {config.metric.name} = {best.metric_value:.6f} (experiment #{best.experiment_number})")
    else:
        print(f"Best:        (no results yet)")

    total = stats.get("total", 0)
    keep = stats.get("keep", 0)
    discard = stats.get("discard", 0)
    crash = stats.get("crash", 0)
    running = stats.get("running", 0)
    print(f"Experiments: {total} total ({keep} keep, {discard} discard, {crash} crash, {running} running)")

    if latest and latest.status != "running":
        val = latest.metric_value if latest.metric_value is not None else 0.0
        print(f"Last:        #{latest.experiment_number} {latest.status} "
              f"{config.metric.name}={val:.6f} "
              f"\"{latest.description}\"")

    api_status = "connected" if config.api.key else "not configured"
    print(f"API:         {api_status}")

    tracker.close()


def cmd_key(args):
    """Generate a new API key and save it to autoresearch.yaml."""
    import requests as req

    config = _load_config_or_exit()
    endpoint = config.api.endpoint.rstrip("/")
    # endpoint is like https://autoresearch.studio/api
    # keys endpoint is at /api/keys/
    base_url = endpoint.rsplit("/api", 1)[0]
    url = f"{base_url}/api/keys/"

    name = config.project.name or "cli"

    try:
        resp = req.post(url, json={"name": name}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error: could not generate key: {e}")
        sys.exit(1)

    key = resp.json()["key"]

    # Write key into autoresearch.yaml
    with open(CONFIG_FILENAME) as f:
        raw = f.read()

    # Replace existing key line or add it
    import re as re_mod
    if re_mod.search(r"^(\s*key:)", raw, re_mod.MULTILINE):
        raw = re_mod.sub(r"^(\s*key:).*", r"\1 " + key, raw, flags=re_mod.MULTILINE)
    else:
        raw = raw.rstrip() + f"\n  key: {key}\n"

    with open(CONFIG_FILENAME, "w") as f:
        f.write(raw)

    print(f"API key: {key}")
    print(f"Saved to {CONFIG_FILENAME}")
    print(f"Dashboard: {base_url}/dashboard/?key={key}")

    # Sync ALL experiments (including running) with the new key
    config = load_config()  # reload with new key
    tracker = _load_tracker(config)
    all_exps = tracker.get_all()
    unsynced = [e for e in all_exps if e.synced_at is None]
    if unsynced:
        for e in unsynced:
            tracker.sync(e.id)
        import time
        time.sleep(2)
        print(f"Synced {len(unsynced)} experiment(s) to dashboard.")


def cmd_generate(args):
    """Regenerate program.md from current config."""
    config = _load_config_or_exit()
    program = generate_program_md(config)

    output = args.output or "program.md"
    with open(output, "w") as f:
        f.write(program)
    print(f"Generated {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="ars",
        description="autoresearchstudio — autonomous ML research framework",
    )
    parser.add_argument("--version", action="version", version=f"ars {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # ars init
    p_init = subparsers.add_parser("init", help="Initialize a new autoresearch project")
    p_init.add_argument("--from-template", default=None, choices=["karpathy"],
                        help="Use a predefined template")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite existing config")

    # ars setup
    p_setup = subparsers.add_parser("setup", help="Set up a fresh experiment run")
    p_setup.add_argument("--tag", required=True, help="Run tag (e.g. mar21)")
    p_setup.add_argument("--skip-branch", action="store_true",
                         help="Don't create a new git branch")
    p_setup.add_argument("--skip-setup", action="store_true",
                         help="Don't run the setup command")

    # ars run
    p_run = subparsers.add_parser("run", help="Execute an experiment")
    p_run.add_argument("run_command", nargs="?", default=None,
                       help="Command to run (default from config)")
    p_run.add_argument("--timeout", type=int, default=None,
                       help="Timeout in seconds (default from config)")
    p_run.add_argument("--description", "-d", default=None,
                       help="Description of this experiment")

    # ars log
    p_log = subparsers.add_parser("log", help="Extract and record metrics from latest run")
    p_log.add_argument("--file", "-f", default=None,
                       help="Log file to read (default from config)")
    p_log.add_argument("--description", "-d", default=None,
                       help="Experiment description")
    p_log.add_argument("--metric", type=float, default=None,
                       help="Manually provide metric value")

    # ars judge
    p_judge = subparsers.add_parser("judge", help="Decide keep/discard")
    p_judge.add_argument("--force-keep", action="store_true")
    p_judge.add_argument("--force-discard", action="store_true")
    p_judge.add_argument("--threshold", type=float, default=None)

    # ars results
    p_results = subparsers.add_parser("results", help="Show results table")
    p_results.add_argument("--format", choices=["table", "tsv", "json"], default="table")
    p_results.add_argument("--status", default=None,
                           help="Filter by status (keep, discard, crash)")
    p_results.add_argument("--last", type=int, default=None,
                           help="Show last N experiments")

    # ars status
    subparsers.add_parser("status", help="Show current project state")

    # ars key
    subparsers.add_parser("key", help="Generate a new API key and save to config")

    # ars generate
    p_gen = subparsers.add_parser("generate", help="Regenerate program.md from config")
    p_gen.add_argument("--output", "-o", default=None,
                       help="Output file (default: program.md)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "init": cmd_init,
        "setup": cmd_setup,
        "run": cmd_run,
        "log": cmd_log,
        "judge": cmd_judge,
        "results": cmd_results,
        "status": cmd_status,
        "key": cmd_key,
        "generate": cmd_generate,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()
        sys.exit(1)
