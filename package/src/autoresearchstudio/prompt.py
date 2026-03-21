"""program.md generation from project config."""

from .config import Config


PROGRAM_TEMPLATE = """# {project_name}

This is an autonomous research experiment managed by autoresearchstudio.

{description}

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Set up the run**: `ars setup --tag <tag>` — creates the branch and initializes tracking.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
{files_to_read}
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is run via `ars run`. The run command executes: `{run_command}`.
The training/evaluation runs for a **fixed timeout of {timeout} seconds**.

**What you CAN do:**
{editable_rules}

**What you CANNOT do:**
{readonly_rules}
- Install new packages or add dependencies.
- Modify the evaluation harness or the metric computation.

**The goal: {goal}** The metric is **{metric_name}** — {direction_text}.
{constraints}
{simplicity_note}
**The first run**: Your very first run should always be to establish the baseline, so you will run the code as is.

## Output format

Once the run finishes, the key metric is extracted automatically from the output log using `ars log`.

Primary metric: `{metric_name}` ({direction_text})
{secondary_metrics_section}
You can check results so far with: `ars results`
You can check current state with: `ars status`

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar21`).

LOOP FOREVER:

1. Check current state: `ars status`
2. Modify {editable_files_inline} with an experimental idea.
3. Commit: `git add {editable_files_inline} && git commit -m "description of change"`
4. Run the experiment: `ars run --description "description of change"`
5. Extract and log results: `ars log --description "description of change"`
6. Judge the result: `ars judge`
7. If the output says KEEP: great, the branch advances with your change.
8. If the output says DISCARD: the commit was reverted automatically, you're back to the previous best.
9. Repeat from step 1.

**Timeout**: Each experiment should take ~{timeout_minutes} minutes total. If a run exceeds the timeout, it is killed and treated as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: If it's something trivial to fix (typo, missing import), fix it and re-run. If the idea is fundamentally broken, just move on — the crash is already logged.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
"""


def generate_program_md(config: Config) -> str:
    """Generate program.md content from the project config."""

    # Files to read section
    files_lines = []
    for f in config.files.readonly:
        files_lines.append(f"   - `{f}` — read-only, do not modify.")
    for f in config.files.editable:
        files_lines.append(f"   - `{f}` — the file you modify.")
    for f in config.files.context:
        files_lines.append(f"   - `{f}` — context/reference.")
    files_to_read = "\n".join(files_lines) if files_lines else "   - (read relevant project files)"

    # Editable rules
    editable_rules_lines = []
    for f in config.files.editable:
        editable_rules_lines.append(
            f"- Modify `{f}` — everything is fair game: architecture, hyperparameters, "
            f"optimizer, training loop, etc."
        )
    editable_rules = "\n".join(editable_rules_lines)

    # Readonly rules
    readonly_rules_lines = []
    for f in config.files.readonly:
        readonly_rules_lines.append(f"- Modify `{f}`. It is read-only.")
    readonly_rules = "\n".join(readonly_rules_lines)

    # Direction text
    if config.metric.direction == "minimize":
        direction_text = "lower is better"
    else:
        direction_text = "higher is better"

    # Editable files inline
    editable_files_inline = " ".join(f"`{f}`" for f in config.files.editable)

    # Constraints
    constraints = ""
    if config.experiment.constraints:
        constraints = f"\n{config.experiment.constraints.strip()}\n"

    # Simplicity note
    simplicity_note = ""
    if config.judge.simplicity_note:
        simplicity_note = f"\n{config.judge.simplicity_note.strip()}\n"

    # Secondary metrics
    secondary_metrics_section = ""
    if config.metric.secondary:
        lines = ["Secondary metrics (extracted automatically):"]
        for s in config.metric.secondary:
            lines.append(f"- `{s.name}`")
        secondary_metrics_section = "\n".join(lines) + "\n"

    # Description
    description = config.project.description if config.project.description else ""

    # Goal
    goal = config.project.goal if config.project.goal else f"Get the best {config.metric.name}."

    # Timeout in minutes
    timeout_minutes = config.experiment.timeout // 60

    return PROGRAM_TEMPLATE.format(
        project_name=config.project.name,
        description=description,
        files_to_read=files_to_read,
        run_command=config.experiment.run_command,
        timeout=config.experiment.timeout,
        timeout_minutes=timeout_minutes,
        editable_rules=editable_rules,
        readonly_rules=readonly_rules,
        goal=goal,
        metric_name=config.metric.name,
        direction_text=direction_text,
        editable_files_inline=editable_files_inline,
        constraints=constraints,
        simplicity_note=simplicity_note,
        secondary_metrics_section=secondary_metrics_section,
    )
