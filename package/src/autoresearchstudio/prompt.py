"""program.md generation from project config."""

from .config import Config


PROGRAM_TEMPLATE = """# {project_name}

This is an autonomous research experiment managed by **autoresearchstudio**.
All experiment tracking and dashboard sync are handled automatically by the `ars` CLI commands — you never need to make API calls yourself.

{description}

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist.
2. **Set up the run**: `ars setup --tag <tag>` — creates the branch, initializes tracking, and runs the setup command (data download etc.).
3. **Read the in-scope files** for full context:
{files_to_read}
4. **Confirm and go**: Confirm setup looks good, then start experimenting.

## Files

**Editable** (you modify these during experiments):
{editable_rules}

**Read-only** (DO NOT modify during experiments):
{readonly_rules}
- The evaluation harness and metric computation
- Do not install new packages or add dependencies

## Goal

**{goal}** The metric is **{metric_name}** — {direction_text}.
{constraints}
{simplicity_note}

## Output format

Metrics are extracted automatically from the output log by `ars log`.

Primary metric: `{metric_name}` ({direction_text})
{secondary_metrics_section}
Check results: `ars results`
Check state: `ars status`

## The experiment loop

**The first run** is always the baseline — run the code as-is to establish the starting point.

LOOP FOREVER:

1. `ars status` — check current state
2. Modify {editable_files_inline} with an experimental idea
3. `git add {editable_files_plain} && git commit -m "description of change"`
4. `ars run --description "description of change"`
5. `ars log --description "description of change"`
6. `ars judge`
7. **KEEP** → the branch advances with your change
8. **DISCARD** → the commit was reverted automatically, you're back to the previous best
9. Repeat from step 1

**Timeout**: Each run has a {timeout}s timeout (~{timeout_minutes} min). Exceeding it → crash → auto-revert.

**Crashes**: If trivial to fix (typo, missing import), fix and re-run. If fundamentally broken, move on — the crash is already logged.

**NEVER STOP**: Do NOT pause to ask the human if you should continue. The human might be asleep and expects you to work *indefinitely* until manually stopped. If you run out of ideas, think harder — re-read files, combine previous near-misses, try radical changes. The loop runs until the human interrupts you.
"""


def generate_program_md(config: Config) -> str:
    """Generate program.md content from the project config."""

    # Files to read section
    files_lines = []
    for f in config.files.readonly:
        files_lines.append(f"   - `{f}` — read-only, do not modify")
    for f in config.files.editable:
        files_lines.append(f"   - `{f}` — editable, this is what you modify")
    for f in config.files.context:
        files_lines.append(f"   - `{f}` — context/reference")
    files_to_read = "\n".join(files_lines) if files_lines else "   - (read relevant project files)"

    # Editable rules
    editable_rules_lines = []
    for f in config.files.editable:
        editable_rules_lines.append(
            f"- `{f}` — architecture, hyperparameters, optimizer, training loop, etc."
        )
    editable_rules = "\n".join(editable_rules_lines)

    # Readonly rules
    readonly_rules_lines = []
    for f in config.files.readonly:
        readonly_rules_lines.append(f"- `{f}` — DO NOT modify")
    readonly_rules = "\n".join(readonly_rules_lines)

    # Direction text
    if config.metric.direction == "minimize":
        direction_text = "lower is better"
    else:
        direction_text = "higher is better"

    # Editable files inline
    editable_files_inline = " ".join(f"`{f}`" for f in config.files.editable)
    editable_files_plain = " ".join(config.files.editable)

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
    timeout_minutes = max(1, config.experiment.timeout // 60)

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
        editable_files_plain=editable_files_plain,
        constraints=constraints,
        simplicity_note=simplicity_note,
        secondary_metrics_section=secondary_metrics_section,
    )
