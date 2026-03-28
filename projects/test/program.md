# my-autoresearch

This is an autonomous research experiment managed by **autoresearchstudio**.
All experiment tracking and dashboard sync are handled automatically by the `ars` CLI commands — you never need to make API calls yourself.

Autonomous ML optimization experiment.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist.
2. **Set up the run**: `ars setup --tag <tag>` — creates the branch, initializes tracking, and runs the setup command (data download etc.).
3. **Read the in-scope files** for full context:
   - `prepare.py` — read-only, do not modify
   - `train.py` — editable, this is what you modify
4. **Confirm and go**: Confirm setup looks good, then start experimenting.

## Files

**Editable** (you modify these during experiments):
- `train.py` — architecture, hyperparameters, optimizer, training loop, etc.

**Read-only** (DO NOT modify during experiments):
- `prepare.py` — DO NOT modify
- The evaluation harness and metric computation
- Do not install new packages or add dependencies

## Goal

**Get the highest val_accuracy.** The metric is **val_accuracy** — higher is better.

Keep the model small enough to train on CPU in the time budget. A GPU is not required for this example.


**Simplicity criterion**: All else being equal, simpler is better. A tiny accuracy gain that adds ugly complexity is not worth it. Removing complexity while keeping accuracy is a great outcome.


## Output format

Metrics are extracted automatically from the output log by `ars log`.

Primary metric: `val_accuracy` (higher is better)
Secondary metrics (extracted automatically):
- `val_loss`
- `num_params`

Check results: `ars results`
Check state: `ars status`

## The experiment loop

**The first run** is always the baseline — run the code as-is to establish the starting point.

LOOP FOREVER:

1. `ars status` — check current state
2. Modify `train.py` with an experimental idea
3. `git add train.py && git commit -m "description of change"`
4. `ars run --description "description of change"`
5. `ars log --description "description of change"`
6. `ars judge`
7. **KEEP** → the branch advances with your change
8. **DISCARD** → the commit was reverted automatically, you're back to the previous best
9. Repeat from step 1

**Timeout**: Each run has a 60s timeout (~1 min). Exceeding it → crash → auto-revert.

**Crashes**: If trivial to fix (typo, missing import), fix and re-run. If fundamentally broken, move on — the crash is already logged.

**NEVER STOP**: Do NOT pause to ask the human if you should continue. The human might be asleep and expects you to work *indefinitely* until manually stopped. If you run out of ideas, think harder — re-read files, combine previous near-misses, try radical changes. The loop runs until the human interrupts you.
