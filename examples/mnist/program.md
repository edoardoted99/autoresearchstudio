# mnist-autoresearch

This is an autonomous research experiment managed by autoresearchstudio.

Autonomous optimization of a CNN for MNIST digit classification.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Set up the run**: `ars setup --tag <tag>` — creates the branch and initializes tracking.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — read-only, do not modify.
   - `train.py` — the file you modify.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is run via `ars run`. The run command executes: `python train.py`.
The training/evaluation runs for a **fixed timeout of 180 seconds**.

**What you CAN do:**
- Modify `train.py` — everything is fair game: architecture, hyperparameters, optimizer, training loop, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness or the metric computation.

**The goal: Get the highest val_accuracy.** The metric is **val_accuracy** — higher is better.

Keep the model small enough to train on CPU in the time budget. A GPU is not required for this example.


**Simplicity criterion**: All else being equal, simpler is better. A tiny accuracy gain that adds ugly complexity is not worth it. Removing complexity while keeping accuracy is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the code as is.

## Output format

Once the run finishes, the key metric is extracted automatically from the output log using `ars log`.

Primary metric: `val_accuracy` (higher is better)
Secondary metrics (extracted automatically):
- `val_loss`
- `num_params`

You can check results so far with: `ars results`
You can check current state with: `ars status`

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar21`).

LOOP FOREVER:

1. Check current state: `ars status`
2. Modify `train.py` with an experimental idea.
3. Commit: `git add `train.py` && git commit -m "description of change"`
4. Run the experiment: `ars run --description "description of change"`
5. Extract and log results: `ars log --description "description of change"`
6. Judge the result: `ars judge`
7. If the output says KEEP: great, the branch advances with your change.
8. If the output says DISCARD: the commit was reverted automatically, you're back to the previous best.
9. Repeat from step 1.

**Timeout**: Each experiment should take ~3 minutes total. If a run exceeds the timeout, it is killed and treated as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: If it's something trivial to fix (typo, missing import), fix it and re-run. If the idea is fundamentally broken, just move on — the crash is already logged.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
