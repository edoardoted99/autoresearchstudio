# autoresearch.studio

This project uses **autoresearchstudio** (`ars`) — an autonomous ML experiment framework.
The `ars` CLI handles experiment execution, metric extraction, keep/discard decisions, and dashboard sync automatically.

## What to do when the user says "configure and start the experiments"

### Step 1: Understand the project

Look at the files in this directory. Decide if the project needs configuration:

- **Default MNIST template** (train.py imports from prepare.py, has `Net` with conv layers for 28x28 images): skip to Step 3.
- **User's own project** (their own training script, dataset, etc.): go to Step 2.

### Step 2: Configure (only if needed)

Adapt all files to the user's project. Ask the user what their task is if unclear.

1. **`prepare.py`** — Data loading and evaluation harness (read-only during experiments):
   - `download_data()`: download or locate the dataset
   - `load_data()`: must return `(train_X, train_y, test_X, test_y)` as tensors
   - `evaluate_accuracy(model, images, labels)`: fixed evaluation function (the agent can't game this)
   - `TIME_BUDGET`: training time in seconds

2. **`train.py`** — Model and training loop (the AI agent optimizes this during experiments):
   - Model class, hyperparameters, optimizer, training loop
   - Must import `load_data`, `evaluate_accuracy`, `TIME_BUDGET` from `prepare.py`
   - Must print metrics at the end in a grep-able format, e.g.:
     ```
     val_accuracy:     0.950000
     val_loss:         0.083412
     num_params:       12345
     ```

3. **`autoresearch.yaml`** — Configuration:
   - `project.name`, `project.goal`: describe the task
   - `files.editable` / `files.readonly`: which files the agent can/cannot modify
   - `experiment.run_command`: how to run one experiment (e.g. `python train.py`)
   - `experiment.timeout`: max seconds per experiment
   - `experiment.setup_command`: one-time setup (e.g. `python prepare.py` to download data)
   - `metric.name`, `metric.pattern`, `metric.direction`: what to optimize and how to extract it
   - `metric.secondary`: optional extra metrics to track

4. Run **`ars generate`** to regenerate `program.md` from the updated config.

5. Do a quick sanity check: run `python train.py` to verify it works and prints the expected metric format.

### Step 3: Start the experiments

1. Read `program.md` for the full experiment loop instructions.
2. Agree on a run tag with the user (e.g. `mar23`).
3. Run `ars setup --tag <tag>` to create the git branch and initialize tracking.
4. Follow the experiment loop in `program.md`: baseline → modify → run → judge → repeat forever.

## Key ars commands

```
ars setup --tag <tag>    # create branch + init tracking + run setup command
ars run -d "description" # run experiment with timeout
ars log -d "description" # extract metrics from output
ars judge                # keep if improved, discard (auto-revert) if not
ars status               # current state, best metric
ars results              # full results table
ars generate             # regenerate program.md from autoresearch.yaml
```

## Rules during the experiment loop

- Only modify files listed under `files.editable` in `autoresearch.yaml`
- Never modify `prepare.py` or the evaluation function
- Never install new packages
- Never stop — run indefinitely until the user interrupts
