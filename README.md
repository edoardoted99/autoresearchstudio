# autoresearchstudio

Turn any ML training script into an autonomous research loop. An AI agent modifies your code, runs experiments, keeps what improves the metric, discards what doesn't — and repeats indefinitely while you sleep.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), generalized to work with **any** ML project: classification, regression, NLP, vision, RL, fine-tuning, you name it.

```
You sleep 8 hours → the agent runs ~100 experiments → you wake up to a better model.
```

## How it works

```
                    ┌─────────────────────────────────────┐
                    │         Your ML project             │
                    │                                     │
                    │  prepare.py  ← read-only (eval,     │
                    │                 data, constants)     │
                    │  train.py    ← agent modifies this  │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────▼─────────────────────┐
              │            autoresearchstudio             │
              │                                          │
              │  autoresearch.yaml  ← you configure this │
              │  program.md         ← generated for the  │
              │                       AI agent to follow  │
              │                                          │
              │  ars run    → execute experiment          │
              │  ars log    → extract metric from output  │
              │  ars judge  → keep or discard + git reset │
              │                                          │
              └────────────────────┬─────────────────────┘
                                   │
                        (optional) │ API sync
                                   ▼
                        autoresearch.studio
                        real-time dashboard
```

The agent follows a simple loop:

1. Modify the training code with an idea
2. `git commit`
3. `ars run` — run the experiment with a timeout
4. `ars log` — extract the metric from the output
5. `ars judge` — if the metric improved: **keep**. Otherwise: **discard** (auto-reverts the commit)
6. Repeat forever

## Install

```bash
pip install autoresearchstudio
```

## Quick start: add autoresearch to your project

You need an existing ML project with:
- A **training script** that prints a metric to stdout (e.g. `val_accuracy: 0.95`)
- A **git repository**

### Step 1: Initialize

```bash
cd your-ml-project
ars init
```

This creates two files:
- **`autoresearch.yaml`** — the configuration (edit this)
- **`program.md`** — instructions for the AI agent (auto-generated)

### Step 2: Configure `autoresearch.yaml`

Tell the framework about your project:

```yaml
project:
  name: my-classifier
  goal: Get the highest val_accuracy.

files:
  editable:
    - train.py          # the file(s) the agent can modify
  readonly:
    - prepare.py        # files the agent must NOT touch

experiment:
  run_command: python train.py    # how to run one experiment
  timeout: 600                     # kill after N seconds
  setup_command: python prepare.py # one-time setup (optional)

metric:
  name: val_accuracy
  pattern: "^val_accuracy:\\s+([\\d.]+)"   # regex to extract from stdout
  direction: maximize                       # or "minimize" for loss

  secondary:                    # optional extra metrics to track
    - name: val_loss
      pattern: "^val_loss:\\s+([\\d.]+)"

judge:
  threshold: 0.0                # minimum improvement to keep (0 = any)
```

The key things to get right:
- **`metric.pattern`**: a regex that matches a line in your script's output and captures the number. Test it: `grep -P "^val_accuracy:\s+([\d.]+)" run.log`
- **`metric.direction`**: `minimize` for loss/error, `maximize` for accuracy/score
- **`experiment.timeout`**: enough time for one full training run, with some margin

### Step 3: Regenerate the agent instructions

```bash
ars generate
```

This updates `program.md` with the correct commands and metric for your project.

### Step 4: Make your script print the metric

Your training script must print the metric in a format that matches `metric.pattern`. For example:

```python
# At the end of train.py:
print(f"val_accuracy: {accuracy:.6f}")
print(f"val_loss: {val_loss:.6f}")
```

### Step 5: Start the agent

Open your favorite AI coding agent (Claude Code, Codex, Cursor, etc.) in the project directory and prompt:

```
Read program.md and let's kick off a new experiment! Do the setup first.
```

The agent will:
1. Run `ars setup --tag mar21` to create a branch
2. Run the baseline
3. Start the experiment loop — modifying code, running, judging, repeating

**That's it.** Go to sleep. Come back to results.

### Step 6: Check results

```bash
# Pretty table
ars results

#   commit  val_accuracy    status    dur  description
------------------------------------------------------
  1  a1b2c3d      0.950000      keep    120s  baseline
  2  b2c3d4e      0.972000      keep    118s  increase LR to 0.003 *
  3  c3d4e5f      0.965000   discard    121s  switch to SGD
  4  d4e5f6g      0.000000     crash      3s  double model width (OOM)
  5  e5f6g7h      0.981000      keep    125s  add batch normalization

* Best: val_accuracy = 0.981000 (experiment #5)

# Export to TSV (like Karpathy's results.tsv)
ars results --format tsv

# JSON for programmatic access
ars results --format json
```

## Dashboard (optional)

The `webapp/` folder contains a Django + htmx dashboard that shows experiments in real-time.

### Setup

```bash
cd webapp
pip install django
python manage.py migrate
python manage.py runserver
```

Create an API key in the Django admin (`/admin/`), then add it to your project config:

```yaml
# autoresearch.yaml
api:
  key: ars_your_key_here
  endpoint: http://localhost:8000
```

The dashboard shows:
- **Progress chart** — scatter plot with keep/discard points and running best line
- **Experiment tree** — SVG branch visualization (trunk of keeps, branches for discards/crashes)
- **Experiment detail** — full info with colored diff of code changes
- **Auto-refresh** — table updates every 5 seconds via htmx

## CLI reference

| Command | What it does |
|---------|-------------|
| `ars init [--from-template karpathy]` | Create `autoresearch.yaml` + `program.md` |
| `ars setup --tag <tag>` | Create git branch, initialize tracking DB |
| `ars run [--description "..."]` | Run experiment with timeout, capture output |
| `ars log [--description "..."]` | Extract metrics from output via regex |
| `ars judge` | Compare with best, KEEP or DISCARD (auto git reset) |
| `ars results [--format table\|tsv\|json]` | Show results table |
| `ars status` | Current project state, best metric, counts |
| `ars generate` | Regenerate `program.md` from config |

## Making your project autoresearch-friendly

A few tips to get the most out of autonomous experimentation:

**1. Fixed time budget, not fixed epochs.** Your training script should stop after N seconds of wall-clock time, not after N epochs. This makes experiments directly comparable regardless of what the agent changes (model size, batch size, etc.).

```python
import time
TIME_BUDGET = 300  # seconds
t0 = time.time()
while time.time() - t0 < TIME_BUDGET:
    # train one step
```

**2. Print a clear summary at the end.** The agent needs to extract metrics from stdout. Print them in a simple, grep-able format:

```python
print(f"val_accuracy: {accuracy:.6f}")
print(f"val_loss:     {loss:.6f}")
print(f"num_params:   {num_params}")
```

**3. Separate eval from training.** Put your evaluation function in a read-only file (e.g. `prepare.py`) so the agent can't game the metric. The agent should only modify the training code.

**4. Single file to modify.** Keep the scope small. One file that contains the model + training loop is ideal. The agent works best when it can see and modify everything in one place.

**5. Fail fast.** Add early stopping for NaN loss or exploding gradients:

```python
if math.isnan(loss) or loss > 100:
    print("FAIL")
    exit(1)
```

## Project structure

```
autoresearchstudio/
├── package/               # pip installable CLI tool
│   └── src/autoresearchstudio/
│       ├── cli.py         # ars commands
│       ├── config.py      # YAML config
│       ├── runner.py      # subprocess + timeout
│       ├── tracker.py     # SQLite + API sync
│       ├── judge.py       # keep/discard logic
│       └── prompt.py      # program.md generation
├── webapp/                # Django dashboard
│   └── dashboard/
│       ├── models.py      # ApiKey, Project, Experiment
│       ├── api.py         # POST /api/experiments/
│       └── views.py       # dashboard views
├── examples/
│   └── mnist/             # working example
└── autoresearch/          # Karpathy's original (reference)
```

## License

MIT
