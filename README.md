# autoresearchstudio

Turn any ML training script into an autonomous research loop. An AI agent modifies your code, runs experiments, keeps what improves the metric, discards what doesn't — and repeats indefinitely while you sleep.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), generalized to work with **any** ML project.

```
You sleep 8 hours → the agent runs ~100 experiments → you wake up to a better model.
```

## Quick start (3 commands)

```bash
pip install autoresearchstudio

mkdir my-experiment && cd my-experiment && git init

ars init
```

That's it. `ars init` creates a complete, working project:

```
my-experiment/
├── train.py             ← the AI agent modifies this (model, hyperparameters, training loop)
├── prepare.py           ← read-only evaluation harness (data loading, metric computation)
├── autoresearch.yaml    ← project configuration (metric, timeout, files, API key)
└── program.md           ← auto-generated instructions for the AI agent
```

The default is a working **MNIST classification** example that trains on CPU in ~50 seconds. You can run it as-is or customize it for your own dataset.

**Output:**

```
Creating project files:
  train.py
  prepare.py

Connecting to dashboard...
  API key: ars_bd02b4cc...

  autoresearch.yaml
  program.md

  Dashboard: https://autoresearch.studio/?key=ars_bd02b4cc...

Done! Next steps:
  1. (optional) Customize train.py / prepare.py / autoresearch.yaml for your task
  2. ars setup --tag <tag>        # create branch + download data
  3. Tell Claude: "read program.md and start the experiments"
```

### Start the experiment loop

```bash
# 1. Set up a run (creates a git branch, downloads data)
ars setup --tag mar23

# 2. Open your AI agent and prompt:
#    "Read program.md and start the experiments"
```

The agent will:
1. Run the baseline (code as-is)
2. Modify `train.py` with an idea → commit → run → judge
3. **KEEP** if the metric improved, **DISCARD** (auto-revert) if not
4. Repeat forever until you stop it

### Check results

```bash
ars results

#   commit  val_accuracy    status    dur  description
------------------------------------------------------
  1  a1b2c3d      0.950000      keep    52s  baseline
  2  b2c3d4e      0.972000      keep    51s  increase LR to 0.003 *
  3  c3d4e5f      0.965000   discard    53s  switch to SGD
  4  d4e5f6g      0.000000     crash     3s  double model width (OOM)
  5  e5f6g7h      0.981000      keep    50s  add batch normalization

* Best: val_accuracy = 0.981000 (experiment #5)
```

Or open the **live dashboard** at the URL printed by `ars init` — it shows a chart, experiment tree, and diffs in real time.

---

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
              │  ars run    → execute with timeout        │
              │  ars log    → extract metric from stdout  │
              │  ars judge  → keep or discard + git reset │
              │                                          │
              │  All API sync happens automatically.      │
              └────────────────────┬─────────────────────┘
                                   │
                        (automatic) │ background sync
                                   ▼
                        autoresearch.studio
                        real-time dashboard
```

Each experiment = one git commit. If the metric improves → keep the commit. If not → `git reset --hard HEAD~1`. The branch only moves forward on improvements.

---

## Customizing for your own project

The default MNIST example works out of the box. To use your own dataset/task, edit these files **before** running `ars setup`:

### 1. `prepare.py` — data + evaluation (read-only during experiments)

This file has clear section markers showing what to change:

```python
# ── Constants ──────────────────────────────────────────────────────
TIME_BUDGET = 20          # training time in seconds (adjust for your task)

# ── Data download ──────────────────────────────────────────────────
def download_data():
    """Replace with your data download/preparation."""
    ...

# ── load_data() ────────────────────────────────────────────────────
def load_data():
    """Must return: (train_X, train_y, test_X, test_y) as tensors."""
    ...

# ── evaluate_accuracy() ───────────────────────────────────────────
def evaluate_accuracy(model, images, labels):
    """Fixed evaluation — DO NOT MODIFY during experiments."""
    ...
```

### 2. `train.py` — model + training (editable during experiments)

```python
# ── Model (modify this) ───────────────────────────────────────────
class Net(nn.Module):
    ...

# ── Hyperparameters (modify these) ────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# ── Training loop (modify this) ───────────────────────────────────
def main():
    ...

# ── Evaluation (do not change output format) ──────────────────────
    print(f"val_accuracy:     {accuracy:.6f}")
    print(f"val_loss:         {val_loss:.6f}")
```

### 3. `autoresearch.yaml` — configuration

```yaml
project:
  name: my-classifier
  goal: Get the highest val_accuracy.

files:
  editable:
    - train.py            # the file(s) the agent CAN modify
  readonly:
    - prepare.py          # files the agent CANNOT touch

experiment:
  run_command: python train.py
  timeout: 60             # kill after N seconds
  setup_command: python prepare.py

metric:
  name: val_accuracy
  pattern: "^val_accuracy:\\s+([\\d.]+)"
  direction: maximize     # or "minimize" for loss

  secondary:
    - name: val_loss
      pattern: "^val_loss:\\s+([\\d.]+)"

judge:
  threshold: 0.0          # minimum improvement to keep (0 = any)
```

After editing, regenerate the agent instructions:

```bash
ars generate   # updates program.md from autoresearch.yaml
```

**Or** just tell Claude: *"I'm working on CIFAR-10 classification, adapt the project for my case"* — it can modify all files before the experiment loop begins.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `ars init` | Create a complete project with working defaults, API key, and dashboard URL |
| `ars setup --tag <tag>` | Create git branch `autoresearch/<tag>`, init tracking DB, run setup command |
| `ars run [-d "desc"]` | Execute experiment with timeout, capture output to log |
| `ars log [-d "desc"]` | Extract metrics from output log via regex |
| `ars judge` | Compare with best: KEEP (advance) or DISCARD (auto-revert) |
| `ars results` | Show results table (`--format table|tsv|json`) |
| `ars status` | Current state: branch, best metric, experiment counts |
| `ars key` | Generate a new API key (already done by `ars init`) |
| `ars generate` | Regenerate `program.md` from current config |

Options for `ars init`:
- `--force` — overwrite existing files
- `--from-template karpathy` — use the Karpathy autoresearch template (val_bpb, uv, VRAM tracking)

---

## Tips for best results

**1. Fixed time budget, not fixed epochs.** Makes experiments comparable regardless of model/batch size changes.

```python
TIME_BUDGET = 300  # seconds
t0 = time.time()
while time.time() - t0 < TIME_BUDGET:
    train_one_step()
```

**2. Print clear metrics.** Simple, grep-able format at the end of training:

```python
print(f"val_accuracy: {accuracy:.6f}")
print(f"val_loss:     {loss:.6f}")
```

**3. Separate eval from training.** Put evaluation in a read-only file so the agent can't game the metric.

**4. Keep scope small.** One editable file with model + training loop is ideal.

**5. Fail fast.** Detect NaN/divergence early:

```python
if math.isnan(loss) or loss > 100:
    print("FAIL")
    exit(1)
```

---

## Dashboard

The live dashboard at `autoresearch.studio` is automatically connected when you run `ars init`. It shows:

- **Progress chart** — keep/discard scatter plot with running best line
- **Experiment tree** — SVG visualization of the git branch history
- **Experiment detail** — full metadata with colored code diffs
- **Auto-refresh** — updates every 5 seconds

All experiment data is synced automatically in the background by the `ars` CLI commands. No manual API calls needed.

### Self-hosting

```bash
cd webapp
pip install django
python manage.py migrate
python manage.py runserver
```

Then set the endpoint in your config:

```yaml
api:
  endpoint: http://localhost:8000/api
```

---

## Project structure

```
autoresearchstudio/
├── package/                    # pip-installable CLI
│   └── src/autoresearchstudio/
│       ├── cli.py              # all ars commands
│       ├── config.py           # YAML config loading + validation
│       ├── runner.py           # subprocess execution + timeout
│       ├── tracker.py          # SQLite local storage + API sync
│       ├── judge.py            # keep/discard decision logic
│       ├── prompt.py           # program.md generation
│       └── templates.py        # default train.py / prepare.py templates
├── webapp/                     # Django dashboard
│   └── dashboard/
│       ├── models.py           # ApiKey, Project, Experiment
│       ├── api.py              # POST /api/experiments/, /api/keys/
│       └── views.py            # dashboard views + chart/tree rendering
└── examples/                   # working examples
    ├── mnist/
    ├── cifar10/
    └── fashion_mnist/
```

## License

MIT
