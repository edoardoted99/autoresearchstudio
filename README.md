# autoresearch.studio

Turn any ML training script into an autonomous research loop. An AI agent modifies your code, runs experiments, keeps what improves the metric, discards what doesn't — and repeats indefinitely while you sleep.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), generalized to work with **any** ML project.

```
You sleep 8 hours → the agent runs ~100 experiments → you wake up to a better model.
```

---

## Quick start

```bash
pip install autoresearchstudio
```

```bash
mkdir my-experiment && cd my-experiment && git init
ars init
```

`ars init` creates everything you need — a complete, runnable project:

```
my-experiment/
├── train.py             ← the AI agent modifies this (model, hyperparameters, training loop)
├── prepare.py           ← read-only evaluation harness (data loading, metric computation)
├── autoresearch.yaml    ← project config (metric, timeout, files, API key)
├── program.md           ← auto-generated instructions for the AI agent
└── CLAUDE.md            ← context file so Claude understands the project when launched
```

```
Creating project files:
  train.py
  prepare.py
  CLAUDE.md

Connecting to dashboard...
  API key: ars_bd02b4cc...

  autoresearch.yaml
  program.md

  Dashboard: https://autoresearch.studio/?key=ars_bd02b4cc...
```

The default is a working **MNIST classification** example that trains on CPU in ~50 seconds. Run it as-is, or customize for your own dataset.

### Run the experiments

```bash
# Create a git branch and download data
ars setup --tag mar23

# Open Claude Code (or any AI coding agent) and say:
# "Read program.md and start the experiments"
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
--------------------------------------------------------------------------------
   1  a1b2c3d      0.950000      keep    52s  baseline
   2  b2c3d4e      0.972000      keep    51s  increase LR to 0.003 *
   3  c3d4e5f      0.965000   discard    53s  switch to SGD
   4  d4e5f6g      0.000000     crash     3s  double model width (OOM)
   5  e5f6g7h      0.981000      keep    50s  add batch normalization

* Best: val_accuracy = 0.981000 (experiment #5)
```

Or open the **live dashboard** at the URL printed by `ars init` — chart, experiment tree, and diffs in real time.

---

## How it works

```
              ┌─────────────────────────────────────────┐
              │           Your ML project               │
              │                                         │
              │  prepare.py   ← read-only (eval, data)  │
              │  train.py     ← agent modifies this     │
              └────────────────┬────────────────────────┘
                               │
          ┌────────────────────▼──────────────────────┐
          │           autoresearch.studio              │
          │                                           │
          │  ars run    → execute with timeout         │
          │  ars log    → extract metric from stdout   │
          │  ars judge  → keep or discard + git reset  │
          │                                           │
          │  API sync happens automatically.           │
          └────────────────────┬──────────────────────┘
                               │  background sync
                               ▼
                    autoresearch.studio
                     live dashboard
```

Each experiment = one git commit. If the metric improves → keep. If not → `git reset --hard HEAD~1`. The branch only moves forward on improvements.

---

## Customize for your project

The default MNIST example works out of the box. To use your own dataset, edit these files **before** `ars setup`:

### prepare.py — data + evaluation (read-only during experiments)

```python
# ── Constants ──────────────────────────────────────────────────────
TIME_BUDGET = 20          # training time in seconds — adjust for your task

# ── Data download ──────────────────────────────────────────────────
def download_data():
    """Replace with your data download/preparation."""
    ...

# ── load_data() ────────────────────────────────────────────────────
def load_data():
    """Must return (train_X, train_y, test_X, test_y) as tensors."""
    ...

# ── evaluate_accuracy() ───────────────────────────────────────────
def evaluate_accuracy(model, images, labels):
    """Fixed evaluation — DO NOT MODIFY during experiments."""
    ...
```

### train.py — model + training (editable during experiments)

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

### autoresearch.yaml — configuration

```yaml
project:
  name: my-classifier
  goal: Get the highest val_accuracy.

files:
  editable:
    - train.py
  readonly:
    - prepare.py

experiment:
  run_command: python train.py
  timeout: 60
  setup_command: python prepare.py

metric:
  name: val_accuracy
  pattern: "^val_accuracy:\\s+([\\d.]+)"
  direction: maximize     # or "minimize" for loss

  secondary:
    - name: val_loss
      pattern: "^val_loss:\\s+([\\d.]+)"

judge:
  threshold: 0.0
```

After editing, run `ars generate` to update `program.md`.

Or just tell Claude: *"I'm working on CIFAR-10, adapt the project for my case"* — the `CLAUDE.md` gives it full context to reconfigure everything before experiments start.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `ars init` | Create a complete project with working defaults, API key, and dashboard URL |
| `ars setup --tag <tag>` | Create branch `autoresearch/<tag>`, init tracking, run setup command |
| `ars run [-d "desc"]` | Execute experiment with timeout, capture output |
| `ars log [-d "desc"]` | Extract metrics from output log via regex |
| `ars judge` | Compare with best: KEEP or DISCARD (auto-revert) |
| `ars results` | Show results table (`--format table\|tsv\|json`) |
| `ars status` | Branch, best metric, experiment counts |
| `ars key` | Generate a new API key (already done by `ars init`) |
| `ars generate` | Regenerate `program.md` from config |

`ars init` options:
- `--force` — overwrite existing files
- `--from-template karpathy` — Karpathy autoresearch template (val_bpb, uv, VRAM tracking)

---

## Tips for best results

**Fixed time budget, not fixed epochs.** Makes experiments comparable regardless of model/batch size changes.

```python
TIME_BUDGET = 300  # seconds
t0 = time.time()
while time.time() - t0 < TIME_BUDGET:
    train_one_step()
```

**Print clear metrics.** Simple, grep-able format:

```python
print(f"val_accuracy: {accuracy:.6f}")
print(f"val_loss:     {loss:.6f}")
```

**Separate eval from training.** Put evaluation in a read-only file so the agent can't game the metric.

**Keep scope small.** One editable file with model + training loop is ideal.

**Fail fast.** Detect NaN/divergence early:

```python
if math.isnan(loss) or loss > 100:
    print("FAIL")
    exit(1)
```

---

## Dashboard

The live dashboard at [autoresearch.studio](https://autoresearch.studio) is automatically connected when you run `ars init`. It shows:

- **Progress chart** — keep/discard scatter plot with running best line
- **Experiment tree** — SVG branch visualization (trunk of keeps, branches for discards)
- **Experiment detail** — full metadata with colored code diffs
- **Auto-refresh** — updates every 5 seconds

All sync is handled in the background by the `ars` CLI. No manual API calls needed.

### Self-hosting

```bash
cd webapp
pip install django
python manage.py migrate
python manage.py runserver
```

```yaml
api:
  endpoint: http://localhost:8000/api
```

---

## License

MIT
