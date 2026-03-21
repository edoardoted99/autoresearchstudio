# autoresearchstudio

Generalized autonomous ML research framework. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Install

```bash
pip install autoresearchstudio
```

## Quick start

```bash
# Initialize a project (use --from-template karpathy for LLM pretraining)
ars init

# Edit autoresearch.yaml to match your project

# Start a run
ars setup --tag mar21

# The AI agent then uses these commands in a loop:
ars run --description "baseline"
ars log --description "baseline"
ars judge
```
