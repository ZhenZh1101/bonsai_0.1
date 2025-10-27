# Bonsai 0.1

An experimental English language model training pipeline built with MLX for Apple Silicon.
The default model has 121M parameters and uses a Qwen3 tokenizer with a 65,536-token vocabulary subset.

Includes pretraining, Qwen3 teacher distillation, identity tuning, evaluation, and Hugging Face export.

## Setup

Requires an Apple Silicon Mac, Python 3.11 or 3.12, and `uv`.

```bash
uv sync
uv run python -m scripts.gen_identity merge
uv run python -m scripts.gen_identity pretrain
uv run pytest -q
```

## Training

Training scripts are in `runs/`, configuration files in `configs/`, and core code in `bonsai/`.
Prepare datasets, token shards, and teacher weights before training.

Datasets, model weights, logs, caches, and generated outputs are excluded from this repository.
