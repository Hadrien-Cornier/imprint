# Datasets

## RMBench 450

Hugging Face dataset:
[`Amazon-FAR/imprint_datasets`](https://huggingface.co/datasets/Amazon-FAR/imprint_datasets),
path `simulation/rmbench/450/`.

The store contains 450 demonstrations across the nine RMBench tasks. It includes
policy samples, eight history variants per episode, strictly causal
variant-matched actions, proprioception, normalized progress, offline visual
latents, and pooled DINOv3 features.

Paper dimensions:

- RGB views: head, left wrist, right wrist
- source rendering: 320 x 240
- feature extraction: resize/pad to 256 x 256
- DINOv3: 64 pooled patch tokens per view, width 768
- action width: 16
- proprio width: 16
- Stage-1/Stage-2 action-proprio cap: 1244

## RoboCasa 600

Hugging Face dataset:
[`Amazon-FAR/imprint_datasets`](https://huggingface.co/datasets/Amazon-FAR/imprint_datasets),
path `simulation/robocasa/600/`.

The store contains 600 demonstrations across 12 customized RoboCasa tasks. It
uses the same eight-variant, causal-action, offline-latent, and DINOv3 layout.

Paper dimensions:

- RGB views: left agent, right agent, eye in hand
- DINOv3: 64 pooled patch tokens per view, width 768
- action width: 12
- proprio width: 16
- Stage-1/Stage-2 action-proprio cap: 1776

## Evaluation and Environment Assets

- `simulation/rmbench/molmoact2_eval_assets/` contains the RMBench Stage-2
  episode metadata and stored language embeddings used by MolmoAct2.
- `simulation/robocasa/environment/` contains the patched RoboCasa evaluation
  environment archive and its checksums.

## Download

```bash
hf download Amazon-FAR/imprint_datasets \
  --repo-type dataset \
  --include "simulation/rmbench/450/**" \
  --local-dir artifacts/downloads/imprint_datasets
python scripts/reproduce/extract_dataset_store.py \
  --archive-root artifacts/downloads/imprint_datasets/simulation/rmbench/450 \
  --output artifacts/datasets/rmbench

hf download Amazon-FAR/imprint_datasets \
  --repo-type dataset \
  --include "simulation/robocasa/600/**" \
  --local-dir artifacts/downloads/imprint_datasets
python scripts/reproduce/extract_dataset_store.py \
  --archive-root artifacts/downloads/imprint_datasets/simulation/robocasa/600 \
  --output artifacts/datasets/robocasa
```

The stores use deterministic tar shards to avoid hundreds of thousands of
small files. Extraction verifies every shard's byte size and SHA256 before
restoring the original directory layout.

## Rebuilding DINOv3 Features

```bash
python -m mme_vla_suite.dataset_builder.build_dinov3_store \
  --help
```

The builder creates a new immutable derived store and never mutates the source.
It replaces the recent and alignment visual streams with a single
`dinov3.npy`, preserving all other fields. Validate the resulting store with
`scripts/reproduce/validate_store.py`.
