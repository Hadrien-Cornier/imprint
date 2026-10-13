---
pretty_name: IMPRINT RoboCasa 600 Training Store
tags:
  - robotics
  - robocasa
  - dinov3
  - vision-language-action
---

# IMPRINT RoboCasa 600 Training Store

This public repository contains the immutable shared store used by the
canonical IMPRINT customized RoboCasa experiments:

- 600 demonstrations
- 12 task groups
- horizon-20 policy samples
- three-view pooled DINOv3 features
- eight history variants
- strictly causal variant-matched actions

The release is packaged as deterministic tar shards. Download the repository,
then reconstruct the `store/` root with:

```bash
python scripts/reproduce/extract_dataset_store.py \
  --archive-root /path/to/download \
  --output artifacts/datasets/robocasa
```

See the manifests for per-shard SHA256 checksums. The exact patched simulator
source is under `simulation/robocasa/environment/` in
`Amazon-FAR/imprint_datasets`.

The corresponding code, launch configurations, and validation tools are
published in the IMPRINT GitHub repository.
