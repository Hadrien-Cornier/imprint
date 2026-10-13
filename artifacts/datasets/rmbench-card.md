---
pretty_name: IMPRINT RMBench 450 Training Store
tags:
  - robotics
  - rmbench
  - dinov3
  - vision-language-action
---

# IMPRINT RMBench 450 Training Store

This public repository contains the immutable stores used by the canonical
IMPRINT RMBench experiments:

- `stage1/`: 450-episode DINOv3 feature and causal-history store
- `stage2/`: horizon-50 policy samples for the same 450 demonstrations

The release is packaged as deterministic tar shards. Download the repository,
then reconstruct both roots with:

```bash
python scripts/reproduce/extract_dataset_store.py \
  --archive-root /path/to/download \
  --output artifacts/datasets/rmbench
```

The resulting `stage1/` and `stage2/` directories are consumed directly by
the release launchers. See the manifests for per-shard SHA256 checksums.

The corresponding code, launch configurations, and validation tools are
published in the IMPRINT GitHub repository.
