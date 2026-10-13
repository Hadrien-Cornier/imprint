# RMBench Stage-1 Memory Pretraining

This directory contains the source-controlled `mme_vla_suite` Stage-1 memory
bottleneck trainer used by the RMBench MolmoAct2 memory pipeline.

## Feature Sources

The offline Stage-1 trainer selects the recent visual stream with
`--recent_feature_source`:

| Source | Store file | Memory input | Reconstruction target |
| --- | --- | --- | --- |
| `siglip` | `recent_siglip.npy` | Existing configured SigLIP width | Existing `align_prefix.npy` target |
| `dinov3` | `dinov3.npy` | 768 | Queried three-view DINO grid, 192 tokens x 768 |

`siglip` remains the default to preserve existing runs. DINOv3 is supported
for offline stores because its recent stream and reconstruction target are both
precomputed in the store.

The DINO config at
[`mme_vla_suite/configs/rmbench_all9_compact_nl256_dinov3.yaml`](mme_vla_suite/configs/rmbench_all9_compact_nl256_dinov3.yaml)
matches the published DINO Stage-1 architecture:

- `cosmo_bottleneck_memory.siglip_dim: 768`
- `cosmo_decoder.align_num_patches: 192`
- `cosmo_decoder.align_dim: 768`

The field remains named `siglip_dim` for checkpoint and module compatibility;
it denotes the recent-visual input width.

## Environment

Use the repository's OpenPI/JAX Stage-1 runtime and expose both the vendored
package and OpenPI source:

```bash
export PYTHONPATH="$PWD/experiments/stage1:/path/to/openpi/src:${PYTHONPATH:-}"
export OPENPI_DATA_HOME=/path/to/openpi-cache
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_NVLS_ENABLE=0
```

## DINOv3 Training

The following is the one-step smoke command validated against the released
450-episode DINO store. Increase `--num_steps` and `--batch_size` for a real
run.

```bash
python -u -m mme_vla_suite.scripts.pretrain_bottleneck \
  --config experiments/stage1/mme_vla_suite/configs/rmbench_all9_compact_nl256_dinov3.yaml \
  --store /path/to/rmbench/stage1 \
  --recent_feature_source dinov3 \
  --num_steps 1 \
  --batch_size 1 \
  --fsdp_devices 1 \
  --latent_buckets 4096 \
  --max_recent_tokens 768 \
  --max_step_tokens 1244 \
  --checkpoint_dir /tmp/stage1-dinov3-smoke \
  --ckpt_every 1 \
  --log_every 1
```

At startup, the trainer validates every episode's selected feature file and
dimension. For DINOv3 it also validates the 192 reconstruction tokens per
frame, so a mismatched store/config fails before the first optimizer step.

## SigLIP Training

Existing SigLIP commands remain valid because the default source is `siglip`.
Use a config with a `cosmo_bottleneck_memory.siglip_dim` matching
`recent_siglip.npy` and an `align_prefix.npy` target in every episode:

```bash
python -u -m mme_vla_suite.scripts.pretrain_bottleneck \
  --config /path/to/siglip-stage1.yaml \
  --store /path/to/siglip-store \
  --recent_feature_source siglip
```
