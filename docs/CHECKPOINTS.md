# Checkpoints

All checkpoints are stored in
[`Amazon-FAR/imprint_models`](https://huggingface.co/Amazon-FAR/imprint_models).

| Repository path | Benchmark | Encoder | Stage 1 step | Stage 2 step |
| --- | --- | --- | --- | --- |
| `simulation/rmbench/full_attention/` | RMBench | Perceiver/full attention | 160k | 100k |
| `simulation/rmbench/recurrent/` | RMBench | gated delta hybrid v1 | 160k | 100k |
| `simulation/rmbench/pi0/` | RMBench | Perceiver/full attention + native Pi0 | 160k | 100k |
| `simulation/rmbench/molmoact2/` | RMBench | Perceiver/full attention + MolmoAct2 | 160k | 100k |
| `simulation/robocasa/full_attention/` | RoboCasa | Perceiver/full attention | 160k | 150k |
| `simulation/robocasa/recurrent/` | RoboCasa | gated delta hybrid v1 | 160k | 150k |

Each model prefix contains:

```text
stage1/stage1_step160000.npz
stage2/step-*/_CHECKPOINT_METADATA
stage2/step-*/params/
stage2/step-*/assets/
config/history_config.yaml
config/input_provenance.json
SHA256SUMS
README.md
```

The Stage-2 directories contain complete parameter trees, including the
PaliGemma/Gemma backbone, and are subject to the included Gemma Terms of Use.
Stage-2 `train_state` is omitted because it is not needed for inference.

Download only the native Pi0 checkpoint with:

```bash
bash scripts/reproduce/download_release.sh rmbench pi0 --model-only
```
