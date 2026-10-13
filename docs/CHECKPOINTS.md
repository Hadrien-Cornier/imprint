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

The Pi0.5 and native Pi0 prefixes contain:

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

The MolmoAct2 prefix instead contains `checkpoint/model.pt` and
`checkpoint/config.yaml`. It is an unsharded model-only conversion; optimizer
and trainer state are excluded.
