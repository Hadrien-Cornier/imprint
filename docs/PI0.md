# Native Pi0 Integration

The Stage-1 memory encoder is policy-agnostic. This branch loads the same
RMBench DINOv3 Stage-1 checkpoint used by Pi0.5 and integrates its
`256 x 1024` memory tokens into the native Pi0 action expert.

## Architecture

- The policy prefix is the standard Pi0 PaliGemma/SigLIP representation.
- The action expert is native Pi0 Gemma 300M with width 1024.
- State uses the native continuous `state_proj` suffix token.
- Flow time uses Pi0 `action_time_mlp_in` and `action_time_mlp_out`.
- Each action-expert block cross-attends to the memory tokens before its
  feed-forward layer.
- The memory encoder and policy are jointly finetuned in Stage 2; the Pi0
  image encoder remains frozen.

Pi0.5 adaptive RMSNorm, residual gates, and `time_mlp_*` parameters are not
used by this branch.

## Checkpoint Validation

`NativePi0Stage1BottleneckWeightLoader` rejects incompatible initialization:

1. `pi0_base` must contain `state_proj`, `action_time_mlp_in`, and
   `action_time_mlp_out`.
2. Pi0.5 `time_mlp_in` and `time_mlp_out` roots are rejected.
3. Every non-memory parameter must exist in `pi0_base` with the expected
   shape.
4. Every Stage-1 memory-encoder parameter must be restored.
5. Memory tokens must have width 1024.

Only the memory-to-action cross-attention seam is newly initialized.

## Training

```bash
export IMPRINT_STAGE1_STORE=/path/to/rmbench/stage1
export IMPRINT_STAGE2_STORE=/path/to/rmbench/stage2
export IMPRINT_STAGE1_CHECKPOINT=/path/to/stage1_step160000.npz
export IMPRINT_BASE_PARAMS=/path/to/pi0_base/params
export IMPRINT_OUTPUT_DIR=/path/to/new/pi0-stage2-output

bash scripts/reproduce/train_stage2.sh \
  configs/paper/rmbench/pi0/experiment.env
```

The paper profile uses a 50-action horizon, batch size 64, four-way FSDP,
100,000 updates, and memory caps `4096/768/1244`.

Download the released initialization and final policy with:

```bash
bash scripts/reproduce/download_release.sh rmbench pi0 --model-only
```

## Evaluation

The Pi0 branch's RMBench launcher resolves
`mme_wan_stage2_pi0_memft` and exports `MEMFT_EXPECTED_MODEL_TYPE=pi0`.
The policy server fails before rollout if the configuration resolves to
Pi0.5.

```bash
python benchmarks/rmbench/eval/run_local.py \
  --rmbench-root "$RMBENCH_ROOT" \
  --checkpoint /path/to/pi0/stage2/step-100000 \
  --dinov3-model "$DINOV3_MODEL_PATH" \
  --architecture full_attention \
  --task cover_blocks \
  --trials 1 \
  --policy-python "$PWD/.venv/bin/python" \
  --sim-python "$RMBENCH_PYTHON" \
  --output outputs/rmbench/pi0-cover-smoke
```

Run the native integration tests before a full training or evaluation:

```bash
JAX_PLATFORMS=cpu uv run --frozen pytest -q \
  src/mme_vla_suite/tests/cosmo/test_memory_pi0.py
```
