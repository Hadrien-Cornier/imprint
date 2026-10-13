# RMBench Evaluation Environment

RMBench is the `RoboTwin-Mem` directory in EventVLA. Install its simulator and
renderer in a dedicated environment:

```bash
git clone https://github.com/InternRobotics/EventVLA.git .external/EventVLA
git -C .external/EventVLA checkout \
  a4e02f741f78ab01eb76f6a0bc6685f72f60ed24
export RMBENCH_ROOT=$PWD/.external/EventVLA/RoboTwin-Mem

bash scripts/reproduce/setup_rmbench_eval.sh \
  "$RMBENCH_ROOT" .external/rmbench-env

python scripts/reproduce/download_rmbench_assets.py \
  "$RMBENCH_ROOT" --max-workers 16
```

The helper explicitly installs the upstream requirements before applying the
RoboTwin-Mem SAPIEN/MPLib compatibility edits and installing CuRobo v0.7.8.
Its default build targets are `8.9;9.0+PTX`; override them with
`TORCH_CUDA_ARCH_LIST` before setup when needed.
By default, the host must provide the CUDA development toolkit and `nvcc`.
Alternatively, set `IMPRINT_CUROBO_WHEEL` to a compatible CuRobo v0.7.8 wheel
built for Linux x86_64, Python 3.10, the same PyTorch/CUDA stack, and the target
GPU architecture. The helper installs the wheel's non-Torch dependencies and
runs `uv pip check` without replacing the simulator's pinned PyTorch build.
IMPRINT does not vendor the benchmark, simulator assets, or renderer binaries.

The upstream requirements pin PyTorch 2.4.1/CUDA 12.1. The default recipe is
validated for L40S and graphics-enabled Hopper hosts. A B200 simulator needs a
separate PyTorch/CuRobo build with `sm_100` support.

The local evaluator launches two processes:

- the IMPRINT policy process from the root `.venv`
- the RMBench simulator process from the benchmark environment

Pass both interpreters explicitly:

```bash
python benchmarks/rmbench/eval/run_local.py \
  --rmbench-root /path/to/RMBench \
  --policy-python /path/to/imprint/.venv/bin/python \
  --sim-python /path/to/rmbench/.venv/bin/python \
  --checkpoint /path/to/checkpoint \
  --dinov3-model /path/to/dinov3-vitb16 \
  --architecture full_attention \
  --task cover_blocks \
  --trials 1 \
  --output outputs/rmbench/cover-smoke
```

See `benchmarks/rmbench/eval/EVALUATION_PROTOCOL.md` for the fixed causal,
consolidated, 50-step evaluation contract.
