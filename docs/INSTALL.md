# Installation

IMPRINT uses one training environment and two benchmark-specific evaluation
environments. Keeping them separate avoids simulator dependency conflicts.

| Process | Python | Environment |
| --- | --- | --- |
| Training and policy server | 3.11 | Root `.venv` |
| RMBench simulator | Benchmark-defined | Dedicated RMBench environment |
| RoboCasa simulator | 3.10 | Patched `.venv-mem` |

## Training

Requirements:

- Linux x86_64
- Python 3.11
- NVIDIA driver compatible with CUDA 12.8
- Four 80 GB GPUs for the paper training configurations
- `uv`

RMBench evaluation also requires CuRobo's compiled CUDA extensions. The setup
helper builds them with the full CUDA toolkit and `nvcc`, or installs a
compatible wheel supplied through `IMPRINT_CUROBO_WHEEL`.

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --python 3.11 --frozen
source .venv/bin/activate
python -c "import jax; print(jax.devices())"
```

The lock uses `torch==2.7.1+cu128`, matching the evaluated policy runtime and
supporting L40S, H200, and B200 GPUs.

On shared machines where the default `uv` cache is not writable, use a
repository-local cache:

```bash
UV_CACHE_DIR=$PWD/.uv-cache \
  UV_PROJECT_ENVIRONMENT=$PWD/.venv \
  uv sync --python 3.11 --frozen
```

The default OpenPI download helper can fetch `pi05_base`. A local checkpoint can
instead be supplied through `IMPRINT_PI05_PARAMS`.

The DINOv3 feature-store builder uses the pinned `transformers` and `torch`
versions from the root environment.

Online evaluation also uses DINOv3 ViT-B/16 for the recent-frame stream:

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --local-dir artifacts/models/dinov3-vitb16
export DINOV3_MODEL_PATH=$PWD/artifacts/models/dinov3-vitb16
```

The memory tokenizer uses the official WAN2.2 VAE:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  vae/config.json vae/diffusion_pytorch_model.safetensors \
  --revision b8fff7315c768468a5333511427288870b2e9635 \
  --local-dir artifacts/models/wan22-ti2v-5b-local
```

Use the root lockfile: its `diffusers==0.38.0`,
`huggingface-hub==0.36.2`, and `transformers==4.56.2` versions match the
validated policy environment.

## RMBench Evaluation

Clone the `RoboTwin-Mem` directory distributed inside EventVLA, then use the
release helper to create its isolated environment:

```bash
git clone https://github.com/InternRobotics/EventVLA.git .external/EventVLA
git -C .external/EventVLA checkout \
  a4e02f741f78ab01eb76f6a0bc6685f72f60ed24
export RMBENCH_ROOT=$PWD/.external/EventVLA/RoboTwin-Mem
bash scripts/reproduce/setup_rmbench_eval.sh \
  "$RMBENCH_ROOT" .external/rmbench-env

python scripts/reproduce/download_rmbench_assets.py \
  "$RMBENCH_ROOT" --max-workers 16

export IMPRINT_PYTHON=/path/to/imprint/.venv/bin/python
export RMBENCH_PYTHON=$PWD/.external/rmbench-env/.venv/bin/python
export DINOV3_MODEL_PATH=/path/to/dinov3-vitb16
```

The helper installs `script/requirements.txt`, applies the SAPIEN and MPLib
compatibility edits used by RoboTwin-Mem, installs CuRobo v0.7.8, and verifies
the simulator imports. It defaults `TORCH_CUDA_ARCH_LIST` to
`8.9;9.0+PTX`; export a different value before setup when targeting another
architecture. The IMPRINT asset helper replaces the stale upstream filter,
pins the dataset revision, and generates embodiment configs noninteractively.

The default upstream RMBench environment pins PyTorch 2.4.1/CUDA 12.1. Use an
L40S or a graphics-enabled H100/H200 host for that stack. B200 requires a
separately validated PyTorch and CuRobo build with `sm_100` support; changing
only `TORCH_CUDA_ARCH_LIST` is insufficient.

For a renderer host without `nvcc`, first build CuRobo v0.7.8 on a compatible
CUDA development host, then set:

```bash
export IMPRINT_CUROBO_WHEEL=\
/path/to/nvidia_curobo-0.7.8-cp310-cp310-linux_x86_64.whl
```

The wheel must match Linux x86_64, Python 3.10, the installed PyTorch/CUDA
stack, and the target GPU architecture.

The policy process imports IMPRINT and runs on one GPU. The simulator process
imports RMBench and runs on a second GPU. The adapter under
`benchmarks/rmbench/eval/runtime_overlay/adapter` preserves the official raw-TCP
policy contract without importing simulator packages into the policy process.

RMBench rendering also requires the renderer dependencies specified by the
upstream benchmark, including the NVIDIA Vulkan/GLX driver exposed inside the
container. `run_local.py` checks CUDA execution, the CuRobo planner import, and
SAPIEN initialization before loading the policy, and records the diagnostic in
`simulator_preflight.log`. The release does not redistribute those driver
binaries.

## RoboCasa Evaluation

Use a separate Python 3.10 environment with the patched RoboCasa 1.0.1 replay
environment described in `benchmarks/robocasa/README.md`. Do not install a
newer unpatched RoboCasa package over that environment: archived scene indices,
the framed-window alias, and the custom memory tasks are required for replay.

The exact environment archive is available under
`simulation/robocasa/environment/` in
[`Amazon-FAR/imprint_datasets`](https://huggingface.co/datasets/Amazon-FAR/imprint_datasets).
Recreate it with `scripts/reproduce/setup_robocasa_eval.sh`; Python 3.11
applies to the IMPRINT training and policy environment, not the simulator. The
setup helper downloads the official RoboCasa assets and the pinned legacy 0.2
additions automatically.

Use `benchmarks/robocasa/eval/run_local.py` to coordinate the policy and
simulator processes. Its simulator subprocess applies the legacy scene-index
and transient-scene construction fixes through `run_eval_compat.py`.

## Containers

`envs/training/Dockerfile` builds the training environment. Simulator
environments and licensed assets are installed separately.
