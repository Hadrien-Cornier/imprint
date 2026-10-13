# RMBench Stage-2 Evaluation

> **IMPRINT release note:** use
> [`../../README.md`](../../README.md#evaluation) as the supported public
> entry point. Models and compact evaluation data are downloaded from
> Hugging Face; the RMBench simulator runtime must be installed separately.

This package evaluates the memory-enabled MolmoAct2 Stage-2 policy in
closed-loop RMBench rollouts. It owns the policy-server integration, canonical
client patch, four-view evaluator overlay, and direct worker launcher. The
simulator runtime remains separate because it contains RMBench and third-party
components that IMPRINT does not redistribute.

## Runtime Installation

Install the canonical RMBench simulator and policy environments on a Linux GPU
host. If you already have the verified root-relative archives, place
`runtime.tar`, `uv-python.tar`, and `runtime-manifest.json` in one local
directory:

```bash
export RMBENCH_RUNTIME_ARCHIVE_DIR=/path/to/rmbench-runtime-archives
sudo -E experiments/scripts/install_rmbench_molmoact2_runtime.sh
```

The worker assumes the runtime is already installed by default. Set
`RMBENCH_SKIP_RUNTIME_STAGE=0` and `RMBENCH_RUNTIME_ARCHIVE_DIR` only when a
worker should install the local archives itself.

The worker uses passwordless `sudo` only for root-relative runtime extraction
and the optional B200 CuRobo rebuild. It resolves `uv` from the invoking
user's home directory when `/root/.local/bin` is absent. For a clean checkout,
use [`bootstrap_molmoact2.sh`](../scripts/bootstrap_molmoact2.sh).

## Training Inputs

Stage 1 produces the Pi0.5-compatible stored language embeddings and DINOv3/
Wan memory encoder initialization. Stage 2 starts from the MolmoAct2 base
checkpoint plus the Stage-1 checkpoint and trains the memory bottleneck to
FiLM-condition the action expert. See
[Stage-2 memory bottleneck](../README.md#stage-2-memory-bottleneck) for
frozen and joint-finetuning commands and the feature schema.

The standalone Stage-1 checkpoint used by the published 100k configuration is
`simulation/rmbench/full_attention/stage1/stage1_step160000.npz` in
`Amazon-FAR/imprint_models`, revision `main`.

Its SHA-256 is
`164c94f3ff8fcd84aee61e88ec0feb97faf99d203eac1be0f4b4146d464d2044`;
closed-loop evaluation does not load this Flax file because the selected
Stage-2 checkpoint already contains the imported PyTorch memory encoder and
FiLM parameters.

Create the public model-only checkpoint and compact evaluation-assets releases
from the exact Stage-2 checkpoint and data roots:

```bash
python experiments/scripts/unshard_rmbench_molmoact2_checkpoint.py \
  --checkpoint /path/to/step100000 \
  --output /path/to/step100000-model-only

export MOLMOACT2_CHECKPOINT=/path/to/step100000-model-only
export RMBENCH_STAGE2_ROOT=/path/to/stage2
export RMBENCH_STAGE1_FEATURE_ROOT=/path/to/stored-language
export RMBENCH_MOLMOACT2_INPUT_REPO=Amazon-FAR/imprint_models
export RMBENCH_MOLMOACT2_INPUT_PREFIX=simulation/rmbench/molmoact2
export RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO=Amazon-FAR/imprint_datasets
export RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX=simulation/rmbench/molmoact2_eval_assets
bash experiments/scripts/prepare_rmbench_molmoact2_smoke_bundle.sh
```

The model repository excludes optimizer and trainer state. The evaluation
assets repository contains only Stage-2 metadata and stored language
embeddings. Source code comes from this branch, while DINOv3 and Qwen are
downloaded from the pinned upstream revisions in the release manifests. The
worker rejects offline DINO/Wan feature arrays.

## Policy and Simulator Flow

The worker starts two processes:

1. `serve_rmbench_molmoact2_tcp.py` runs on the policy GPU. It loads the
   MolmoAct2 checkpoint, Stage-2 memory layout, online DINOv3 and Wan memory
   encoder, and stored Pi0.5 language embedding resolver.
2. The canonical RMBench `eval_policy_client.py` runs on the simulator GPU.
   `patch_policy_client.py` writes a job-local copy with configurable trials,
   timeout, output location, renderer selection, and policy-error propagation.
   `eval_policy_four_view.py` loads canonical `eval_policy.py` and replaces
   only its trial loop to record results and video.

The client connects through the existing raw TCP, four-byte length-prefixed
JSON protocol. NumPy payloads use the server's base64 representation. The
simulator adapter invokes `eval`, maps model action chunks to native RMBench
actions, steps the simulator, and invokes `add_buffer` with each live
post-step observation and executed action.

## Parity Contract

| Contract | Required behavior |
| --- | --- |
| Visual history | Use online DINOv3 and Wan only; never use offline visual arrays. |
| Language | Resolve stored Pi0.5 language embeddings, never new MolmoAct2 text embeddings. |
| Cameras | `head_camera`, `left_camera`, `right_camera` in exactly that order. |
| Pixels | Aspect-preserving 256x256 resize, then RGB-to-BGR conversion. |
| Memory | Reset at episode start and append every live post-step observation/action. |
| Actions | Serve 50 x 16-D model chunks and map to native 14-D before simulator stepping. |
| Replanning | `RMBENCH_EXECUTION_HORIZON` controls executed actions before the next request. |
| Consolidation | `EVAL_CONSOLIDATE_ALL=1`, matching Stage-2 training. |
| DINO cap | The Stage-2 layout and Stage-1 checkpoint must have matching DINO/action/proprio caps. |

The worker exports `MEMFT_DINOV3_REQUIRED=1`, checks the
`online_wan_and_dinov3` bundle declaration, and writes effective seed, flow
seed, and horizon to `runtime_contract.json`.

## Smoke Evaluation

```bash
export RMBENCH_MOLMOACT2_INPUT_REPO=Amazon-FAR/imprint_models
export RMBENCH_MOLMOACT2_INPUT_REVISION=main
export RMBENCH_MOLMOACT2_INPUT_PREFIX=simulation/rmbench/molmoact2
export RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO=Amazon-FAR/imprint_datasets
export RMBENCH_MOLMOACT2_EVAL_ASSETS_REVISION=main
export RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX=simulation/rmbench/molmoact2_eval_assets
export RMBENCH_TASK=cover_blocks
export MEMFT_TRIAL_COUNT=1
# Optional: export WANDB_API_KEY through your shell environment.
bash experiments/rmbench_eval/scripts/launch_eval.sh \
  --task "$RMBENCH_TASK" \
  --trials "$MEMFT_TRIAL_COUNT"
```

On a two-GPU host, GPU 0 runs the policy server and GPU 1 runs the
renderer/simulator. To run both processes on one GPU, set
`RMBENCH_SINGLE_GPU=1` before invoking the same command. The launcher executes
the checked-in worker directly:

```bash
bash experiments/rmbench_eval/scripts/launch_eval.sh \
  --task cover_blocks \
  --trials 1
```

B200 workers verify and, when needed, rebuild CuRobo `sm_100` extensions in
the pinned simulator runtime. You can also invoke the backwards-compatible
worker directly:

```bash
bash experiments/scripts/run_molmoact2_rmbench_smoke_worker.sh \
  --input-repo "$RMBENCH_MOLMOACT2_INPUT_REPO" \
  --input-revision "$RMBENCH_MOLMOACT2_INPUT_REVISION" \
  --input-prefix "$RMBENCH_MOLMOACT2_INPUT_PREFIX" \
  --eval-assets-prefix "$RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX" \
  --task cover_blocks \
  --trials 1
```

## Results and Monitoring

Set `RMBENCH_RESULT_HF_REPO` to publish results to a Hugging Face dataset and
`RMBENCH_WANDB_REQUIRED=1` to make telemetry mandatory:

```bash
export RMBENCH_RESULT_HF_REPO=Amazon-FAR/imprint-rmbench-molmoact2-eval-results
export RMBENCH_RESULT_HF_PATH=cover_blocks/smoke
export RMBENCH_WANDB_REQUIRED=1
```

The output directory contains `trials.jsonl`, evaluator/policy/runtime
configuration JSON, policy/evaluator/W&B logs, and five videos per trial:
`head_camera`, `left_camera`, `right_camera`, `third_person`, and `mosaic`.
With `WANDB_API_KEY`, the worker logs trial completion, success rate, last
trial fields, and the five MP4 views.

Use `--prepare-only` with staged roots to validate wiring without starting
either GPU process. The worker always terminates the server on exit, prints
the final policy/evaluator logs on failure, and surfaces policy exceptions
through the patched client.
