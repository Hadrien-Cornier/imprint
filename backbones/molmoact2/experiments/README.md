# MolmoAct2 Experiments

> **IMPRINT release note:** this is the preserved native MolmoAct2 source
> snapshot used by IMPRINT. Start with
> [`../README.md`](../README.md) for the supported public Stage-2 and
> evaluation workflow. Release models and datasets are resolved from pinned
> Hugging Face repositories; private storage locations are not required.

This directory contains the open-sourced MolmoAct2 training and evaluation code used to replicate our experiments and fine-tune on new LeRobot datasets.

The experiments package is organized for release use: training commands are
written directly below, infrastructure-specific job specifications are not
included, and deployment documentation lives in the vendored LeRobot docs.

## Setup

From an IMPRINT checkout, install this source snapshot in editable mode:

```bash
cd backbones/molmoact2/experiments
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python -m pip install -e "./lerobot"
```

Install additional LeRobot simulator or hardware extras only when required by
the target environment.

Configure local paths for your machine:

```bash
export LEROBOT_DATA_ROOT=/path/to/lerobot/data
export LEROBOT_DEPTH_DATA_ROOT=/path/to/lerobot/depth_data
export MOLMO_DATA_DIR=/path/to/molmo/data
export SPATIAL_DATA_HOME=/path/to/spatial/embodied_training_data
export HF_HOME=/path/to/huggingface/cache
export LEROBOT_VIDEO_BACKEND=pyav
export PYTHONPATH="$PWD:$PWD/lerobot/src:${PYTHONPATH:-}"
```

Set credentials through environment variables:

```bash
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
```

### Portable Bootstrap

For a fresh Linux GPU host, clone the public branch and use the bootstrap
helper:

```bash
git clone --branch molmoact2 https://github.com/amazon-far/imprint.git
cd imprint
export MOLMOACT2_CHECKOUT_DIR="${MOLMOACT2_CHECKOUT_DIR:-$PWD/.external/molmoact2}"
export MOLMOACT2_REVISION=448b3b6916d62a35d2088b1e2425d5d7626c8e03
bash backbones/molmoact2/experiments/scripts/bootstrap_molmoact2.sh \
  --revision "$MOLMOACT2_REVISION"
```

The bootstrap verifies GitHub access before cloning, initializes submodules,
downloads the Stage-1 checkpoint from `Amazon-FAR/imprint_models`, path
`simulation/rmbench/full_attention/stage1/stage1_step160000.npz`, and checks
its SHA-256. For a
user-provided evaluation runtime, set
`RMBENCH_RUNTIME_ARCHIVE_DIR` and add `--install-eval-runtime`; passwordless
`sudo` is required.

The helper performs a `git ls-remote` preflight. Configure Git authentication
using the standard mechanism for your environment before running it.

## RMBench Stage-2 Runtime

The memory-enabled RMBench rollout worker requires the pinned Linux simulator
runtime used for Stage-2 evaluation. IMPRINT does not redistribute that
root-relative runtime because it contains RMBench and third-party simulator
components. Install the benchmark environment from its authorized sources.
If you already have the verified runtime archives, place `runtime.tar`,
`uv-python.tar`, and `runtime-manifest.json` in one local directory:

```bash
export RMBENCH_RUNTIME_ARCHIVE_DIR=/path/to/rmbench-runtime-archives
sudo -E backbones/molmoact2/experiments/scripts/install_rmbench_molmoact2_runtime.sh
```

The installer extracts the pinned Python and simulator archives into `/`. To
stage elsewhere for inspection, set `RMBENCH_RUNTIME_EXTRACT_ROOT` and
`RMBENCH_RUNTIME_STAGE_DIR`. The rollout worker assumes the runtime is already
installed by default.

## RMBench Stage-1 Memory Pretraining

The `dino_build` branch also tracks the complete Stage-1 `mme_vla_suite`
source under [`stage1/`](stage1/). Its offline trainer preserves the legacy
SigLIP input path and adds the DINOv3 reconstruction path used by the
published RMBench Stage-1 run. See [the Stage-1 guide](stage1/README.md) for
the required feature-store contract and exact DINO smoke command.

## RMBench MolmoAct2 Base Checkpoint Cache

RMBench Stage-2 training starts from `allenai/MolmoAct2`. Its model
configuration resolves the tokenizer from the separate `Qwen/Qwen3-4B`
repository. Download both upstream repositories at their exact revisions into
the shared Hugging Face cache:

```bash
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_MODULES_CACHE="$HF_HOME/modules"
hf download allenai/MolmoAct2 \
  --revision e432d85f6e039edca44afb93c262f3084ab72a9c \
  --cache-dir "$HF_HOME/hub"
hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --cache-dir "$HF_HOME/hub"
```

After both downloads complete, offline launches may set:

```bash
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

This makes missing or malformed cache state fail immediately instead of
silently contacting Hugging Face.

## RMBench Stage-2 Evaluation

[`rmbench_eval/`](rmbench_eval/) is the source-controlled closed-loop
evaluation package. It owns the policy-server/client transport, four-view
overlay, direct worker launcher, and the training-to-evaluation parity
contract. A separately installed RMBench runtime supplies the canonical
simulator.

See the [RMBench Stage-2 evaluation guide](rmbench_eval/README.md) for bundle
preparation, single- and two-GPU commands, Hugging Face artifacts, and the
required online DINOv3/Wan, stored Pi0.5 language, camera/RGB-to-BGR, action,
and `EVAL_CONSOLIDATE_ALL=1` contracts.

## Checkpoints

`launch_scripts/train_lerobot.py` accepts local checkpoint paths, URLs, and Hugging Face model IDs.

| Stage | Start checkpoint |
| --- | --- |
| ER training | `https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar` |
| MolmoAct2 pretraining | `allenai/Molmo2-ER` |
| MolmoAct2 post-training | `allenai/MolmoAct2-Pretrain` |
| Standard fine-tuning | `allenai/MolmoAct2` |
| Depth fine-tuning | `allenai/MolmoAct2-Think` |
| Bimanual YAM fine-tuning | `allenai/MolmoAct2-BimanualYAM` |
| DROID fine-tuning | `allenai/MolmoAct2-DROID` |
| SO100/SO101 fine-tuning | `allenai/MolmoAct2-SO100_101` |

## Depth Annotations

Generate depth companion datasets before depth post-training or depth fine-tuning:

```bash
python scripts/generate_depth_annotation.py \
  "${LEROBOT_DATA_ROOT}/<repo_id>" \
  --camera-key observation.images.primary
```

When `LEROBOT_DEPTH_DATA_ROOT` is set, the generator writes to `${LEROBOT_DEPTH_DATA_ROOT}/<repo_id>` by default, matching the training lookup. It writes `buffer_codes` with 100 entries by default, matching training's default `--num_depth_tokens_per_image 100`.

## Fine-Tuning A New Dataset

The primary use case for this repo is adapting MolmoAct2 to a new LeRobot dataset. Add a mixture in `launch_scripts/data_mixtures.py`, then train with full fine-tuning, LoRA, or action-expert-only tuning.

### Register A Mixture

For a single LeRobot dataset, use `build_single_lerobot_mixture` and register the builder in `MOLMOACT2_LEROBOT_MIXTURES`. `libero_goal` is the minimal built-in example:

```python
def build_molmoact2_libero_goal():
    return build_single_lerobot_mixture(
        name="libero_goal",
        tag="libero",
        repo_ids=["allenai/MolmoAct2-LIBERO-Dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


MOLMOACT2_LEROBOT_MIXTURES["libero_goal"] = build_molmoact2_libero_goal
```

For a new embodiment, create a new tag and describe the robot/action convention literally:

```python
def build_molmoact2_my_robot():
    return build_single_lerobot_mixture(
        name="my_robot",
        tag="my_robot",
        repo_ids=["my-org/my-lerobot-dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.front",
            "observation.images.wrist",
        ],
        normalize_gripper=False,
        setup_type="single robot arm in my workspace",
        control_mode="delta end-effector pose",
        action_horizon=30,
        n_action_steps=30,
    )


MOLMOACT2_LEROBOT_MIXTURES["my_robot"] = build_molmoact2_my_robot
```

Use the same `tag` only for datasets that share robot semantics, action/state normalization, camera conventions, `setup_type`, and `control_mode`.

### Smoke Test

For a quick single-dataset validation, disable packing and use dynamic sequence lengths:

```bash
export EXP_NAME="molmoact2-my-robot-smoke"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=20 \
  --device_batch_size=1 \
  --global_batch_size=1 \
  --num_workers=0 --pin_memory=false \
  --save_folder="checkpoints/smoke/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none
```

### Full Fine-Tuning

Full fine-tuning updates the VLM, vision tower, connector, LM head, and action expert. Use it for larger datasets or substantial embodiment changes.

```bash
export EXP_NAME="molmoact2-my-robot-fft"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=false \
  --llm_learning_rate=1e-5 \
  --vit_learning_rate=5e-6 \
  --connector_learning_rate=5e-6 \
  --action_expert_learning_rate=5e-5
```

### LoRA Fine-Tuning

LoRA fine-tuning updates LoRA adapters on the VLM path and fully trains the action expert. Use it for smaller datasets or similar embodiments.

```bash
export EXP_NAME="molmoact2-my-robot-lora"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=true \
  --lora_rank=64 \
  --llm_learning_rate=5e-5 \
  --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 \
  --action_expert_learning_rate=5e-5
```

### Action-Expert-Only Fine-Tuning

Action-expert-only fine-tuning freezes the VLM, vision tower, connector, embeddings, and LM head. Use it when the vision-language behavior should remain fixed and the new dataset mainly changes the continuous control head.

```bash
export EXP_NAME="molmoact2-my-robot-ae-only"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --action_expert_learning_rate=5e-5
```

### Stage-2 Memory Bottleneck

Stage-2 memory conditioning is available in native MolmoAct2 training and
local-checkpoint inference. Stage-1 feature production is unchanged: each
LeRobot row must already contain either the raw Stage-1 streams or cached
memory tokens. Memory only conditions the action expert; it is never appended
to the VLM token context.

#### RMBench Compact Store

Use the `rmbench_stage2` mixture for the compact RMBench store. It requires
the Stage-2 sample store and the separate Stage-1 feature store; pass both
paths explicitly so worker processes reopen the same data after serialization:

```bash
export RMBENCH_STAGE2_DATA_ROOT=/path/to/compact-rmbench-stage2
export RMBENCH_STAGE1_FEATURE_ROOT=/path/to/stage1-dinov3-features

torchrun --standalone --nproc-per-node=4 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 rmbench_stage2 \
  --rmbench_stage2_data_root="$RMBENCH_STAGE2_DATA_ROOT" \
  --rmbench_stage1_feature_root="$RMBENCH_STAGE1_FEATURE_ROOT" \
  --memory_enabled=true \
  --memory_mode=frozen \
  --memory_feature_schema=pi05_dinov3 \
  --packing=false \
  --ft_vlm=false \
  --ft_action_expert=true
```

`rmbench_stage2` is a 16-D, three-camera, 50-step action mixture. The
evaluation bundle intentionally contains only Stage-2 metadata, stored
language embeddings, DINOv3, and the selected checkpoint; it is not a
substitute for the full `stage2/data` training store.

Raw encoder modes require these fields:

```text
cosmo_latent_emb             [L, 48]
cosmo_recent_emb             [R, 768]
cosmo_action_emb             [R, 16]
cosmo_proprio_emb            [R, 16]
cosmo_mem_mask               [L] bool
cosmo_recent_mask            [R] bool
cosmo_action_mask            [R] bool
cosmo_proprio_mask           [R] bool
cosmo_lang_emb               [2048]
```

The default `pi05_dinov3` schema maps these external fields to the stable
internal `memory_*` model arguments. Its recent stream is DINOv3
`dinov3.npy`, flattened in time-major order from `[T, 3, 64, 768]` to
`[R, 768]`. Use `--memory_feature_schema=canonical` when a dataset already
uses the `memory_*` field names; `--memory_feature_prefix` applies only to
that canonical schema.

Cached mode requires `memory_tokens [M, 1024]` and
`memory_attention_mask [M] bool`. The launcher validates the complete feature
schema before workers are created.

#### Stage-1 Checkpoint Archive

The exact Stage-1 initialization used by the published 100k configuration is
included with the RMBench full-attention release:

```bash
export MEMORY_STAGE1_CHECKPOINT="${PWD}/checkpoints/stage1_step160000.npz"
mkdir -p "$(dirname "$MEMORY_STAGE1_CHECKPOINT")"
downloaded_checkpoint="$(
  hf download Amazon-FAR/imprint_models \
    simulation/rmbench/full_attention/stage1/stage1_step160000.npz \
    --revision main \
    --quiet
)"
cp "$downloaded_checkpoint" "$MEMORY_STAGE1_CHECKPOINT"
echo "164c94f3ff8fcd84aee61e88ec0feb97faf99d203eac1be0f4b4146d464d2044  $MEMORY_STAGE1_CHECKPOINT" | sha256sum --check
```

The file is 331,937,012 bytes. The loader requires a local path, so pass
`MEMORY_STAGE1_CHECKPOINT` to `--memory_stage1_checkpoint`.

Freeze the imported Stage-1 encoder while training the action expert:

```bash
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 memory_test \
  --memory_enabled=true \
  --memory_mode=frozen \
  --memory_stage1_checkpoint="$MEMORY_STAGE1_CHECKPOINT" \
  --memory_feature_schema=pi05_dinov3 \
  --packing=false \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none
```

For joint Stage-1/Stage-2 fine-tuning, set `--memory_mode=finetune`, keep
`--ft_action_expert=true`, and use a continuous action format; the launcher
rejects a configuration that would freeze the encoder. To match the DINOv3
Pi0.5 Stage-2 run, fine-tune the language model, connector, action expert,
and memory encoder while keeping the vision encoder frozen:

```bash
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 memory_test \
  --memory_enabled=true \
  --memory_mode=finetune \
  --memory_stage1_checkpoint="$MEMORY_STAGE1_CHECKPOINT" \
  --memory_feature_schema=pi05_dinov3 \
  --packing=false \
  --ft_action_expert=true \
  --ft_vlm=true \
  --ft_llm=true \
  --ft_connector=true \
  --ft_vit=false \
  --ft_embedding=all \
  --llm_learning_rate=2.5e-5 \
  --connector_learning_rate=2.5e-5 \
  --action_expert_learning_rate=2.5e-5
```

`--ft_llm`, `--ft_connector`, and `--ft_vit` override `--ft_vlm` only when
specified, so existing launcher commands retain their previous behavior. For
cached encoder output, omit the Stage-1 checkpoint:

```bash
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 memory_test \
  --memory_enabled=true \
  --memory_mode=precomputed \
  --packing=false \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none
```

Saved Stage-2 checkpoints include the PyTorch memory encoder and FiLM seam,
so local checkpoint resume and native local-policy inference do not need the
original Flax `.npz` again. The released DROID and YAM Hugging Face
checkpoints use separate remote-code implementations and are not
memory-conditioned checkpoints.

### Depth Fine-Tuning

For depth-reasoning fine-tuning, start from `allenai/MolmoAct2-Think` and add:

```bash
--enable_depth_reasoning=true \
--num_depth_tokens=128 \
--num_depth_tokens_per_image=100 \
--depth_code_input_noise_rate=0.1 \
--style_robot_action=1.0 \
--style_robot_depth=0.0 \
--style_robot_depth_action=1.0
```

## Application

The vendored `lerobot` package is the supported inference and deployment path for MolmoAct2. It covers simulator rollout, real-robot rollout/recording, async inference, CUDA graph inference, and the MolmoAct2 policy configuration surface.

See `lerobot/docs/source/molmoact2.mdx` for deployment commands and LeRobot-specific details.

## Reproducing Released Training Stages

The commands below reproduce the major MolmoAct2 training stages. They are references for release replication; for new projects, start with the new-dataset fine-tuning section above.

### Molmo2-ER Training

Molmo2-ER starts from the public Molmo2-4B checkpoint archive, not from a Hugging Face model. The launcher downloads and extracts this archive into `${MOLMOACT2_CHECKPOINT_CACHE:-${HF_HOME}/molmoact2/checkpoints}` before loading it.

```bash
export EXP_NAME="molmo2-er"
export SPATIAL_DATA_HOME="${SPATIAL_DATA_HOME:-/path/to/spatial/embodied_training_data}"

WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=2 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/sft.py \
  https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar \
  molmo2_embodied_spatial_mix_50 \
  --seq_len=16384 \
  --device_batch_size=1 \
  --max_duration=20000 \
  --save_interval=1000 \
  --save_num_checkpoints_to_keep=20 \
  --num_workers=2 \
  --data.persistent_workers=true \
  --save_folder="checkpoints/er/${EXP_NAME}" \
  --wandb.name="${EXP_NAME}" \
  --wandb.entity=<wandb-entity> \
  --wandb.project=<wandb-project> \
  --model.vision_backbone.compile_connector=null \
  --model.mm_preprocessor.max_frames=128 \
  --model.mm_preprocessor.use_frame_special_tokens=true \
  --model.mm_preprocessor.max_subtitle_tokens=null
```

Use `spatial-all-v4` for the spatial-only stage, or `molmo2_embodied_spatial_mix_30`, `molmo2_embodied_spatial_mix_50`, `molmo2_embodied_spatial_mix_70`, and `molmo2_embodied_spatial_mix_90` for general/spatial recovery ablations.

### MolmoAct2 Pretraining

```bash
export EXP_NAME="molmoact2-pretrain"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=8 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/Molmo2-ER \
  pre_post_train \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=4200 \
  --max_duration=200000 \
  --device_batch_size=2 \
  --global_batch_size=128 \
  --log_interval=20 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/pretrain/${EXP_NAME}" \
  --packing=true \
  --crop_mode=resize \
  --ft_embedding=added_tokens \
  --add_action_expert=false \
  --action_format=discrete \
  --ft_vlm=true \
  --connector_learning_rate=5e-6 \
  --vit_learning_rate=5e-6 \
  --llm_learning_rate=1e-5 \
  --random_camera_order=episode \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=true \
  --sample_annotated_task=false
```

### Post-Training

```bash
export EXP_NAME="molmoact2-posttrain"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=8 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-Pretrain \
  pre_post_train \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 --vlm_seq_len=4200 \
  --max_duration=100000 \
  --device_batch_size=2 \
  --global_batch_size=128 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/posttrain/${EXP_NAME}" \
  --separate_vlm_dataloader=true \
  --packing=true \
  --pad_packed_action_chunks=true \
  --packed_action_chunk_cap=5 \
  --crop_mode=resize \
  --ft_embedding=added_tokens \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=4 \
  --mask_action_dim_padding=true \
  --random_camera_order=episode \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=true \
  --sample_annotated_task=false
```

Depth post-training:

```bash
export EXP_NAME="molmoact2-posttrain-depth"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=8 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-Pretrain \
  pre_post_train \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 --vlm_seq_len=4200 \
  --max_duration=100000 \
  --device_batch_size=2 \
  --global_batch_size=128 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/posttrain/${EXP_NAME}" \
  --separate_vlm_dataloader=true \
  --skip_missing_vlm_examples=true \
  --packing=true \
  --pad_packed_action_chunks=true \
  --packed_action_chunk_cap=5 \
  --crop_mode=resize \
  --ft_embedding=added_tokens \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=4 \
  --mask_action_dim_padding=true \
  --random_camera_order=episode \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=true \
  --sample_annotated_task=false \
  --num_depth_tokens=128 \
  --enable_depth_reasoning=true \
  --num_depth_tokens_per_image=100 \
  --style_robot_action=1.0 \
  --style_robot_depth=1.0 \
  --style_robot_depth_action=1.0
```

### Fine-Tuning Existing Mixtures

Use the closest released checkpoint for the target embodiment:

- General LIBERO or Franka tabletop tasks: `allenai/MolmoAct2`
- Depth reasoning: `allenai/MolmoAct2-Think`
- DROID: `allenai/MolmoAct2-DROID`
- Bimanual YAM: `allenai/MolmoAct2-BimanualYAM`
- SO100/SO101: `allenai/MolmoAct2-SO100_101`

DROID full fine-tuning:

```bash
export EXP_NAME="molmoact2-droid"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=4 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-DROID \
  droid \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=true \
  --crop_mode=resize \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=8 \
  --mask_action_dim_padding=true \
  --random_camera_order=none \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=false \
  --sample_annotated_task=false
```

Depth fine-tuning uses the same mixture names and `--enable_depth_reasoning=true`:

```bash
export EXP_NAME="molmoact2-droid-depth"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=4 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-Think \
  droid \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=true \
  --crop_mode=resize \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=8 \
  --mask_action_dim_padding=true \
  --random_camera_order=none \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=false \
  --sample_annotated_task=false \
  --num_depth_tokens=128 \
  --enable_depth_reasoning=true \
  --num_depth_tokens_per_image=100 \
  --depth_code_input_noise_rate=0.1 \
  --style_robot_action=1.0 \
  --style_robot_depth=0.0 \
  --style_robot_depth_action=1.0
```
