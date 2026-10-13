#!/usr/bin/env bash
set -euo pipefail

# Launch one exact four-GPU Stage-2 DINOv3 memory-seam ablation.
#
# Required environment:
#   RUN_VARIANT={native,native_nonzero_film,pi05}
#   STAGE1_CKPT=/path/to/stage1_step160000.npz
#   STAGE2_ROOT=/path/to/rmbench/stage2
#   STAGE1_FEATURE_ROOT=/path/to/rmbench/stage1/features

: "${RUN_VARIANT:?Set RUN_VARIANT to native, native_nonzero_film, or pi05.}"
: "${STAGE1_CKPT:?Set STAGE1_CKPT to the Stage-1 .npz checkpoint.}"
: "${STAGE2_ROOT:?Set STAGE2_ROOT to the RMBench Stage-2 store.}"
: "${STAGE1_FEATURE_ROOT:?Set STAGE1_FEATURE_ROOT to the Stage-1 features directory.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MASTER_PORT="${MASTER_PORT:-29500}"
BASE_MODEL="${BASE_MODEL:-allenai/MolmoAct2}"
MAX_DURATION="${MAX_DURATION:-100000}"
MAX_ACTION_HORIZON="${MAX_ACTION_HORIZON:-50}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-16}"
MEMORY_NUM_LATENTS="${MEMORY_NUM_LATENTS:-256}"
MEMORY_DIM="${MEMORY_DIM:-1024}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"

case "${RUN_VARIANT}" in
    native)
        RUN_NAME="${RUN_NAME:-imprint-molmoact2-rmbench-native-memory-100k}"
        SEAM_ARGS=(--memory_seam_mode=native --memory_film_identity_init=true)
        ;;
    native_nonzero_film)
        RUN_NAME="${RUN_NAME:-imprint-molmoact2-rmbench-native-nonzero-film-100k}"
        SEAM_ARGS=(--memory_seam_mode=native --memory_film_identity_init=false --memory_film_init_std=0.002)
        ;;
    pi05)
        RUN_NAME="${RUN_NAME:-imprint-molmoact2-rmbench-pi05-seam-100k}"
        SEAM_ARGS=(--memory_seam_mode=pi05 --memory_film_init_std=0.002)
        ;;
    *)
        echo "RUN_VARIANT must be native, native_nonzero_film, or pi05, got '${RUN_VARIANT}'" >&2
        exit 2
        ;;
esac

SAVE_FOLDER="${SAVE_FOLDER:-${SOURCE_ROOT}/outputs/${RUN_NAME}}"
LAYOUT="${STAGE2_ROOT}/meta/memory_layout.json"
STAGE1_META="${STAGE1_FEATURE_ROOT%/features}/meta/stage1_compact.json"

[[ -d "${SOURCE_ROOT}/experiments" ]]
[[ -x "$(command -v "${PYTHON_BIN}")" || -x "${PYTHON_BIN}" ]]
[[ -f "${STAGE1_CKPT}" ]]
[[ -f "${LAYOUT}" ]]
[[ -f "${STAGE1_META}" ]]

require_layout_value() {
    local file="$1"
    local key="$2"
    local value="$3"
    grep -Eq "\"${key}\"[[:space:]]*:[[:space:]]*${value}([,[:space:]]|$)" "${file}"
}

# Enforce the Stage-1 DINOv3 feature contract at launch, rather than silently
# training with an incompatible memory store.
for file in "${LAYOUT}" "${STAGE1_META}"; do
    require_layout_value "${file}" "max_recent_tokens" 768
    require_layout_value "${file}" "max_step_tokens" 1244
done
require_layout_value "${LAYOUT}" "latent_cap" 4096
require_layout_value "${STAGE1_META}" "max_pooled_latent_tokens" 3720
require_layout_value "${LAYOUT}" "latent_dim" 48
require_layout_value "${LAYOUT}" "recent_dim" 768
require_layout_value "${LAYOUT}" "stream_dim" 16

export WANDB_PROJECT="${WANDB_PROJECT:-Robocasa}"
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-${STAGE2_ROOT}}"
export PYTHONPATH="${SOURCE_ROOT}/experiments:${SOURCE_ROOT}/experiments/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${SOURCE_ROOT}/experiments"
exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    launch_scripts/train_lerobot.py \
    "${BASE_MODEL}" \
    rmbench_stage2 \
    --frame_loading_backend=av \
    --memory_enabled=true \
    --memory_mode=finetune \
    --memory_stage1_checkpoint="${STAGE1_CKPT}" \
    --memory_feature_schema=pi05_dinov3 \
    --memory_num_latents="${MEMORY_NUM_LATENTS}" \
    --memory_dim="${MEMORY_DIM}" \
    --rmbench_stage2_data_root="${STAGE2_ROOT}" \
    --rmbench_stage1_feature_root="${STAGE1_FEATURE_ROOT}" \
    --max_action_horizon="${MAX_ACTION_HORIZON}" \
    --max_action_dim="${MAX_ACTION_DIM}" \
    --ft_action_expert=true \
    --ft_vlm=true \
    --ft_llm=true \
    --ft_connector=true \
    --ft_vit=false \
    --ft_embedding=lm_head \
    --llm_learning_rate=2.5e-5 \
    --connector_learning_rate=2.5e-5 \
    --action_expert_learning_rate=2.5e-5 \
    "${SEAM_ARGS[@]}" \
    --device_batch_size="${DEVICE_BATCH_SIZE}" \
    --global_batch_size="${GLOBAL_BATCH_SIZE}" \
    --num_workers=0 \
    --dynamic_seq_len=true \
    --max_duration="${MAX_DURATION}" \
    --run_name="${RUN_NAME}" \
    --save_folder="${SAVE_FOLDER}" \
    --save_final_unsharded_checkpoint=true \
    --save_final_optim=false \
    --save_overwrite=false
