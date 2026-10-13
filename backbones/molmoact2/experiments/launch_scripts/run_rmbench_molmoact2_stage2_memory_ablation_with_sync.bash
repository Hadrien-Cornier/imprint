#!/usr/bin/env bash
set -euo pipefail

# Run a Stage-2 memory-seam ablation and publish the final model-only checkpoint.
#
# Required environment is inherited from
# launch_rmbench_molmoact2_stage2_memory_ablation.bash. RUN_HF_REPO may be set
# explicitly; otherwise it is deterministically derived from RUN_VARIANT.

: "${RUN_VARIANT:?Set RUN_VARIANT before invoking this wrapper.}"

case "${RUN_VARIANT}" in
    native)
        RUN_NAME="molmoact2-stage2-dinov3-memory-native-fixed-film-100k-pi05-ablation"
        ;;
    native_nonzero_film)
        RUN_NAME="molmoact2-stage2-dinov3-memory-native-nonzero-film-100k-pi05-ablation"
        ;;
    pi05)
        RUN_NAME="molmoact2-stage2-dinov3-memory-pi05-seam-100k-ablation"
        ;;
    *)
        echo "Unsupported RUN_VARIANT '${RUN_VARIANT}'." >&2
        exit 2
        ;;
esac

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SAVE_FOLDER="${SAVE_FOLDER:-${SOURCE_ROOT}/outputs/${RUN_NAME}}"
RUN_HF_REPO="${RUN_HF_REPO:-Amazon-FAR/${RUN_NAME}}"
UPLOAD_WORKERS="${IMPRINT_UPLOAD_WORKERS:-16}"
LAUNCHER="$(dirname "$0")/launch_rmbench_molmoact2_stage2_memory_ablation.bash"

bash "${LAUNCHER}"

command -v hf >/dev/null || {
    echo "Hugging Face CLI 'hf' is required to publish the final checkpoint." >&2
    exit 127
}

FINAL_CHECKPOINT="${SAVE_FOLDER}/step${MAX_DURATION:-100000}-unsharded"
[[ -f "${FINAL_CHECKPOINT}/model.pt" ]] || {
    echo "Final model-only checkpoint is missing: ${FINAL_CHECKPOINT}/model.pt" >&2
    exit 2
}
[[ -f "${FINAL_CHECKPOINT}/config.yaml" ]] || {
    echo "Final checkpoint config is missing: ${FINAL_CHECKPOINT}/config.yaml" >&2
    exit 2
}

hf upload-large-folder "$RUN_HF_REPO" "$FINAL_CHECKPOINT" \
    --exclude "optim.pt" "train/**" "*.tmp" "*.lock" \
    --num-workers "$UPLOAD_WORKERS"
printf 'Published model-only checkpoint to https://huggingface.co/%s\n' "$RUN_HF_REPO"
