#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            printf 'Required environment variable %s is unset or empty.\n' "$name" >&2
            exit 2
        fi
    done
}

require_dir() {
    local name="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        printf '%s must be an existing directory: %s\n' "$name" "$path" >&2
        exit 2
    fi
}

require_file() {
    local name="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        printf '%s must be an existing file: %s\n' "$name" "$path" >&2
        exit 2
    fi
}

reject_offline_visual_environment() {
    local name value upper_name
    while IFS='=' read -r name value; do
        upper_name="${name^^}"
        case "$upper_name" in
            *OFFLINE*FEATURE*|*OFFLINE*VISUAL*|*OFFLINE*LATENT*)
                if [[ -n "$value" ]]; then
                    printf 'Offline visual input is forbidden: %s is set.\n' "$name" >&2
                    exit 2
                fi
                ;;
        esac
    done < <(env)
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_env \
    MOLMOACT2_CHECKPOINT \
    RMBENCH_STAGE2_ROOT \
    RMBENCH_STAGE1_FEATURE_ROOT \
    RMBENCH_DINOV3_MODEL_PATH \
    RMBENCH_PI05_SOURCE_ROOT \
    RMBENCH_TASK \
    RMBENCH_POLICY_MODEL_SERVER
reject_offline_visual_environment

require_dir MOLMOACT2_CHECKPOINT "$MOLMOACT2_CHECKPOINT"
require_dir RMBENCH_STAGE2_ROOT "$RMBENCH_STAGE2_ROOT"
require_dir RMBENCH_STAGE1_FEATURE_ROOT "$RMBENCH_STAGE1_FEATURE_ROOT"
require_dir RMBENCH_DINOV3_MODEL_PATH "$RMBENCH_DINOV3_MODEL_PATH"
require_dir RMBENCH_PI05_SOURCE_ROOT "$RMBENCH_PI05_SOURCE_ROOT"

checkpoint_metadata="${MOLMOACT2_CHECKPOINT}/config.json"
dino_config="${RMBENCH_DINOV3_MODEL_PATH}/config.json"
stage2_episodes="${RMBENCH_STAGE2_ROOT}/meta/episodes.jsonl"
memory_layout="${RMBENCH_STAGE2_ROOT}/meta/memory_layout.json"
require_file checkpoint_metadata "$checkpoint_metadata"
require_file dino_config "$dino_config"
require_file stage2_episodes "$stage2_episodes"
require_file memory_layout "$memory_layout"

export MEMFT_DINOV3_REQUIRED=1
export EVAL_CONSOLIDATE_ALL=1
export RMBENCH_CAMERA_ORDER=head,left,right
export RMBENCH_PREDICTION_HORIZON=50
export RMBENCH_EXECUTION_HORIZON=50

provenance_path="${RMBENCH_PROVENANCE_PATH:-rmbench_molmoact2_eval_provenance.json}"
checkpoint_metadata_sha256="$(sha256_file "$checkpoint_metadata")"
dino_config_sha256="$(sha256_file "$dino_config")"
stage2_episodes_sha256="$(sha256_file "$stage2_episodes")"
memory_layout_sha256="$(sha256_file "$memory_layout")"
task_sha256="$(printf '%s' "$RMBENCH_TASK" | sha256sum | awk '{print $1}')"

python - "$provenance_path" \
    "$checkpoint_metadata" "$checkpoint_metadata_sha256" \
    "$dino_config" "$dino_config_sha256" \
    "$stage2_episodes" "$stage2_episodes_sha256" \
    "$memory_layout" "$memory_layout_sha256" \
    "$RMBENCH_TASK" "$task_sha256" <<'PY'
import json
import sys
from pathlib import Path

output, *values = sys.argv[1:]
artifacts = {
    "checkpoint_metadata": values[0:2],
    "dino_config": values[2:4],
    "stage2_episodes": values[4:6],
    "memory_layout": values[6:8],
    "task": values[8:10],
}
Path(output).write_text(
    json.dumps(
        {
            name: {"path": path, "sha256": sha256}
            for name, (path, sha256) in artifacts.items()
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

exec "$RMBENCH_POLICY_MODEL_SERVER" \
    --policy-model scripts.serve_rmbench_molmoact2 \
    --checkpoint "$MOLMOACT2_CHECKPOINT" \
    --stage2-root "$RMBENCH_STAGE2_ROOT" \
    --feature-root "$RMBENCH_STAGE1_FEATURE_ROOT" \
    --dino-model-path "$RMBENCH_DINOV3_MODEL_PATH" \
    --stage1-source-root "$RMBENCH_PI05_SOURCE_ROOT" \
    --prediction-horizon "$RMBENCH_PREDICTION_HORIZON" \
    --execution-horizon "$RMBENCH_EXECUTION_HORIZON" \
    --task "$RMBENCH_TASK"
