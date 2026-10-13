#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=configure_training_runtime.sh
source "$ROOT/scripts/reproduce/configure_training_runtime.sh"
CONFIG_FILE="${1:-}"
[[ -f "$CONFIG_FILE" ]] || {
  echo "usage: $0 configs/paper/<benchmark>/<architecture>/experiment.env" >&2
  exit 2
}
# shellcheck disable=SC1090
source "$CONFIG_FILE"

STORE="${IMPRINT_STAGE2_STORE:-${IMPRINT_STORE:-}}"
FEATURE_STORE="${IMPRINT_STAGE1_STORE:-${IMPRINT_FEATURE_STORE:-$STORE}}"
STAGE1="${IMPRINT_STAGE1_CHECKPOINT:-}"
PI05="${IMPRINT_PI05_PARAMS:-}"
OUTPUT="${IMPRINT_OUTPUT_DIR:-}"
PYTHON="${IMPRINT_PYTHON:-$ROOT/.venv/bin/python}"
SOURCE_HISTORY="$ROOT/src/mme_vla_suite/models/config/robomme/$IMPRINT_HISTORY_CONFIG"
EXP_NAME="${IMPRINT_EXP_NAME:-imprint-${IMPRINT_BENCHMARK}-${IMPRINT_ARCHITECTURE}-stage2}"
STAGE2_STEPS="$IMPRINT_STAGE2_STEPS"
STAGE2_WARMUP_STEPS="${IMPRINT_STAGE2_WARMUP_STEPS:-500}"

: "${STORE:?set IMPRINT_STAGE2_STORE or IMPRINT_STORE}"
: "${STAGE1:?set IMPRINT_STAGE1_CHECKPOINT}"
: "${PI05:?set IMPRINT_PI05_PARAMS}"
: "${OUTPUT:?set IMPRINT_OUTPUT_DIR}"
[[ "$STAGE2_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "IMPRINT_STAGE2_STEPS must be a positive integer" >&2
  exit 2
}
[[ "$STAGE2_WARMUP_STEPS" =~ ^[0-9]+$ ]] || {
  echo "IMPRINT_STAGE2_WARMUP_STEPS must be a non-negative integer" >&2
  exit 2
}
if (( STAGE2_WARMUP_STEPS >= STAGE2_STEPS )); then
  STAGE2_WARMUP_STEPS=$((STAGE2_STEPS / 2))
  echo "Clamped Stage-2 warm-up to ${STAGE2_WARMUP_STEPS} for ${STAGE2_STEPS} training steps"
fi
test -x "$PYTHON"
test -d "$STORE"
test -f "$STAGE1"
test -d "$PI05"
test -f "$SOURCE_HISTORY"

CHECKPOINT_DIR="$OUTPUT/$IMPRINT_STAGE2_CONFIG/$EXP_NAME"
if [[ -e "$CHECKPOINT_DIR" ]] && [[ -n "$(find "$CHECKPOINT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite nonempty checkpoint directory: $CHECKPOINT_DIR" >&2
  exit 3
fi

RUNTIME_HISTORY="$OUTPUT/config/$IMPRINT_HISTORY_CONFIG"
prepare_args=(
  --source "$SOURCE_HISTORY"
  --output "$RUNTIME_HISTORY"
  --latent-cap "$IMPRINT_LATENT_CAP"
  --max-recent-tokens "$IMPRINT_MAX_RECENT_TOKENS"
  --max-step-tokens "$IMPRINT_MAX_STEP_TOKENS"
  --num-latents "$IMPRINT_NUM_LATENTS"
)
if [[ "$IMPRINT_BENCHMARK" == "rmbench" ]]; then
  prepare_args+=(--feature-store "$FEATURE_STORE")
fi
"$PYTHON" "$ROOT/scripts/reproduce/prepare_history_config.py" "${prepare_args[@]}"

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OUTPUT/.openpi-cache}"
imprint_configure_training_runtime
export IMPRINT_PI05_PARAMS="$PI05"
export IMPRINT_RMBENCH_STORE="$STORE"
export IMPRINT_ROBOCASA_STORE="$STORE"
export IMPRINT_RMBENCH_ATTENTION_STAGE1="$STAGE1"
export IMPRINT_RMBENCH_RECURRENT_STAGE1="$STAGE1"
export IMPRINT_ROBOCASA_ATTENTION_STAGE1="$STAGE1"
export IMPRINT_ROBOCASA_RECURRENT_STAGE1="$STAGE1"

args=(
  "$IMPRINT_STAGE2_CONFIG"
  --exp-name="$EXP_NAME"
  --dataset-path="$STORE"
  --model.history-config="$RUNTIME_HISTORY"
  --model.action-horizon="$IMPRINT_ACTION_HORIZON"
  --weight-loader.params-path="$PI05"
  --weight-loader.stage1-ckpt="$STAGE1"
  --batch-size="$IMPRINT_BATCH_SIZE"
  --num-train-steps="$STAGE2_STEPS"
  --lr-schedule.warmup-steps="$STAGE2_WARMUP_STEPS"
  --lr-schedule.decay-steps="$STAGE2_STEPS"
  --fsdp-devices="$IMPRINT_STAGE2_FSDP_DEVICES"
  --num-workers="${IMPRINT_NUM_WORKERS:-4}"
  --save-interval=10000
  --keep-period=10000
  --checkpoint-base-dir="$OUTPUT"
  --seed=42
  --ema-decay=0.999
)
if [[ "${IMPRINT_WANDB_ENABLED:-1}" != "1" ]]; then
  args+=(--no-wandb-enabled)
fi

cd "$ROOT"
exec "$PYTHON" -u scripts/train_mme.py "${args[@]}"
