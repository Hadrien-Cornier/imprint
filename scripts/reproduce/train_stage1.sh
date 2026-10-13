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

STORE="${IMPRINT_STAGE1_STORE:-${IMPRINT_STORE:-}}"
OUTPUT="${IMPRINT_OUTPUT_DIR:-}"
PYTHON="${IMPRINT_PYTHON:-$ROOT/.venv/bin/python}"
HISTORY="$ROOT/src/mme_vla_suite/models/config/robomme/$IMPRINT_HISTORY_CONFIG"
EXP_NAME="${IMPRINT_EXP_NAME:-imprint-${IMPRINT_BENCHMARK}-${IMPRINT_ARCHITECTURE}-stage1}"
LATENT_BUCKETS="${IMPRINT_STAGE1_LATENT_BUCKETS:-256,512,1024,2048,4096}"

: "${STORE:?set IMPRINT_STAGE1_STORE or IMPRINT_STORE}"
: "${OUTPUT:?set IMPRINT_OUTPUT_DIR}"
test -x "$PYTHON"
test -f "$HISTORY"
test -d "$STORE"
if [[ ! "$LATENT_BUCKETS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "IMPRINT_STAGE1_LATENT_BUCKETS must be comma-separated positive integers" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
if [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to use nonempty output directory: $OUTPUT" >&2
  exit 3
fi

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OUTPUT/.openpi-cache}"
imprint_configure_training_runtime

args=(
  --config "$HISTORY"
  --store "$STORE"
  --num_steps "$IMPRINT_STAGE1_STEPS"
  --batch_size "$IMPRINT_BATCH_SIZE"
  --lr 1e-4
  --weight_decay 1e-4
  --warmup_steps "${IMPRINT_STAGE1_WARMUP_STEPS:-1000}"
  --lr_decay_steps "$IMPRINT_STAGE1_STEPS"
  --fsdp_devices "$IMPRINT_STAGE1_FSDP_DEVICES"
  --seed 0
  --w_align 1.0
  --w_action 0.5
  --w_proprio 0.5
  --w_reward 0.5
  --latent_buckets "$LATENT_BUCKETS"
  --max_recent_tokens "$IMPRINT_MAX_RECENT_TOKENS"
  --max_step_tokens "$IMPRINT_MAX_STEP_TOKENS"
  --episode_cache_size "${IMPRINT_EPISODE_CACHE_SIZE:-450}"
  --gather_workers "${IMPRINT_GATHER_WORKERS:-1}"
  --prefetch_batches "${IMPRINT_PREFETCH_BATCHES:-2}"
  --dino_transport_dtype float16
  --module_compute_dtype "$IMPRINT_STAGE1_MODULE_DTYPE"
  --checkpoint_dir "$OUTPUT"
  --ckpt_every 10000
  --log_every 50
  --project_name "${IMPRINT_WANDB_PROJECT:-Robocasa}"
  --exp_name "$EXP_NAME"
)
if [[ "${IMPRINT_LENGTH_BUCKETED_BATCHES:-0}" == "1" ]]; then
  args+=(--length_bucketed_batches)
fi
if [[ "${IMPRINT_WANDB_ENABLED:-1}" == "1" ]]; then
  args+=(--wandb_enabled)
fi

cd "$ROOT"
exec "$PYTHON" -u -m mme_vla_suite.scripts.pretrain_bottleneck "${args[@]}"
