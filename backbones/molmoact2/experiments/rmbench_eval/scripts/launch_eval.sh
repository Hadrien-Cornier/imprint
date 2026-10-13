#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${RMBENCH_TASK:-cover_blocks}"
TRIALS="${MEMFT_TRIAL_COUNT:-1}"
INPUT_REPO="${RMBENCH_MOLMOACT2_INPUT_REPO:-Amazon-FAR/imprint_models}"
INPUT_REVISION="${RMBENCH_MOLMOACT2_INPUT_REVISION:-main}"
INPUT_PREFIX="${RMBENCH_MOLMOACT2_INPUT_PREFIX:-simulation/rmbench/molmoact2}"
EVAL_ASSETS_REPO="${RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO:-Amazon-FAR/imprint_datasets}"
EVAL_ASSETS_REVISION="${RMBENCH_MOLMOACT2_EVAL_ASSETS_REVISION:-main}"
EVAL_ASSETS_PREFIX="${RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX:-simulation/rmbench/molmoact2_eval_assets}"

usage() {
  echo "usage: $0 [--task TASK] [--trials N]" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --task) TASK="${2:-}"; shift 2 ;;
    --trials) TRIALS="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$INPUT_REPO" == */* ]] || {
  echo "RMBENCH_MOLMOACT2_INPUT_REPO must be a Hugging Face repository ID" >&2
  exit 2
}
[[ "$EVAL_ASSETS_REPO" == */* ]] || {
  echo "RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO must be a Hugging Face repository ID" >&2
  exit 2
}
[[ -n "$INPUT_PREFIX" ]] || {
  echo "RMBENCH_MOLMOACT2_INPUT_PREFIX must be non-empty" >&2
  exit 2
}
[[ -n "$EVAL_ASSETS_PREFIX" ]] || {
  echo "RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX must be non-empty" >&2
  exit 2
}
[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] || usage

export RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO="$EVAL_ASSETS_REPO"
export RMBENCH_MOLMOACT2_EVAL_ASSETS_REVISION="$EVAL_ASSETS_REVISION"

exec bash "${PACKAGE_ROOT}/scripts/run_worker.sh" \
  --input-repo "$INPUT_REPO" \
  --input-revision "$INPUT_REVISION" \
  --input-prefix "$INPUT_PREFIX" \
  --eval-assets-prefix "$EVAL_ASSETS_PREFIX" \
  --task "$TASK" \
  --trials "$TRIALS"
