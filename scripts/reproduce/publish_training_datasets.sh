#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:-all}"
OUTPUT_ROOT="${IMPRINT_DATASET_RELEASE_ROOT:-}"
PYTHON="${IMPRINT_PYTHON:-python3}"
PACKAGE_WORKERS="${IMPRINT_PACKAGE_WORKERS:-4}"
UPLOAD_WORKERS="${IMPRINT_UPLOAD_WORKERS:-16}"
DATA_FILES_PER_SHARD="${IMPRINT_DATA_FILES_PER_SHARD:-5000}"
FEATURE_EPISODES_PER_SHARD="${IMPRINT_FEATURE_EPISODES_PER_SHARD:-4}"
HF_PRIVATE="${IMPRINT_HF_PRIVATE:-0}"
HF_REPO="${IMPRINT_DATASET_REPO:-Amazon-FAR/imprint_datasets}"
HF_REVISION="${IMPRINT_DATASETS_REVISION:-main}"

case "$HF_PRIVATE" in
  1|true|TRUE|yes|YES)
    HUB_VISIBILITY_ARGS=(--private)
    ;;
  0|false|FALSE|no|NO)
    HUB_VISIBILITY_ARGS=()
    ;;
  *)
    echo "IMPRINT_HF_PRIVATE must be one of: 1, 0, true, false, yes, no" >&2
    exit 2
    ;;
esac

require_directory() {
  if [[ ! -d "$2" ]]; then
    echo "$1 must name an existing directory: $2" >&2
    exit 2
  fi
}

package_layout() {
  local store="$1"
  local output="$2"
  local layout="$3"
  local store_id="$4"

  "$PYTHON" "$ROOT/scripts/reproduce/package_dataset_store.py" \
    --store "$store" \
    --output "$output" \
    --layout-prefix "$layout" \
    --store-id "$store_id" \
    --workers "$PACKAGE_WORKERS" \
    --data-files-per-shard "$DATA_FILES_PER_SHARD" \
    --feature-episodes-per-shard "$FEATURE_EPISODES_PER_SHARD"
}

publish_folder() {
  local repo_id="$1"
  local release_root="$2"
  local prefix="$3"
  local card="$4"

  cp "$card" "$release_root/$prefix/README.md"
  if hf upload-large-folder --help >/dev/null 2>&1; then
    hf upload-large-folder "$repo_id" "$release_root" \
      --repo-type dataset \
      --revision "$HF_REVISION" \
      --include "$prefix/**" \
      "${HUB_VISIBILITY_ARGS[@]}" \
      --num-workers "$UPLOAD_WORKERS"
  else
    hf upload "$repo_id" "$release_root/$prefix" "$prefix" \
      --repo-type dataset \
      --revision "$HF_REVISION" \
      "${HUB_VISIBILITY_ARGS[@]}"
  fi
}

publish_rmbench() {
  local prefix="simulation/rmbench/450"
  local output="$OUTPUT_ROOT/$prefix"
  require_directory IMPRINT_RMBENCH_STAGE1_STORE "$IMPRINT_RMBENCH_STAGE1_STORE"
  require_directory IMPRINT_RMBENCH_STAGE2_STORE "$IMPRINT_RMBENCH_STAGE2_STORE"
  mkdir -p "$output"

  package_layout \
    "$IMPRINT_RMBENCH_STAGE1_STORE" \
    "$output" \
    stage1 \
    rmbench-450-stage1-dinov3-causal
  package_layout \
    "$IMPRINT_RMBENCH_STAGE2_STORE" \
    "$output" \
    stage2 \
    rmbench-450-stage2-h50-dinov3-causal
  publish_folder \
    "$HF_REPO" \
    "$OUTPUT_ROOT" \
    "$prefix" \
    "$ROOT/artifacts/datasets/rmbench-card.md"
}

publish_robocasa() {
  local prefix="simulation/robocasa/600"
  local output="$OUTPUT_ROOT/$prefix"
  require_directory IMPRINT_ROBOCASA_STORE "$IMPRINT_ROBOCASA_STORE"
  mkdir -p "$output"

  package_layout \
    "$IMPRINT_ROBOCASA_STORE" \
    "$output" \
    store \
    robocasa-600-h20-dinov3-causal
  publish_folder \
    "$HF_REPO" \
    "$OUTPUT_ROOT" \
    "$prefix" \
    "$ROOT/artifacts/datasets/robocasa-card.md"
}

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "IMPRINT_DATASET_RELEASE_ROOT is required" >&2
  exit 2
fi
command -v "$PYTHON" >/dev/null || {
  echo "Python executable not found: $PYTHON" >&2
  exit 2
}
command -v hf >/dev/null || {
  echo "Hugging Face CLI 'hf' is required" >&2
  exit 2
}

case "$TARGET" in
  all)
    publish_rmbench
    publish_robocasa
    ;;
  rmbench)
    publish_rmbench
    ;;
  robocasa)
    publish_robocasa
    ;;
  *)
    echo "usage: $0 {all|rmbench|robocasa}" >&2
    exit 2
    ;;
esac
