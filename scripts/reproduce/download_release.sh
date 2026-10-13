#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${IMPRINT_ARTIFACT_ROOT:-$ROOT/artifacts}"
BENCHMARK="${1:-}"
ARCHITECTURE="${2:-}"
SELECTION="${3:---all}"
MODEL_REPO="${IMPRINT_MODEL_REPO:-Amazon-FAR/imprint_models}"
DATASET_REPO="${IMPRINT_DATASET_REPO:-Amazon-FAR/imprint_datasets}"
MODEL_REVISION="${IMPRINT_MODELS_REVISION:-main}"
DATASET_REVISION="${IMPRINT_DATASETS_REVISION:-main}"
EVAL_ASSETS_PREFIX=
EVAL_ASSETS_DIR=

case "$BENCHMARK/$ARCHITECTURE" in
  rmbench/full_attention)
    MODEL_PREFIX=simulation/rmbench/full_attention
    MODEL_DIR=rmbench-full-attention
    DATASET_PREFIX=simulation/rmbench/450
    DATASET_DIR=rmbench
    ;;
  rmbench/recurrent)
    MODEL_PREFIX=simulation/rmbench/recurrent
    MODEL_DIR=rmbench-recurrent
    DATASET_PREFIX=simulation/rmbench/450
    DATASET_DIR=rmbench
    ;;
  rmbench/molmoact2)
    MODEL_PREFIX=simulation/rmbench/molmoact2
    MODEL_DIR=rmbench-molmoact2
    DATASET_PREFIX=simulation/rmbench/450
    DATASET_DIR=rmbench
    EVAL_ASSETS_PREFIX=simulation/rmbench/molmoact2_eval_assets
    EVAL_ASSETS_DIR=rmbench-molmoact2-eval-assets
    ;;
  robocasa/full_attention)
    MODEL_PREFIX=simulation/robocasa/full_attention
    MODEL_DIR=robocasa-full-attention
    DATASET_PREFIX=simulation/robocasa/600
    DATASET_DIR=robocasa
    ;;
  robocasa/recurrent)
    MODEL_PREFIX=simulation/robocasa/recurrent
    MODEL_DIR=robocasa-recurrent
    DATASET_PREFIX=simulation/robocasa/600
    DATASET_DIR=robocasa
    ;;
  *)
    echo "usage: $0 {rmbench|robocasa} {full_attention|recurrent|molmoact2} [--all|--model-only|--dataset-only]" >&2
    exit 2
    ;;
esac

case "$SELECTION" in
  --all|--model-only|--dataset-only) ;;
  *)
    echo "unknown selection: $SELECTION" >&2
    echo "expected --all, --model-only, or --dataset-only" >&2
    exit 2
    ;;
esac

HF_CLI="${IMPRINT_HF_CLI:-$(command -v hf 2>/dev/null || true)}"
if [[ -z "$HF_CLI" || ! -x "$HF_CLI" ]]; then
  echo "Hugging Face CLI 'hf' is required" >&2
  exit 2
fi

DOWNLOAD_WORKERS="${IMPRINT_DOWNLOAD_WORKERS:-16}"
MODEL_CACHE="$ARTIFACT_ROOT/downloads/imprint_models"
DATASET_CACHE="$ARTIFACT_ROOT/downloads/imprint_datasets"

if [[ "$SELECTION" != "--dataset-only" ]]; then
  "$HF_CLI" download "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --include "$MODEL_PREFIX/**" \
    --local-dir "$MODEL_CACHE" \
    --max-workers "$DOWNLOAD_WORKERS"
  mkdir -p "$ARTIFACT_ROOT/checkpoints/$MODEL_DIR"
  cp -a "$MODEL_CACHE/$MODEL_PREFIX/." \
    "$ARTIFACT_ROOT/checkpoints/$MODEL_DIR/"
  if [[ -n "$EVAL_ASSETS_PREFIX" ]]; then
    "$HF_CLI" download "$DATASET_REPO" \
      --repo-type dataset \
      --revision "$DATASET_REVISION" \
      --include "$EVAL_ASSETS_PREFIX/**" \
      --local-dir "$DATASET_CACHE" \
      --max-workers "$DOWNLOAD_WORKERS"
    mkdir -p "$ARTIFACT_ROOT/datasets/$EVAL_ASSETS_DIR"
    cp -a "$DATASET_CACHE/$EVAL_ASSETS_PREFIX/." \
      "$ARTIFACT_ROOT/datasets/$EVAL_ASSETS_DIR/"
  fi
fi

if [[ "$SELECTION" != "--model-only" ]]; then
  PYTHON="${IMPRINT_PYTHON:-$ROOT/.venv/bin/python}"
  [[ -x "$PYTHON" ]] || {
    echo "Python environment is required to extract the dataset; run 'uv sync --python 3.11 --frozen' or set IMPRINT_PYTHON" >&2
    exit 2
  }
  "$HF_CLI" download "$DATASET_REPO" \
    --repo-type dataset \
    --revision "$DATASET_REVISION" \
    --include "$DATASET_PREFIX/**" \
    --local-dir "$DATASET_CACHE" \
    --max-workers "$DOWNLOAD_WORKERS"
  "$PYTHON" "$ROOT/scripts/reproduce/extract_dataset_store.py" \
    --archive-root "$DATASET_CACHE/$DATASET_PREFIX" \
    --output "$ARTIFACT_ROOT/datasets/$DATASET_DIR" \
    --skip-existing
fi
