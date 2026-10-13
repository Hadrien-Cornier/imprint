#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORE="${1:-}"
OUTPUT="${2:-}"
LAYOUT="${3:-}"
STORE_ID="${4:-}"
HUB_ID="${5:-}"
CARD="${6:-}"
PYTHON="${IMPRINT_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -d "$STORE" || -z "$OUTPUT" || -z "$LAYOUT" || -z "$STORE_ID" || -z "$HUB_ID" || ! -f "$CARD" ]]; then
  echo "usage: $0 STORE OUTPUT LAYOUT STORE_ID HUB_ID DATASET_CARD" >&2
  exit 2
fi
command -v hf >/dev/null || {
  echo "Hugging Face CLI 'hf' is required" >&2
  exit 2
}

"$PYTHON" "$ROOT/scripts/reproduce/package_dataset_store.py" \
  --store "$STORE" \
  --output "$OUTPUT" \
  --layout-prefix "$LAYOUT" \
  --store-id "$STORE_ID" \
  --workers "${IMPRINT_PACKAGE_WORKERS:-4}" \
  --data-files-per-shard "${IMPRINT_DATA_FILES_PER_SHARD:-10000}" \
  --feature-episodes-per-shard "${IMPRINT_FEATURE_EPISODES_PER_SHARD:-10}"
cp "$CARD" "$OUTPUT/README.md"
hf upload-large-folder "$HUB_ID" "$OUTPUT" \
  --repo-type dataset \
  --num-workers "${IMPRINT_UPLOAD_WORKERS:-16}"
