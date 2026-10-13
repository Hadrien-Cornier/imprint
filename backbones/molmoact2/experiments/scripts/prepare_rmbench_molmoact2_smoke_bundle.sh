#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name
    for name in "$@"; do
        [[ -n "${!name:-}" ]] || {
            printf 'Required environment variable %s is unset or empty.\n' "$name" >&2
            exit 2
        }
    done
}

require_dir() {
    local name="$1"
    local path="$2"
    [[ -d "$path" ]] || {
        printf '%s must identify an existing directory: %s\n' "$name" "$path" >&2
        exit 2
    }
}

require_env \
    MOLMOACT2_CHECKPOINT \
    RMBENCH_STAGE2_ROOT \
    RMBENCH_STAGE1_FEATURE_ROOT

MODEL_REPO="${RMBENCH_MOLMOACT2_INPUT_REPO:-Amazon-FAR/imprint_models}"
MODEL_PREFIX="${RMBENCH_MOLMOACT2_INPUT_PREFIX:-simulation/rmbench/molmoact2}"
EVAL_ASSETS_REPO="${RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO:-Amazon-FAR/imprint_datasets}"
EVAL_ASSETS_PREFIX="${RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX:-simulation/rmbench/molmoact2_eval_assets}"
RELEASE_ROOT="${RMBENCH_MOLMOACT2_RELEASE_ROOT:-}"
UPLOAD="${RMBENCH_MOLMOACT2_UPLOAD:-1}"
UPLOAD_WORKERS="${IMPRINT_UPLOAD_WORKERS:-16}"

case "$UPLOAD" in
    0|1) ;;
    *)
        echo "RMBENCH_MOLMOACT2_UPLOAD must be 0 or 1." >&2
        exit 2
        ;;
esac

require_dir MOLMOACT2_CHECKPOINT "$MOLMOACT2_CHECKPOINT"
require_dir RMBENCH_STAGE2_ROOT "$RMBENCH_STAGE2_ROOT"
require_dir RMBENCH_STAGE1_FEATURE_ROOT "$RMBENCH_STAGE1_FEATURE_ROOT"

stage2_meta="${RMBENCH_STAGE2_ROOT}/meta"
[[ -f "${stage2_meta}/episodes.jsonl" ]] || exit 2
[[ -f "${stage2_meta}/memory_layout.json" ]] || exit 2

model_file="${MOLMOACT2_CHECKPOINT}/model.pt"
checkpoint_metadata="${MOLMOACT2_CHECKPOINT}/config.yaml"
if [[ ! -f "$checkpoint_metadata" ]]; then
    checkpoint_metadata="${MOLMOACT2_CHECKPOINT}/config.json"
fi
[[ -f "$model_file" ]] || {
    echo "The public release requires a model-only checkpoint at ${model_file}." >&2
    echo "Run unshard_rmbench_molmoact2_checkpoint.py on the original checkpoint first." >&2
    exit 2
}
[[ -f "$checkpoint_metadata" ]] || {
    echo "Checkpoint must contain config.yaml or config.json: $MOLMOACT2_CHECKPOINT" >&2
    exit 2
}

if [[ -z "$RELEASE_ROOT" ]]; then
    RELEASE_ROOT="$(mktemp -d)"
    trap 'rm -rf "$RELEASE_ROOT"' EXIT
else
    mkdir -p "$RELEASE_ROOT"
fi

model_repo_root="${RELEASE_ROOT}/model-repository"
dataset_repo_root="${RELEASE_ROOT}/dataset-repository"
model_root="${model_repo_root}/${MODEL_PREFIX}"
assets_root="${dataset_repo_root}/${EVAL_ASSETS_PREFIX}"
rm -rf "$model_repo_root" "$dataset_repo_root"
mkdir -p "$model_root/checkpoint" "$assets_root/stage2/meta" \
    "$assets_root/stage1-language"

cp "$model_file" "$model_root/checkpoint/model.pt"
cp "$checkpoint_metadata" "$model_root/checkpoint/$(basename "$checkpoint_metadata")"
cp "${stage2_meta}/episodes.jsonl" "$assets_root/stage2/meta/episodes.jsonl"
cp "${stage2_meta}/memory_layout.json" "$assets_root/stage2/meta/memory_layout.json"

while IFS= read -r -d '' source; do
    relative="${source#${RMBENCH_STAGE1_FEATURE_ROOT%/}/}"
    destination="${assets_root}/stage1-language/${relative}"
    mkdir -p "$(dirname "$destination")"
    cp "$source" "$destination"
done < <(
    find "$RMBENCH_STAGE1_FEATURE_ROOT" \
        -type f -path "*/episode_*/lang_emb.npy" -print0
)

language_count="$(
    find "$assets_root/stage1-language" -type f -name lang_emb.npy | wc -l | tr -d ' '
)"
(( language_count > 0 )) || {
    echo "No stored Pi0.5 language embeddings were found." >&2
    exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cp "$root/artifacts/checkpoints/molmoact2-card.md" "$model_root/README.md"
cp "$root/artifacts/datasets/molmoact2-eval-assets-card.md" "$assets_root/README.md"

python3 - "$model_root/manifest.json" "$assets_root/manifest.json" \
    "$model_file" "$checkpoint_metadata" "$language_count" \
    "$MODEL_REPO" "$MODEL_PREFIX" "$EVAL_ASSETS_REPO" \
    "$EVAL_ASSETS_PREFIX" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    model_manifest,
    assets_manifest,
    model_file,
    checkpoint_metadata,
    language_count,
    model_repo,
    model_prefix,
    assets_repo,
    assets_prefix,
) = sys.argv[1:]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


model_payload = {
    "schema_version": 1,
    "hf_repo": model_repo,
    "path_prefix": model_prefix,
    "release_revision": "main",
    "checkpoint_step": 100000,
    "checkpoint": {
        "path": "checkpoint/model.pt",
        "sha256": sha256(model_file),
    },
    "checkpoint_config": {
        "path": f"checkpoint/{Path(checkpoint_metadata).name}",
        "sha256": sha256(checkpoint_metadata),
    },
    "memory_enabled": True,
    "visual_history": "online_wan_and_dinov3",
    "offline_visual_features": False,
    "eval_assets_repo": assets_repo,
    "eval_assets_path_prefix": assets_prefix,
    "base_model": {
        "repo": "allenai/MolmoAct2",
        "revision": "e432d85f6e039edca44afb93c262f3084ab72a9c",
    },
}
Path(model_manifest).write_text(
    json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

assets_payload = {
    "schema_version": 1,
    "hf_repo": assets_repo,
    "path_prefix": assets_prefix,
    "release_revision": "main",
    "stored_language_embedding_files": int(language_count),
    "stage2_metadata": [
        "stage2/meta/episodes.jsonl",
        "stage2/meta/memory_layout.json",
    ],
    "dino_model": {
        "repo": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
    },
    "qwen_tokenizer": {
        "repo": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
}
Path(assets_manifest).write_text(
    json.dumps(assets_payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if (( UPLOAD )); then
    command -v hf >/dev/null || {
        echo "Hugging Face CLI 'hf' is required to publish the release." >&2
        exit 127
    }
    hf upload-large-folder "$MODEL_REPO" "$model_repo_root" \
        --num-workers "$UPLOAD_WORKERS"
    hf upload-large-folder "$EVAL_ASSETS_REPO" "$dataset_repo_root" \
        --repo-type dataset \
        --num-workers "$UPLOAD_WORKERS"
fi

printf 'model_release=%s\n' "$model_root"
printf 'eval_assets_release=%s\n' "$assets_root"
printf 'model_repo=https://huggingface.co/%s\n' "$MODEL_REPO"
printf 'model_prefix=%s\n' "$MODEL_PREFIX"
printf 'eval_assets_repo=https://huggingface.co/datasets/%s\n' "$EVAL_ASSETS_REPO"
printf 'eval_assets_prefix=%s\n' "$EVAL_ASSETS_PREFIX"
