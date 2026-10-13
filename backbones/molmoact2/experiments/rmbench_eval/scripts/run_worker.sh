#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${EXPERIMENTS_ROOT}/.." && pwd)"
MOLMOACT2_MOUNT="${MOLMOACT2_MOUNT:-${REPOSITORY_ROOT}}"
EVAL_PACKAGE_ROOT="${RMBENCH_EVAL_PACKAGE_ROOT:-${PACKAGE_ROOT}}"
RUNTIME_ROOT="${RMBENCH_RUNTIME_ROOT:-/opt/imprint/rmbench/runtime}"
OIDN_ROOT="${RMBENCH_OIDN_ROOT:-/opt/imprint/rmbench/oidn233}"
POLICY_PY="${RMBENCH_POLICY_PYTHON:-${RUNTIME_ROOT}/openpi_rc/.venv/bin/python}"
WORK_ROOT="${RMBENCH_MOLMOACT2_SMOKE_WORK_ROOT:-${TMPDIR:-/tmp}/imprint-molmoact2-rmbench}"
SIMULATOR_CACHE_ROOT="${WORK_ROOT}/simulator_cache"
RUNTIME_ARCHIVE_DIR="${RMBENCH_RUNTIME_ARCHIVE_DIR:-}"
MODEL_REPO="${RMBENCH_MOLMOACT2_INPUT_REPO:-Amazon-FAR/imprint_models}"
MODEL_REVISION="${RMBENCH_MOLMOACT2_INPUT_REVISION:-main}"
MODEL_PREFIX="${RMBENCH_MOLMOACT2_INPUT_PREFIX:-simulation/rmbench/molmoact2}"
EVAL_ASSETS_REPO="${RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO:-Amazon-FAR/imprint_datasets}"
EVAL_ASSETS_REVISION="${RMBENCH_MOLMOACT2_EVAL_ASSETS_REVISION:-main}"
EVAL_ASSETS_PREFIX="${RMBENCH_MOLMOACT2_EVAL_ASSETS_PREFIX:-simulation/rmbench/molmoact2_eval_assets}"
DINO_HF_REPO="${RMBENCH_DINO_HF_REPO:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
DINO_HF_REVISION="${RMBENCH_DINO_HF_REVISION:-5931719e67bbdb9737e363e781fb0c67687896bc}"
QWEN_HF_REPO="${RMBENCH_QWEN_HF_REPO:-Qwen/Qwen3-4B}"
QWEN_HF_REVISION="${RMBENCH_QWEN_HF_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
HF_DOWNLOAD_WORKERS="${RMBENCH_HF_DOWNLOAD_WORKERS:-16}"
SKIP_RUNTIME_STAGE="${RMBENCH_SKIP_RUNTIME_STAGE:-1}"
USE_STAGED_INPUT="${RMBENCH_USE_STAGED_INPUT:-0}"
PORT="${RMBENCH_MOLMOACT2_PORT:-8031}"
RESULT_HF_REPO="${RMBENCH_RESULT_HF_REPO:-}"
RESULT_HF_PATH="${RMBENCH_RESULT_HF_PATH:-}"
WANDB_REQUIRED="${RMBENCH_WANDB_REQUIRED:-0}"
WANDB_RUN_NAME="${RMBENCH_WANDB_RUN_NAME:-}"
OFFICIAL_ALL9_TASKS=(
  "battery_try"
  "blocks_ranking_try"
  "cover_blocks"
  "observe_and_pickup"
  "press_button"
  "put_back_block"
  "rearrange_blocks"
  "swap_T"
  "swap_blocks"
)
TASK=
TRIALS=
PREPARE_ONLY=0
policy_pid=
wandb_pid=
wandb_done=

usage() {
  echo "usage: $0 [--input-repo REPO_ID] [--input-revision REVISION] [--input-prefix PATH] [--eval-assets-prefix PATH] --task TASK --trials N [--prepare-only]" >&2
  exit 2
}

require_file() {
  local name="$1"
  local path="$2"
  [[ -f "$path" ]] || {
    echo "$name is missing: $path" >&2
    exit 2
  }
}

require_dir() {
    local name="$1"
    local path="$2"
    [[ -d "$path" ]] || {
        echo "$name is missing: $path" >&2
        exit 2
    }
}

run_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        python "$@"
        return
    fi
    if [[ -x "$POLICY_PY" ]]; then
        "$POLICY_PY" "$@"
        return
    fi
    echo "python3 or python is required for worker metadata validation." >&2
    exit 2
}

ensure_hf_cli() {
  command -v hf >/dev/null 2>&1 || {
    echo "Hugging Face CLI 'hf' is required to stage evaluation artifacts." >&2
    exit 127
  }
}

run_as_root() {
  if (( EUID == 0 )); then
    "$@"
    return
  fi
  command -v sudo >/dev/null 2>&1 || {
    echo "sudo is required to install the root-relative RMBench runtime." >&2
    exit 127
  }
  sudo -E "$@"
}

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return
  fi
  run_as_root env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  run_as_root env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y -qq --no-install-recommends ffmpeg
  command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg installation did not provide ffmpeg" >&2
    exit 1
  }
}

stage_hf() {
  local label="$1"
  shift
  printf 'RMBench Hugging Face stage: %s\n' "$label"
  hf download "$@" --max-workers "$HF_DOWNLOAD_WORKERS"
}

stop_policy() {
  [[ -n "$policy_pid" ]] || return 0
  if kill -0 "$policy_pid" 2>/dev/null; then
    kill -- "-${policy_pid}" 2>/dev/null || true
  fi
  wait "$policy_pid" 2>/dev/null || true
}

stop_wandb_logger() {
  [[ -n "$wandb_pid" ]] || return 0
  [[ -z "$wandb_done" ]] || touch "$wandb_done"
  wait "$wandb_pid"
}

dump_failure_logs() {
  local status="$1"
  local log_path
  (( status != 0 )) || return 0
  [[ -n "${OUTPUT_ROOT:-}" ]] || return 0
  for log_path in "${OUTPUT_ROOT}/evaluator.log" "${OUTPUT_ROOT}/policy.log"; do
    [[ -f "$log_path" ]] || continue
    printf '\n--- Last 240 lines: %s ---\n' "$log_path" >&2
    tail -240 "$log_path" >&2 || true
  done
}

publish_results() {
  [[ -n "$RESULT_HF_REPO" ]] || return 0
  ensure_hf_cli
  hf upload "$RESULT_HF_REPO" "$OUTPUT_ROOT" "$RESULT_HF_PATH" \
    --repo-type dataset \
    --commit-message "Add ${TASK} RMBench evaluation outputs"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_policy
  if ! stop_wandb_logger && (( WANDB_REQUIRED )); then
    status=1
  fi
  dump_failure_logs "$status"
  exit "$status"
}
trap cleanup EXIT INT TERM

while (($#)); do
  case "$1" in
    --input-repo) MODEL_REPO="${2:-}"; shift 2 ;;
    --input-revision) MODEL_REVISION="${2:-}"; shift 2 ;;
    --input-prefix) MODEL_PREFIX="${2:-}"; shift 2 ;;
    --eval-assets-prefix) EVAL_ASSETS_PREFIX="${2:-}"; shift 2 ;;
    --task) TASK="${2:-}"; shift 2 ;;
    --trials) TRIALS="${2:-}"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    *) usage ;;
  esac
done

[[ "$USE_STAGED_INPUT" == "0" || "$USE_STAGED_INPUT" == "1" ]] || {
  echo "RMBENCH_USE_STAGED_INPUT must be 0 or 1" >&2
  exit 2
}
[[ "$WANDB_REQUIRED" == "0" || "$WANDB_REQUIRED" == "1" ]] || {
  echo "RMBENCH_WANDB_REQUIRED must be 0 or 1" >&2
  exit 2
}
if (( WANDB_REQUIRED )); then
  [[ -n "${WANDB_API_KEY:-}" ]] || {
    echo "WANDB_API_KEY is required when RMBENCH_WANDB_REQUIRED=1" >&2
    exit 2
  }
fi
if (( ! USE_STAGED_INPUT )); then
  [[ "$MODEL_REPO" == */* && -n "$MODEL_REVISION" ]] || usage
  [[ "$EVAL_ASSETS_REPO" == */* && -n "$EVAL_ASSETS_REVISION" ]] || usage
  [[ -n "$MODEL_PREFIX" && "$MODEL_PREFIX" != /* ]] || usage
  [[ -n "$EVAL_ASSETS_PREFIX" && "$EVAL_ASSETS_PREFIX" != /* ]] || usage
fi
task_supported=0
for official_task in "${OFFICIAL_ALL9_TASKS[@]}"; do
  if [[ "$TASK" == "$official_task" ]]; then
    task_supported=1
    break
  fi
done
(( task_supported )) || {
  echo "MolmoAct2 RMBench worker only supports official all9 tasks" >&2
  exit 2
}
[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] || usage
trial_count="$TRIALS"
[[ "$SKIP_RUNTIME_STAGE" == "0" || "$SKIP_RUNTIME_STAGE" == "1" ]] || {
  echo "RMBENCH_SKIP_RUNTIME_STAGE must be 0 or 1" >&2
  exit 2
}
START_SEED="${RMBENCH_START_SEED:-100000}"
[[ "$START_SEED" =~ ^[0-9]+$ ]] || {
  echo "RMBENCH_START_SEED must be a non-negative integer" >&2
  exit 2
}
EXECUTION_HORIZON="${RMBENCH_EXECUTION_HORIZON:-50}"
[[ "$EXECUTION_HORIZON" =~ ^[1-9][0-9]*$ ]] &&
  (( EXECUTION_HORIZON <= 50 )) || {
  echo "RMBENCH_EXECUTION_HORIZON must be an integer in [1, 50]" >&2
  exit 2
}
SINGLE_GPU="${RMBENCH_SINGLE_GPU:-0}"
[[ "$SINGLE_GPU" == "0" || "$SINGLE_GPU" == "1" ]] || {
  echo "RMBENCH_SINGLE_GPU must be 0 or 1" >&2
  exit 2
}
WORKER_GPU="${RMBENCH_WORKER_GPU:-0}"
[[ "$WORKER_GPU" =~ ^[0-9]+$ ]] || {
  echo "RMBENCH_WORKER_GPU must be a non-negative integer" >&2
  exit 2
}
POLICY_GPU="$WORKER_GPU"
EVALUATOR_GPU=$((WORKER_GPU + 1))
if (( SINGLE_GPU )); then
  EVALUATOR_GPU="$WORKER_GPU"
fi

is_b200_gpu() {
  local gpu_index="$1"
  nvidia-smi --id="$gpu_index" --query-gpu=name --format=csv,noheader \
    | grep -qi 'b200'
}

FLOW_SEED="${RMBENCH_FLOW_SEED:-0}"
[[ "$FLOW_SEED" =~ ^[0-9]+$ ]] || {
  echo "RMBENCH_FLOW_SEED must be a non-negative integer" >&2
  exit 2
}

require_dir "mounted MolmoAct2 repository" "$MOLMOACT2_MOUNT"
require_dir "RMBench evaluation package" "$EVAL_PACKAGE_ROOT"
require_file "RMBench evaluation overlay" \
  "$EVAL_PACKAGE_ROOT/runtime/eval_policy_four_view.py"
require_file "RMBench four-view recorder" \
  "$EVAL_PACKAGE_ROOT/four_view.py"

mkdir -p "$WORK_ROOT" "$SIMULATOR_CACHE_ROOT/warp" "$SIMULATOR_CACHE_ROOT/torch_extensions"
if (( ! USE_STAGED_INPUT )); then
  ensure_hf_cli
fi

INPUT_ROOT="${WORK_ROOT}/input"
OUTPUT_ROOT="${WORK_ROOT}/output"
RUNTIME_STAGE="${WORK_ROOT}/runtime"
HF_CACHE_ROOT="${WORK_ROOT}/hf-cache"
MODEL_ROOT="${INPUT_ROOT}/model-release"
EVAL_ASSETS_ROOT="${INPUT_ROOT}/eval-assets"
MODEL_DOWNLOAD_ROOT="${INPUT_ROOT}/model-repository"
EVAL_ASSETS_DOWNLOAD_ROOT="${INPUT_ROOT}/dataset-repository"
MANIFEST_PATH="${MODEL_ROOT}/manifest.json"
MEMORY_ENABLED=
if [[ -z "$RESULT_HF_PATH" ]]; then
  RESULT_HF_PATH="${TASK}/seed-${RMBENCH_START_SEED:-100000}"
fi

if (( USE_STAGED_INPUT )); then
  MEMORY_ENABLED="${RMBENCH_MEMORY_ENABLED:-1}"
  [[ "$MEMORY_ENABLED" == "0" || "$MEMORY_ENABLED" == "1" ]] || {
    echo "RMBENCH_MEMORY_ENABLED must be 0 or 1 in staged-input mode" >&2
    exit 2
  }
  MOLMOACT2_SOURCE="$MOLMOACT2_MOUNT"
  CHECKPOINT_ROOT="${RMBENCH_STAGED_CHECKPOINT:-}"
  STAGE2_ROOT="${RMBENCH_STAGED_STAGE2_ROOT:-}"
  FEATURE_ROOT="${RMBENCH_STAGED_STAGE1_FEATURE_ROOT:-}"
  DINO_ROOT="${RMBENCH_STAGED_DINO_ROOT:-}"
  PI05_ROOT="${RMBENCH_STAGED_PI05_ROOT:-}"
  required_staged_paths=("$CHECKPOINT_ROOT")
  if (( MEMORY_ENABLED )); then
    required_staged_paths+=("$STAGE2_ROOT" "$FEATURE_ROOT" "$DINO_ROOT" "$PI05_ROOT")
  fi
  for path in "${required_staged_paths[@]}"; do
    [[ -n "$path" ]] || {
      echo "All RMBENCH_STAGED_* input roots are required in staged-input mode" >&2
      exit 2
    }
    require_dir "staged evaluation input" "$path"
  done
  rm -rf "$OUTPUT_ROOT" "$RUNTIME_STAGE"
  mkdir -p "$OUTPUT_ROOT" "$RUNTIME_STAGE"
  printf 'Using staged evaluation inputs rooted at: %s\n' "$MOLMOACT2_SOURCE"
else
  MOLMOACT2_SOURCE="$MOLMOACT2_MOUNT"
  CHECKPOINT_ROOT="${MODEL_ROOT}/checkpoint"
  STAGE2_ROOT="${EVAL_ASSETS_ROOT}/stage2"
  FEATURE_ROOT="${EVAL_ASSETS_ROOT}/stage1-language"
  DINO_ROOT="${INPUT_ROOT}/dinov3"
  PI05_ROOT="${EXPERIMENTS_ROOT}/stage1"
  rm -rf \
    "$MODEL_ROOT" \
    "$EVAL_ASSETS_ROOT" \
    "$MODEL_DOWNLOAD_ROOT" \
    "$EVAL_ASSETS_DOWNLOAD_ROOT" \
    "$DINO_ROOT" \
    "$OUTPUT_ROOT" \
    "$RUNTIME_STAGE"
  mkdir -p "$INPUT_ROOT" "$OUTPUT_ROOT" "$RUNTIME_STAGE"
fi

# Reuse the reference evaluator's pinned simulator and policy runtimes.
if (( ! SKIP_RUNTIME_STAGE )); then
  [[ -n "$RUNTIME_ARCHIVE_DIR" && -d "$RUNTIME_ARCHIVE_DIR" ]] || {
    echo "RMBENCH_RUNTIME_ARCHIVE_DIR is required when RMBENCH_SKIP_RUNTIME_STAGE=0." >&2
    exit 2
  }
  for artifact in runtime.tar uv-python.tar runtime-manifest.json; do
    require_file "RMBench runtime artifact" "${RUNTIME_ARCHIVE_DIR}/${artifact}"
    cp "${RUNTIME_ARCHIVE_DIR}/${artifact}" "${RUNTIME_STAGE}/${artifact}"
  done
  run_as_root tar --extract --file "${RUNTIME_STAGE}/uv-python.tar" --directory /
  run_as_root tar --extract --file "${RUNTIME_STAGE}/runtime.tar" --directory /
else
  printf 'Skipping runtime archive staging; using pre-staged runtime: %s\n' \
    "$RUNTIME_ROOT"
fi

if (( ! USE_STAGED_INPUT )); then
  stage_hf "MolmoAct2 model-only checkpoint" \
    "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --include "${MODEL_PREFIX}/**" \
    --local-dir "$MODEL_DOWNLOAD_ROOT"
  mkdir -p "$MODEL_ROOT"
  cp -a "${MODEL_DOWNLOAD_ROOT}/${MODEL_PREFIX}/." "$MODEL_ROOT/"
  stage_hf "MolmoAct2 compact evaluation assets" \
    "$EVAL_ASSETS_REPO" \
    --repo-type dataset \
    --revision "$EVAL_ASSETS_REVISION" \
    --include "${EVAL_ASSETS_PREFIX}/**" \
    --local-dir "$EVAL_ASSETS_DOWNLOAD_ROOT"
  mkdir -p "$EVAL_ASSETS_ROOT"
  cp -a "${EVAL_ASSETS_DOWNLOAD_ROOT}/${EVAL_ASSETS_PREFIX}/." \
    "$EVAL_ASSETS_ROOT/"
  stage_hf "DINOv3 visual encoder" \
    "$DINO_HF_REPO" \
    --revision "$DINO_HF_REVISION" \
    --local-dir "$DINO_ROOT"
  MEMORY_ENABLED="$(run_python - "$MANIFEST_PATH" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    manifest = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid MolmoAct2 bundle manifest at {path}: {exc}") from exc
if not isinstance(manifest, dict):
    raise SystemExit("MolmoAct2 bundle manifest must be a JSON object")
memory_enabled = manifest.get("memory_enabled", True)
if not isinstance(memory_enabled, bool):
    raise SystemExit("bundle memory_enabled must be a boolean")
expected_visual_history = "online_wan_and_dinov3" if memory_enabled else "none"
if manifest.get("visual_history") != expected_visual_history:
    raise SystemExit(
        f"bundle must declare visual_history={expected_visual_history}"
    )
if manifest.get("offline_visual_features") is not False:
    raise SystemExit("bundle must declare offline_visual_features=false")
print(int(memory_enabled))
PY
  )"
fi

qwen_snapshot="${HF_CACHE_ROOT}/hub/models--Qwen--Qwen3-4B/snapshots/${QWEN_HF_REVISION}"
if [[ ! -s "${qwen_snapshot}/tokenizer.json" ]]; then
  mkdir -p "$HF_CACHE_ROOT"
  stage_hf "Qwen3 tokenizer" \
    "$QWEN_HF_REPO" \
    --revision "$QWEN_HF_REVISION" \
    --cache-dir "${HF_CACHE_ROOT}/hub"
fi
require_file "Qwen3 tokenizer cache" "${qwen_snapshot}/tokenizer.json"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "$CHECKPOINT_ROOT/config.yaml" && ! -f "$CHECKPOINT_ROOT/config.json" ]]; then
  echo "MolmoAct2 checkpoint must contain config.yaml or config.json: $CHECKPOINT_ROOT" >&2
  exit 2
fi
require_file "MolmoAct2 serving entrypoint" \
  "$MOLMOACT2_SOURCE/experiments/scripts/serve_rmbench_molmoact2.py"
require_file "MolmoAct2 TCP serving entrypoint" \
  "$MOLMOACT2_SOURCE/experiments/scripts/serve_rmbench_molmoact2_tcp.py"
require_file "MolmoAct2 RMBench policy adapter" \
  "$MOLMOACT2_SOURCE/experiments/scripts/rmbench_molmoact2_adapter.py"
if (( MEMORY_ENABLED )); then
  require_file "Stage-2 prompt metadata" "$STAGE2_ROOT/meta/episodes.jsonl"
  require_file "Stage-2 memory layout" "$STAGE2_ROOT/meta/memory_layout.json"
  require_file "DINO configuration" "$DINO_ROOT/config.json"
  require_file "Pi0.5 Wan frontend" \
    "$PI05_ROOT/mme_vla_suite/models/representation/wan_frontend.py"
  find "$FEATURE_ROOT" -type f -path "*/episode_*/lang_emb.npy" -print -quit | grep -q . || {
    echo "bundle contains no stored Pi0.5 language embeddings" >&2
    exit 2
  }
fi

if (( ! USE_STAGED_INPUT )); then
  offline_visual_paths="$(
    find "$INPUT_ROOT" -type f \
      \( -name "dinov3.npy" -o -name "latents_v*.npy" \
         -o -name "memory_cosmo_latents*.npy" \
         -o -name "memory_recent_embeddings*.npy" \) \
      -print
  )"
  [[ -z "$offline_visual_paths" ]] || {
    printf 'Offline visual arrays are forbidden in the MolmoAct2 smoke bundle:\n%s\n' \
      "$offline_visual_paths" >&2
    exit 2
  }
fi

SIM_PY="${RMBENCH_SIM_PYTHON:-${OIDN_ROOT}/.venv/bin/python}"
RMBENCH_ROOT="${RMBENCH_ROOT:-${RUNTIME_ROOT}/RMBench}"
OVERLAY="$EVAL_PACKAGE_ROOT/runtime/eval_policy_four_view.py"
CLIENT_SOURCE="$RMBENCH_ROOT/script/eval_policy_client.py"
CLIENT="${OUTPUT_ROOT}/eval_policy_client.py"
EVAL_PYTHONPATH="${EXPERIMENTS_ROOT}:${MOLMOACT2_SOURCE}/experiments/scripts"

[[ -x "$POLICY_PY" ]] || {
  echo "RMBench policy Python is not executable: $POLICY_PY" >&2
  exit 2
}
[[ -x "$SIM_PY" ]] || {
  echo "RMBench simulator Python is not executable: $SIM_PY" >&2
  exit 2
}
require_dir "RMBench root" "$RMBENCH_ROOT"
require_file "official RMBench policy client" "$CLIENT_SOURCE"

verify_b200_curobo_runtime() {
  local curobo_root="${OIDN_ROOT}/curobo-v0.7.8"
  local curobo_lib="${curobo_root}/src/curobo/curobolib"
  local extension
  local extensions=(
    "lbfgs_step_cu.cpython-310-x86_64-linux-gnu.so"
    "kinematics_fused_cu.cpython-310-x86_64-linux-gnu.so"
    "line_search_cu.cpython-310-x86_64-linux-gnu.so"
    "tensor_step_cu.cpython-310-x86_64-linux-gnu.so"
    "geom_cu.cpython-310-x86_64-linux-gnu.so"
  )

  for extension in "${extensions[@]}"; do
    if [[ ! -s "${curobo_lib}/${extension}" ]] ||
      ! grep -aFq "sm_100" "${curobo_lib}/${extension}"; then
      return 1
    fi
  done

  CUDA_VISIBLE_DEVICES="$EVALUATOR_GPU" \
    PYTHONPATH="${RMBENCH_ROOT}:${curobo_root}/src" \
    "$SIM_PY" - "$RMBENCH_ROOT" <<'PY'
from pathlib import Path
import sys

import torch
from curobo.wrap.reacher.motion_gen import MotionGenConfig

if torch.cuda.get_device_capability() != (10, 0):
    raise SystemExit(
        f"expected B200 capability (10, 0), got {torch.cuda.get_device_capability()}"
    )

config = MotionGenConfig.load_from_robot_config(
    str(Path(sys.argv[1]) / "assets/embodiments/aloha-agilex/curobo_left.yml"),
    {
        "cuboid": {
            "table": {
                "dims": [0.7, 2.0, 0.04],
                "pose": [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0],
            }
        }
    },
    interpolation_dt=1 / 250,
    num_trajopt_seeds=1,
)
assert config is not None
print("curobo_sm100_motion_gen=ok")
PY
}

ensure_b200_curobo_runtime() {
  is_b200_gpu "$EVALUATOR_GPU" || return 0

  if verify_b200_curobo_runtime; then
    echo "B200 CuRobo preflight passed"
    return 0
  fi

  local rebuild_script="${EVAL_PACKAGE_ROOT}/scripts/rebuild_curobo_b200_in_place.sh"
  require_file "B200 CuRobo rebuild script" "$rebuild_script"
  echo "B200 CuRobo preflight failed; rebuilding sm_100 extensions"
  run_as_root env RMBENCH_BUILD_GPU="$EVALUATOR_GPU" bash "$rebuild_script"
  verify_b200_curobo_runtime || {
    echo "B200 CuRobo rebuild did not produce a usable simulator runtime" >&2
    exit 1
  }
}

ensure_b200_curobo_runtime

DINO_TRANSFORMERS_DEPS="${WORK_ROOT}/dino_transformers_deps"
WANDB_DEPS="${WORK_ROOT}/wandb_deps"
UV_BIN="${RMBENCH_UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
  if [[ -x /root/.local/bin/uv ]]; then
    UV_BIN=/root/.local/bin/uv
  elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
  else
    UV_BIN="$(command -v uv || true)"
  fi
fi
if (( MEMORY_ENABLED )) && ! PYTHONPATH="${DINO_TRANSFORMERS_DEPS}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$POLICY_PY" -c \
  'from transformers import AutoImageProcessor, AutoModel; import sys; AutoImageProcessor.from_pretrained(sys.argv[1], local_files_only=True); AutoModel.from_pretrained(sys.argv[1], local_files_only=True)' \
  "$DINO_ROOT"; then
  [[ -x "$UV_BIN" ]] || {
    echo "uv is required to bootstrap DINOv3 Transformers support: $UV_BIN" >&2
    exit 1
  }
  mkdir -p "$DINO_TRANSFORMERS_DEPS"
  "$UV_BIN" --no-config pip install --index-url https://pypi.org/simple \
    --python "$POLICY_PY" --target "$DINO_TRANSFORMERS_DEPS" \
    "transformers==5.15.0" "tokenizers==0.22.2"
  PYTHONPATH="${DINO_TRANSFORMERS_DEPS}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$POLICY_PY" -c \
    'from transformers import AutoImageProcessor, AutoModel; import sys; AutoImageProcessor.from_pretrained(sys.argv[1], local_files_only=True); AutoModel.from_pretrained(sys.argv[1], local_files_only=True)' \
    "$DINO_ROOT"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  [[ -x "$UV_BIN" ]] || {
    echo "uv is required to bootstrap W&B telemetry: $UV_BIN" >&2
    exit 1
  }
  mkdir -p "$WANDB_DEPS"
  "$UV_BIN" --no-config pip install --index-url https://pypi.org/simple \
    --python "$POLICY_PY" --target "$WANDB_DEPS" \
    "wandb>=0.27,<0.28"
fi

"$POLICY_PY" - "$POLICY_GPU" "$EVALUATOR_GPU" <<'PY'
import subprocess
import sys

rows = [
    [column.strip() for column in line.split(",", 1)]
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if line.strip()
]
requested_gpu_indices = [int(index) for index in sys.argv[1:]]
available_gpu_indices = {int(row[0]) for row in rows}
missing_gpu_indices = [
    str(index) for index in requested_gpu_indices if index not in available_gpu_indices
]
if missing_gpu_indices:
    raise SystemExit(
        "MolmoAct2 RMBench smoke worker requires physical GPU indices "
        + ", ".join(missing_gpu_indices)
    )
PY

PYTHONPATH="${EVAL_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$POLICY_PY" -m rmbench_eval.runtime.patch_policy_client \
    --source "$CLIENT_SOURCE" \
    --output "$CLIENT"

cat >"${OUTPUT_ROOT}/evaluator_args.json" <<EOF
{
  "task_name": "${TASK}",
  "task_config": "demo_clean",
  "ckpt_setting": "molmoact2_stage2",
  "policy_name": "rmbench_molmoact2_adapter",
  "instruction_type": "unseen",
  "seed": 0,
  "port": ${PORT},
  "execution_horizon": ${EXECUTION_HORIZON},
  "start_seed": ${START_SEED},
  "end_seed": null,
  "test_num": ${trial_count},
  "result_dir": "${OUTPUT_ROOT}"
}
EOF

POLICY_SERVER_CONFIG="${OUTPUT_ROOT}/policy_server_config.json"
"$POLICY_PY" - "$POLICY_SERVER_CONFIG" "$CHECKPOINT_ROOT" "$STAGE2_ROOT" \
  "$FEATURE_ROOT" "$DINO_ROOT" "$PI05_ROOT" "$FLOW_SEED" "$MEMORY_ENABLED" <<'PY'
import json
from pathlib import Path
import sys

(
    output,
    checkpoint,
    stage2_root,
    feature_root,
    dino_model_path,
    stage1_source_root,
    flow_seed,
    memory_enabled,
) = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "checkpoint": checkpoint,
            "stage2_root": stage2_root,
            "feature_root": feature_root,
            "dino_model_path": dino_model_path,
            "stage1_source_root": stage1_source_root,
            "memory_enabled": bool(int(memory_enabled)),
            "prediction_horizon": 50,
            "flow_seed": int(flow_seed),
        }
    ),
    encoding="utf-8",
)
PY

"$POLICY_PY" - "${OUTPUT_ROOT}/runtime_contract.json" \
  "${OUTPUT_ROOT}/evaluator_args.json" "$POLICY_SERVER_CONFIG" \
  "$EXECUTION_HORIZON" "$FLOW_SEED" "$START_SEED" "$PREPARE_ONLY" <<'PY'
import json
from pathlib import Path
import sys

(
    output,
    evaluator_args_path,
    policy_config_path,
    execution_horizon,
    flow_seed,
    start_seed,
    prepare_only,
) = sys.argv[1:]
evaluator_args = json.loads(Path(evaluator_args_path).read_text(encoding="utf-8"))
policy_config = json.loads(Path(policy_config_path).read_text(encoding="utf-8"))
if evaluator_args["execution_horizon"] != int(execution_horizon):
    raise SystemExit("evaluator execution horizon disagrees with worker environment")
if policy_config["flow_seed"] != int(flow_seed):
    raise SystemExit("policy flow seed disagrees with worker environment")
if evaluator_args["start_seed"] != int(start_seed):
    raise SystemExit("evaluator start seed disagrees with worker environment")
Path(output).write_text(
    json.dumps(
        {
            "effective_execution_horizon": int(execution_horizon),
            "effective_flow_seed": int(flow_seed),
            "effective_start_seed": int(start_seed),
            "policy_prediction_horizon": policy_config["prediction_horizon"],
            "prepare_only": bool(int(prepare_only)),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

if (( PREPARE_ONLY )); then
  printf 'MolmoAct2 RMBench smoke prepared: %s\n' "$OUTPUT_ROOT"
  exit 0
fi

ensure_ffmpeg

if [[ -z "$WANDB_RUN_NAME" ]]; then
  WANDB_RUN_NAME="imprint-molmoact2-${TASK}-eval"
fi
wandb_done="${OUTPUT_ROOT}/wandb.done"

start_wandb_logger() {
  [[ -n "${WANDB_API_KEY:-}" ]] || return 0
  PYTHONPATH="${WANDB_DEPS}:${EXPERIMENTS_ROOT}" \
    "$POLICY_PY" - "$OUTPUT_ROOT" "$wandb_done" "$TASK" "$trial_count" \
      "$START_SEED" "$RESULT_HF_REPO" "$RESULT_HF_PATH" \
      "$WANDB_RUN_NAME" "$MEMORY_ENABLED" <<'PY' \
      >"${OUTPUT_ROOT}/wandb.log" 2>&1 &
import json
import os
from pathlib import Path
import sys
import time

output = Path(sys.argv[1])
done = Path(sys.argv[2])
task = sys.argv[3]
trial_count = int(sys.argv[4])
start_seed = int(sys.argv[5])
artifact_hf_repo = sys.argv[6]
artifact_hf_path = sys.argv[7]
run_name = sys.argv[8]
memory_enabled = bool(int(sys.argv[9]))
visual_history = "online_wan_and_dinov3" if memory_enabled else "none"
language_source = "stored_pi05_embeddings" if memory_enabled else "molmoact2_prompt"

import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "eval"),
    entity=os.environ.get("WANDB_ENTITY"),
    name=run_name,
    job_type="rmbench-evaluation",
    tags=[
        "rmbench",
        "molmoact2",
        "memory" if memory_enabled else "no-memory",
        task,
    ],
    config={
        "task": task,
        "trial_count": trial_count,
        "seed_start": start_seed,
        "prediction_horizon": 50,
        "execution_horizon": 50,
        "flow_seed": 0,
        "memory_enabled": memory_enabled,
        "visual_history": visual_history,
        "language_source": language_source,
        "artifact_hf_repo": artifact_hf_repo,
        "artifact_hf_path": artifact_hf_path,
    },
)

seen = 0
while True:
    rows = []
    trials = output / "trials.jsonl"
    if trials.is_file():
        rows = [
            json.loads(line)
            for line in trials.read_text(encoding="utf-8").splitlines()
            if line
        ]
    for index, row in enumerate(rows[seen:], start=seen):
        videos = {}
        for view in (
            "head_camera",
            "left_camera",
            "right_camera",
            "third_person",
            "mosaic",
        ):
            path = output / "videos" / f"trial_{index:03d}_{view}.mp4"
            if not path.is_file() or path.stat().st_size < 1:
                break
            videos[f"eval/videos/{view}"] = wandb.Video(str(path), format="mp4")
        else:
            completed = index + 1
            successes = sum(item.get("success") is True for item in rows[:completed])
            run.log(
                {
                    "eval/completed_trials": completed,
                    "eval/successes": successes,
                    "eval/success_rate": successes / completed,
                    "eval/last_seed": row.get("seed"),
                    "eval/last_success": int(row.get("success") is True),
                    **videos,
                },
                step=completed,
            )
            seen = completed
            continue
        break
    if done.exists():
        run.log(
            {
                "eval/completed_trials": seen,
                "eval/successes": sum(item.get("success") is True for item in rows[:seen]),
                "eval/terminal_state": "COMPLETE",
            },
            step=seen,
        )
        run.finish()
        break
    time.sleep(2)
PY
  wandb_pid=$!
}

server_env=(
  "CUDA_VISIBLE_DEVICES=${POLICY_GPU}"
  "PYTHONUNBUFFERED=1"
  "HF_HOME=${HF_HOME}"
  "EVAL_CONSOLIDATE_ALL=1"
)
server_pythonpath="${MOLMOACT2_SOURCE}/experiments:${MOLMOACT2_SOURCE}/experiments/lerobot/src:${MOLMOACT2_SOURCE}/experiments/scripts"
if (( MEMORY_ENABLED )); then
  server_env+=("MEMFT_DINOV3_REQUIRED=1")
  server_pythonpath="${DINO_TRANSFORMERS_DEPS}:${server_pythonpath}:${PI05_ROOT}"
fi
server_env+=("PYTHONPATH=${server_pythonpath}")
setsid env -i "PATH=$PATH" "HOME=$HOME" "${server_env[@]}" \
  "$POLICY_PY" -u "$MOLMOACT2_SOURCE/experiments/scripts/serve_rmbench_molmoact2_tcp.py" \
  --config "$POLICY_SERVER_CONFIG" \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"${OUTPUT_ROOT}/policy.log" 2>&1 &
policy_pid=$!

for _ in $(seq 1 180); do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$policy_pid" 2>/dev/null; then
    cat "${OUTPUT_ROOT}/policy.log" >&2
    exit 1
  fi
  sleep 2
done
timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null || {
  echo "MolmoAct2 policy server did not become ready" >&2
  exit 1
}
start_wandb_logger

oidn_library="${OIDN_ROOT}/oidn/lib"
require_dir "OIDN runtime" "$oidn_library"
export OIDN233_ENV_ROOT="$OIDN_ROOT"
export LD_LIBRARY_PATH="${oidn_library}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${oidn_library}/libOpenImageDenoise_core.so.2.3.3:${oidn_library}/libOpenImageDenoise.so.2.3.3:${oidn_library}/libOpenImageDenoise_device_cuda.so.2.3.3${LD_PRELOAD:+:${LD_PRELOAD}}"
CUDA_VISIBLE_DEVICES="$EVALUATOR_GPU" \
  "$SIM_PY" "$RUNTIME_ROOT/package/eval/rmbench/runtime/verify_oidn233.py"

evaluator_env=(
  "CUDA_VISIBLE_DEVICES=${EVALUATOR_GPU}"
  "PYTHONUNBUFFERED=1"
  "XDG_CACHE_HOME=${SIMULATOR_CACHE_ROOT}"
  "WARP_CACHE_PATH=${SIMULATOR_CACHE_ROOT}/warp"
  "TORCH_EXTENSIONS_DIR=${SIMULATOR_CACHE_ROOT}/torch_extensions"
  "RMBENCH_RENDER_DEVICE=cuda:0"
  "RMBENCH_EXECUTION_HORIZON=${EXECUTION_HORIZON}"
  "RMBENCH_POLICY_RPC_TIMEOUT=${RMBENCH_POLICY_RPC_TIMEOUT:-180}"
  "OIDN233_ENV_ROOT=${OIDN_ROOT}"
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  "LD_PRELOAD=${LD_PRELOAD}"
  "RMBENCH_TASK_CONFIG_PATH=${RMBENCH_ROOT}/task_config/_camera_config.yml"
  "PYTHONPATH=${EVAL_PYTHONPATH}:${RMBENCH_ROOT}/script"
)
(
  cd "$RMBENCH_ROOT"
  env -i "PATH=$PATH" "HOME=$HOME" "${evaluator_env[@]}" \
    "$SIM_PY" -u "$OVERLAY" \
    --canonical-eval-policy "$CLIENT" \
    --canonical-working-directory "$RMBENCH_ROOT" \
    --user-args-json "${OUTPUT_ROOT}/evaluator_args.json" \
    --video-output "$OUTPUT_ROOT" \
    --task_name "$TASK" \
    --execution_horizon "$EXECUTION_HORIZON" \
    --start_seed "$START_SEED"
) >"${OUTPUT_ROOT}/evaluator.log" 2>&1

"$POLICY_PY" - "$OUTPUT_ROOT/trials.jsonl" "$trial_count" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"official RMBench client did not write {path}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != expected:
    raise SystemExit(f"expected {expected} completed smoke trial, found {len(rows)}")
PY

publish_results
printf 'MolmoAct2 RMBench smoke completed: %s\n' "$OUTPUT_ROOT"
