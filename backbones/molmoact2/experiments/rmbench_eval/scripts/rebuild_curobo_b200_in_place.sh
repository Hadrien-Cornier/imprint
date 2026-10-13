#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RMBENCH_RUNTIME_ROOT:-/opt/imprint/rmbench/runtime}"
OIDN_ROOT="${RMBENCH_OIDN_ROOT:-/opt/imprint/rmbench/oidn233}"
BUILD_GPU="${RMBENCH_BUILD_GPU:-0}"
CUDA_TOOLKIT_VERSION="12-8"

[[ "$BUILD_GPU" =~ ^[0-9]+$ ]] || {
  echo "RMBENCH_BUILD_GPU must be a non-negative GPU index" >&2
  exit 2
}

if ! command -v nvcc >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    ca-certificates wget build-essential
  wget -q -O /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  # A minimal CUDA image can mark the CUDA source-list conffile as
  # removed. Force its restoration when reinstalling the NVIDIA keyring.
  dpkg --force-confmiss -i /tmp/cuda-keyring.deb
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "cuda-toolkit-${CUDA_TOOLKIT_VERSION}"
fi

export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="$BUILD_GPU"

TARGET_PYTHON="${OIDN_ROOT}/.venv/bin/python"
CUROBO_ROOT="${OIDN_ROOT}/curobo-v0.7.8"
CUROBO_LIB="${CUROBO_ROOT}/src/curobo/curobolib"
test -x "$TARGET_PYTHON"
test -f "${CUROBO_ROOT}/setup.py"
test -x "${CUDA_HOME}/bin/nvcc"

"$TARGET_PYTHON" - <<'PY'
import torch

if torch.cuda.get_device_capability() != (10, 0):
    raise SystemExit(
        f"expected B200 capability (10, 0), got {torch.cuda.get_device_capability()}"
    )
print(f"building_curobo_for={torch.cuda.get_device_name()} capability=(10, 0)")
PY

rm -rf "${CUROBO_ROOT}/build"
find "$CUROBO_LIB" -maxdepth 1 -type f \
  -name '*_cu.cpython-310-x86_64-linux-gnu.so' -delete
(
  cd "$CUROBO_ROOT"
  "$TARGET_PYTHON" setup.py build_ext --inplace
)

expected_extensions=(
  "lbfgs_step_cu.cpython-310-x86_64-linux-gnu.so"
  "kinematics_fused_cu.cpython-310-x86_64-linux-gnu.so"
  "line_search_cu.cpython-310-x86_64-linux-gnu.so"
  "tensor_step_cu.cpython-310-x86_64-linux-gnu.so"
  "geom_cu.cpython-310-x86_64-linux-gnu.so"
)
for extension in "${expected_extensions[@]}"; do
  path="${CUROBO_LIB}/${extension}"
  test -s "$path"
  grep -aFq "sm_100" "$path"
done

export PYTHONPATH="${RUNTIME_ROOT}/RMBench:${CUROBO_ROOT}/src"
"$TARGET_PYTHON" - "$RUNTIME_ROOT/RMBench" <<'PY'
from curobo.wrap.reacher.motion_gen import MotionGenConfig
from pathlib import Path
import sys

config = MotionGenConfig.load_from_robot_config(
    str(
        Path(sys.argv[1])
        / "assets/embodiments/aloha-agilex/curobo_left.yml"
    ),
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
