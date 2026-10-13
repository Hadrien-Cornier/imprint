#!/usr/bin/env bash
set -Eeuo pipefail

RMBENCH_ROOT="${1:-}"
DESTINATION="${2:-}"

if [[ ! -f "$RMBENCH_ROOT/script/requirements.txt" || -z "$DESTINATION" ]]; then
  echo "usage: $0 /path/to/EventVLA/RoboTwin-Mem /path/to/environment-root" >&2
  exit 2
fi
if [[ -e "$DESTINATION" && -n "$(find "$DESTINATION" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite nonempty destination: $DESTINATION" >&2
  exit 3
fi
command -v git >/dev/null || {
  echo "git is required" >&2
  exit 2
}
command -v uv >/dev/null || {
  echo "uv is required" >&2
  exit 2
}
CUROBO_WHEEL="${IMPRINT_CUROBO_WHEEL:-}"
CUROBO_INSTALL_MODE="prebuilt-wheel"
if [[ -n "$CUROBO_WHEEL" ]]; then
  if [[ ! -f "$CUROBO_WHEEL" ]]; then
    echo "IMPRINT_CUROBO_WHEEL does not exist: $CUROBO_WHEEL" >&2
    exit 2
  fi
  CUROBO_WHEEL="$(cd "$(dirname "$CUROBO_WHEEL")" && pwd)/$(basename "$CUROBO_WHEEL")"
else
  CUROBO_INSTALL_MODE="source-build"
  NVCC="$(command -v nvcc 2>/dev/null || true)"
  if [[ -z "$NVCC" && -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
    NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
  fi
  if [[ -z "$NVCC" ]]; then
    cat >&2 <<'EOF'
RMBench setup requires the full CUDA toolkit, including nvcc, to build CuRobo.
Use a CUDA development image, or set IMPRINT_CUROBO_WHEEL to a compatible
prebuilt CuRobo v0.7.8 wheel.
EOF
    exit 2
  fi
  export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$NVCC")/.." && pwd)}"
fi

RMBENCH_ROOT="$(cd "$RMBENCH_ROOT" && pwd)"
mkdir -p "$DESTINATION/src"
DESTINATION="$(cd "$DESTINATION" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DESTINATION/.uv-cache}"
PYTHON="$DESTINATION/.venv/bin/python"
PYPI_INDEX="https://pypi.org/simple"
UV_PUBLIC=(
  env
  -u UV_INDEX
  -u UV_INDEX_URL
  -u UV_DEFAULT_INDEX
  -u UV_EXTRA_INDEX_URL
  -u PIP_INDEX_URL
  -u PIP_EXTRA_INDEX_URL
  uv
  --no-config
)

"${UV_PUBLIC[@]}" venv --no-project --python 3.10 "$DESTINATION/.venv"
"${UV_PUBLIC[@]}" pip install \
  --default-index "$PYPI_INDEX" \
  --python "$PYTHON" \
  -r "$RMBENCH_ROOT/script/requirements.txt"
"${UV_PUBLIC[@]}" pip install \
  --default-index "$PYPI_INDEX" \
  --python "$PYTHON" \
  setuptools==80.9.0
"${UV_PUBLIC[@]}" pip install \
  --default-index "$PYPI_INDEX" \
  --python "$PYTHON" \
  ninja==1.11.1.4

SAPIEN_ROOT="$(
  "$PYTHON" -c 'import pathlib, sapien; print(pathlib.Path(sapien.__file__).resolve().parent)'
)"
URDF_LOADER="$SAPIEN_ROOT/wrapper/urdf_loader.py"
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
sed -i -E 's/(urdf_file\[:-4\] \+ )"srdf"/\1".srdf"/g' "$URDF_LOADER"

MPLIB_ROOT="$(
  "$PYTHON" -c 'import pathlib, mplib; print(pathlib.Path(mplib.__file__).resolve().parent)'
)"
PLANNER="$MPLIB_ROOT/planner.py"
sed -i -E \
  's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
  "$PLANNER"

if [[ "$CUROBO_INSTALL_MODE" == "prebuilt-wheel" ]]; then
  "${UV_PUBLIC[@]}" pip install \
    --default-index "$PYPI_INDEX" \
    --python "$PYTHON" \
    pybind11 \
    numpy-quaternion \
    "setuptools-scm>=6.2" \
    "yourdfpy>=0.0.53" \
    wheel \
    importlib-resources \
    scikit-image
  "${UV_PUBLIC[@]}" pip install \
    --python "$PYTHON" \
    --no-deps \
    "$CUROBO_WHEEL"
else
  git clone --depth 1 --branch v0.7.8 \
    https://github.com/NVlabs/curobo.git \
    "$DESTINATION/src/curobo"
  CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0+PTX}"
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" "${UV_PUBLIC[@]}" pip install \
    --default-index "$PYPI_INDEX" \
    --python "$PYTHON" \
    --no-build-isolation \
    -e "$DESTINATION/src/curobo"
fi
"${UV_PUBLIC[@]}" pip install \
  --default-index "$PYPI_INDEX" \
  --python "$PYTHON" \
  warp-lang==1.12.0

PYTHONPATH="$RMBENCH_ROOT" "$PYTHON" - <<'PY'
import curobo
import mplib
import open3d
import sapien

print("RMBENCH_IMPORTS_OK")
PY
"${UV_PUBLIC[@]}" pip check --python "$PYTHON"

echo \
  "RMBENCH_ENV_READY root=$RMBENCH_ROOT python=$PYTHON" \
  "curobo=$CUROBO_INSTALL_MODE cuda_home=${CUDA_HOME:-not-required}"
