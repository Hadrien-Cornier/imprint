#!/usr/bin/env bash

_imprint_append_xla_flag() {
  local flag="$1"
  if [[ " ${XLA_FLAGS:-} " != *" ${flag} "* ]]; then
    XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }${flag}"
  fi
}

imprint_configure_training_runtime() {
  local deterministic="${IMPRINT_DETERMINISTIC_TRAINING:-1}"
  if [[ "$deterministic" != "0" && "$deterministic" != "1" ]]; then
    echo "IMPRINT_DETERMINISTIC_TRAINING must be 0 or 1" >&2
    return 2
  fi

  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

  if [[ "$deterministic" == "1" ]]; then
    if [[ " ${XLA_FLAGS:-} " == *" --xla_gpu_deterministic_ops=false "* ]] ||
      [[ " ${XLA_FLAGS:-} " == *" --xla_gpu_exclude_nondeterministic_ops=false "* ]]; then
      echo "XLA_FLAGS disables deterministic GPU operations" >&2
      return 2
    fi
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    _imprint_append_xla_flag "--xla_gpu_deterministic_ops=true"
    _imprint_append_xla_flag "--xla_gpu_exclude_nondeterministic_ops=true"
    export XLA_FLAGS
  fi

  echo "IMPRINT deterministic training: ${deterministic}"
  if [[ "$deterministic" == "1" ]]; then
    echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
    echo "XLA_FLAGS=${XLA_FLAGS}"
  fi
}
