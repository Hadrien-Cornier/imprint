#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${IMPRINT_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in "${ROOT}/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]] && command -v uv >/dev/null; then
  PYTHON="$(uv python find 3.11)"
fi
if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "Python 3.11 or newer is required; set IMPRINT_PYTHON to a compatible interpreter." >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m compileall -q \
  "$ROOT/src" "$ROOT/scripts" "$ROOT/benchmarks"

while IFS= read -r script; do
  bash -n "$script"
done < <(find "$ROOT/scripts/reproduce" "$ROOT/benchmarks" -type f -name '*.sh' | sort)

(
  unset XLA_FLAGS CUBLAS_WORKSPACE_CONFIG PYTHONHASHSEED
  # shellcheck source=configure_training_runtime.sh
  source "$ROOT/scripts/reproduce/configure_training_runtime.sh"
  IMPRINT_DETERMINISTIC_TRAINING=1 imprint_configure_training_runtime >/dev/null
  test "$PYTHONHASHSEED" = "0"
  test "$CUBLAS_WORKSPACE_CONFIG" = ":4096:8"
  [[ "$XLA_FLAGS" == *"--xla_gpu_deterministic_ops=true"* ]]
  [[ "$XLA_FLAGS" == *"--xla_gpu_exclude_nondeterministic_ops=true"* ]]
)

"$PYTHON" - "$ROOT" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
credential_patterns = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)
private_uri = "s3" + "://"
private_patterns = (
    (
        "account-specific cloud endpoint",
        re.compile(r"\b\d{12}\.(?:dkr\.ecr|d\.codeartifact)\.[A-Za-z0-9.-]+"),
    ),
    (
        "account-specific IAM ARN",
        re.compile(r"arn:" + r"aws:[^:\s]+:[^:\s]*:\d{12}:[^\s\"']+"),
    ),
    (
        "internal service URL",
        re.compile(r"https?://[^/\s\"']*(?:\.internal\b|-internal\b|\.wandb\.io\b)"),
    ),
    (
        "internal infrastructure identifier",
        re.compile(r"\bintern-[A-Za-z0-9-]+\b"),
    ),
    (
        "machine-specific data path",
        re.compile(r"(?<![A-Za-z0-9_.-])/(?:data|nfs|fsx)/[A-Za-z0-9_.-]+"),
    ),
    (
        "personal workstation path",
        re.compile(r"(?<![A-Za-z0-9_.-])/Users/[A-Za-z0-9_.-]+"),
    ),
    (
        "private experiment provenance",
        re.compile(r"(?m)^\s*" + "wandb" + r"_run\s*:"),
    ),
)
ignored_parts = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
for path in root.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(root)
    if any(part in ignored_parts for part in relative.parts):
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in credential_patterns:
        if pattern.search(text):
            raise SystemExit(f"credential-like text found in {path}")
    if private_uri in text:
        raise SystemExit(f"private object-storage URI found in release file: {path}")
    for label, pattern in private_patterns:
        if pattern.search(text):
            raise SystemExit(f"{label} found in release file: {path}")
print("RELEASE_SCAN_OK")
PY

if ! "$PYTHON" -c 'import mme_vla_suite.training.config; import openpi.training.config' >/dev/null 2>&1; then
  echo "Runtime dependencies are unavailable; run 'uv sync --python 3.11 --frozen' in the Linux GPU environment." >&2
  exit 1
fi
"$PYTHON" "$ROOT/scripts/reproduce/validate_release_profiles.py"
