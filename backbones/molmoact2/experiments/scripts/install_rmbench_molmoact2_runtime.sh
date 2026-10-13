#!/usr/bin/env bash
set -euo pipefail

# Installs a user-provided simulator and Python runtime used by the RMBench
# MolmoAct2 Stage-2 evaluation worker. The archive is root-relative and is not
# redistributed by IMPRINT.
RUNTIME_ARCHIVE_DIR="${RMBENCH_RUNTIME_ARCHIVE_DIR:-}"
STAGE_DIR="${RMBENCH_RUNTIME_STAGE_DIR:-${TMPDIR:-/tmp}/rmbench-molmoact2-runtime}"
EXTRACT_ROOT="${RMBENCH_RUNTIME_EXTRACT_ROOT:-/}"

[[ -n "$RUNTIME_ARCHIVE_DIR" && -d "$RUNTIME_ARCHIVE_DIR" ]] || {
    echo "RMBENCH_RUNTIME_ARCHIVE_DIR must identify the prepared RMBench runtime archive directory." >&2
    exit 2
}
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The RMBench runtime archive is supported only on Linux." >&2
    exit 2
fi

mkdir -p "$STAGE_DIR"
for artifact in runtime.tar uv-python.tar runtime-manifest.json; do
    [[ -f "${RUNTIME_ARCHIVE_DIR}/${artifact}" ]] || {
        echo "Missing runtime artifact: ${RUNTIME_ARCHIVE_DIR}/${artifact}" >&2
        exit 2
    }
    cp "${RUNTIME_ARCHIVE_DIR}/${artifact}" "${STAGE_DIR}/${artifact}"
done

if [[ "$EXTRACT_ROOT" == "/" ]]; then
    if (( EUID != 0 )); then
        echo "Extracting to / requires sudo; rerun with sudo or set RMBENCH_RUNTIME_EXTRACT_ROOT." >&2
        exit 2
    fi
fi
mkdir -p "$EXTRACT_ROOT"
tar --extract --file "${STAGE_DIR}/uv-python.tar" --directory "$EXTRACT_ROOT"
tar --extract --file "${STAGE_DIR}/runtime.tar" --directory "$EXTRACT_ROOT"

printf 'Installed RMBench MolmoAct2 runtime from %s\n' "$RUNTIME_ARCHIVE_DIR"
