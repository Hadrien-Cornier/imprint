#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${MOLMOACT2_REPOSITORY:-https://github.com/amazon-far/imprint.git}"
BRANCH="${MOLMOACT2_BRANCH:-molmoact2}"
CHECKOUT_DIR="${MOLMOACT2_CHECKOUT_DIR:-${PWD}/imprint-molmoact2}"
REVISION="${MOLMOACT2_REVISION:-}"
STAGE1_REPO="${MEMORY_STAGE1_CHECKPOINT_REPO:-Amazon-FAR/imprint_models}"
STAGE1_REVISION="${MEMORY_STAGE1_CHECKPOINT_REVISION:-main}"
STAGE1_FILE="${MEMORY_STAGE1_CHECKPOINT_FILE:-simulation/rmbench/full_attention/stage1/stage1_step160000.npz}"
STAGE1_PATH="${MEMORY_STAGE1_CHECKPOINT:-${CHECKOUT_DIR}/checkpoints/stage1_step160000.npz}"
INSTALL_EVAL_RUNTIME=0

usage() {
    cat <<'EOF' >&2
usage: bootstrap_molmoact2.sh [--revision SHA] [--install-eval-runtime]

Clones the IMPRINT MolmoAct2 branch, initializes required submodules,
and stages the verified Stage-1 memory checkpoint.
EOF
    exit 2
}

while (($#)); do
    case "$1" in
        --revision) REVISION="${2:-}"; shift 2 ;;
        --install-eval-runtime) INSTALL_EVAL_RUNTIME=1; shift ;;
        *) usage ;;
    esac
done

for command in git hf sha256sum; do
    command -v "$command" >/dev/null || {
        echo "$command is required." >&2
        exit 127
    }
done

if ! git ls-remote "$REPOSITORY" "refs/heads/${BRANCH}" >/dev/null; then
    cat <<'EOF' >&2
GitHub access preflight failed. Configure Git authentication for the
repository, verify that `git ls-remote` succeeds, and rerun this script.
EOF
    exit 1
fi

if [[ -e "$CHECKOUT_DIR" ]]; then
    git -C "$CHECKOUT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        echo "Checkout path exists but is not a Git repository: $CHECKOUT_DIR" >&2
        exit 2
    }
    git -C "$CHECKOUT_DIR" fetch origin "$BRANCH"
else
    git clone --branch "$BRANCH" --single-branch "$REPOSITORY" "$CHECKOUT_DIR"
fi

if [[ -n "$REVISION" ]]; then
    git -C "$CHECKOUT_DIR" checkout --detach "$REVISION"
else
    git -C "$CHECKOUT_DIR" checkout "$BRANCH"
fi
# Initialize only declared top-level submodules required by MolmoAct2.
git -C "$CHECKOUT_DIR" submodule update --init

mkdir -p "$(dirname "$STAGE1_PATH")"
downloaded_stage1="$(
    hf download "$STAGE1_REPO" "$STAGE1_FILE" \
        --revision "$STAGE1_REVISION" \
        --quiet
)"
cp "$downloaded_stage1" "$STAGE1_PATH"
echo "164c94f3ff8fcd84aee61e88ec0feb97faf99d203eac1be0f4b4146d464d2044  $STAGE1_PATH" \
    | sha256sum --check

if (( INSTALL_EVAL_RUNTIME )); then
    [[ -n "${RMBENCH_RUNTIME_ARCHIVE_DIR:-}" ]] || {
        echo "Set RMBENCH_RUNTIME_ARCHIVE_DIR before using --install-eval-runtime." >&2
        exit 2
    }
    sudo -E \
        "${CHECKOUT_DIR}/backbones/molmoact2/experiments/scripts/install_rmbench_molmoact2_runtime.sh"
fi

printf 'MolmoAct2 checkout ready: %s (%s)\n' \
    "$CHECKOUT_DIR" "$(git -C "$CHECKOUT_DIR" rev-parse HEAD)"
