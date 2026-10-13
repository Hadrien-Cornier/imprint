#!/usr/bin/env bash
set -Eeuo pipefail

ARCHIVE="${1:-}"
DESTINATION="${2:-}"
EXPECTED_SHA256="12cb7a92087298e6e9f299070d294655e54f1340ef578ca5e0581c885a79d9fc"

if [[ ! -f "$ARCHIVE" || -z "$DESTINATION" ]]; then
  echo "usage: $0 /path/to/robocasa-memory-setup.tar.gz /path/to/destination" >&2
  exit 2
fi
if [[ -e "$DESTINATION" && -n "$(find "$DESTINATION" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite nonempty destination: $DESTINATION" >&2
  exit 3
fi
command -v uv >/dev/null || {
  echo "uv is required" >&2
  exit 2
}

ACTUAL_SHA256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "archive SHA256 mismatch: $ACTUAL_SHA256" >&2
  exit 4
fi

mkdir -p "$DESTINATION"
tar -xzf "$ARCHIVE" -C "$DESTINATION"
DESTINATION="$(cd "$DESTINATION" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DESTINATION/.uv-cache}"
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

"${UV_PUBLIC[@]}" venv \
  --seed \
  --no-project \
  --python 3.10 \
  "$DESTINATION/.venv-mem"

PYTHON="$DESTINATION/.venv-mem/bin/python"
INSTALL_REQUIREMENTS="$DESTINATION/requirements-venv-mem.install.txt"
PIP_PUBLIC=(
  env
  -u UV_INDEX
  -u UV_INDEX_URL
  -u UV_DEFAULT_INDEX
  -u UV_EXTRA_INDEX_URL
  -u PIP_INDEX_URL
  -u PIP_EXTRA_INDEX_URL
  PIP_CONFIG_FILE=/dev/null
  "$PYTHON"
  -m
  pip
)

# The captured simulator environment contains editable VCS requirements.
# pip supports those requirements; uv pip intentionally rejects them.
# The archived tree is the patched RoboCasa source, so do not fetch the
# redundant upstream RoboCasa editable whose historical commit may disappear.
grep -vE \
  '^-e git\+https://github\.com/robocasa/robocasa(\.git)?@' \
  "$DESTINATION/requirements-venv-mem.freeze.txt" \
  > "$INSTALL_REQUIREMENTS"
"${PIP_PUBLIC[@]}" install \
  --index-url "$PYPI_INDEX" \
  -r "$INSTALL_REQUIREMENTS"
"${PIP_PUBLIC[@]}" install \
  --no-deps \
  -e "$DESTINATION"

# The source release intentionally excludes the approximately 10 GB RoboCasa
# asset bundles. The upstream downloader has one confirmation prompt.
printf 'y\n' | PYTHONPATH="$DESTINATION" "$PYTHON" \
  -m robocasa.scripts.download_kitchen_assets --type all

PYTHONPATH="$DESTINATION" "$PYTHON" - "$DESTINATION" <<'PY'
import pathlib
import sys

assets = pathlib.Path(sys.argv[1]) / "robocasa" / "models" / "assets"
required = (
    assets / "textures",
    assets / "generative_textures",
    assets / "objects" / "objaverse",
    assets / "objects" / "aigen_objs",
    assets / "objects" / "lightwheel",
)
missing = [str(path) for path in required if not path.is_dir() or not any(path.iterdir())]
if missing:
    raise SystemExit(f"RoboCasa asset download is incomplete: {missing}")
print("ROBOCASA_STANDARD_ASSETS_READY")
PY

LEGACY_SOURCE="$DESTINATION/legacy_scene_source"
for kind in kitchen_layouts kitchen_styles; do
  [[ -d "$LEGACY_SOURCE/robocasa/models/assets/scenes/$kind" ]] || {
    echo "archive is missing bundled legacy scene metadata: $kind" >&2
    exit 5
  }
done

PYTHONPATH="$DESTINATION" "$PYTHON" - "$DESTINATION" "$LEGACY_SOURCE" <<'PY'
import pathlib
import sys

from scripts.replay.fetch_legacy_assets import fetch_legacy_assets

destination = pathlib.Path(sys.argv[1])
legacy_source = pathlib.Path(sys.argv[2])
result = fetch_legacy_assets(str(destination), str(legacy_source))
required = (
    destination
    / "robocasa"
    / "models"
    / "assets"
    / "scenes"
    / "kitchen_styles"
    / "legacy"
    / "transitional_1.yaml"
)
if not required.is_file():
    raise SystemExit(f"legacy asset setup did not create {required}")
print(f"ROBOCASA_LEGACY_ASSETS_READY {result}")
PY

echo "ROBOCASA_ENV_READY root=$DESTINATION python=$PYTHON"
