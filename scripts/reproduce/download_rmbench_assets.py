#!/usr/bin/env python3
"""Download the pinned RMBench assets and materialize embodiment configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY = "ganlinyang/RoboTwin-MeM"
REVISION = "f67a4ee99a20c65c86897b85d3f5309b205cc897"
ASSET_PATTERNS = ("assets/embodiments/**", "assets/objects/**")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rmbench_root", type=Path, help="Path to EventVLA/RoboTwin-Mem")
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required; run this helper from the IMPRINT root environment"
        ) from error

    root = args.rmbench_root.expanduser().resolve()
    updater = root / "script" / "update_embodiment_config_path.py"
    if not updater.is_file():
        raise SystemExit(f"not an RMBench checkout: {root}")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=list(ASSET_PATTERNS),
        local_dir=root,
        max_workers=args.max_workers,
    )

    embodiments = root / "assets" / "embodiments"
    objects = root / "assets" / "objects"
    templates = sorted(embodiments.rglob("*_tmp.yml"))
    object_files = sum(path.is_file() for path in objects.rglob("*"))
    if not templates or object_files == 0:
        raise SystemExit(
            "asset download is incomplete: expected embodiment templates and object files "
            f"under {root / 'assets'}"
        )

    subprocess.run([sys.executable, str(updater)], cwd=root, check=True)
    generated = sum(
        path.is_file() and not path.name.endswith("_tmp.yml")
        for path in embodiments.rglob("*.yml")
    )
    if generated == 0:
        raise SystemExit("no embodiment configs were generated from the downloaded templates")

    print(
        "RMBENCH_ASSETS_READY "
        f"root={root} revision={REVISION} templates={len(templates)} "
        f"configs={generated} object_files={object_files}"
    )


if __name__ == "__main__":
    main()
