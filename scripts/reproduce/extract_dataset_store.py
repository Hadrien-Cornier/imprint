#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and extract IMPRINT dataset shards."
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_member(member: tarfile.TarInfo, output: Path) -> None:
    if member.name.startswith("/"):
        raise ValueError(f"absolute archive member: {member.name}")
    target = (output / member.name).resolve()
    if target != output and output not in target.parents:
        raise ValueError(f"archive member escapes output: {member.name}")
    if member.issym() or member.islnk():
        link = Path(member.linkname)
        if link.is_absolute() or ".." in link.parts:
            raise ValueError(f"unsafe archive link: {member.name} -> {member.linkname}")


def main() -> None:
    args = parse_args()
    archive_root = args.archive_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifests = sorted(archive_root.glob("manifest-*.json"))
    if not manifests:
        raise FileNotFoundError(f"no manifest-*.json under {archive_root}")
    output.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        layout_prefix = manifest["layout_prefix"]
        for shard in manifest["shards"]:
            archive = archive_root / shard["path"]
            marker = (
                output
                / ".imprint_extracted"
                / layout_prefix
                / f"{archive.name}.json"
            )
            if args.skip_existing and marker.is_file():
                saved = json.loads(marker.read_text())
                if saved.get("sha256") == shard["sha256"]:
                    continue
            if archive.stat().st_size != shard["bytes"]:
                raise ValueError(f"size mismatch for {archive}")
            actual_sha256 = sha256(archive)
            if actual_sha256 != shard["sha256"]:
                raise ValueError(f"SHA256 mismatch for {archive}")
            with tarfile.open(archive, "r:") as tar:
                for member in tar:
                    validate_member(member, output)
                    tar.extract(member, output)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {"archive": shard["path"], "sha256": shard["sha256"]},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            extracted += 1
    print(f"STORE_EXTRACTED shards={extracted} output={output}")


if __name__ == "__main__":
    main()
