#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Optional
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class Shard:
    path: str
    kind: str
    entries: int
    bytes: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an immutable IMPRINT store into deterministic tar shards."
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-prefix", required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--data-files-per-shard", type=int, default=10_000)
    parser.add_argument("--feature-episodes-per-shard", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def numeric_suffix(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[1]), path.name
    except ValueError:
        return 2**63 - 1, path.name


def chunks(values: list[T], size: int) -> list[list[T]]:
    if size < 1:
        raise ValueError("shard size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_gnu_tar() -> Optional[str]:
    for candidate in ("gtar", "tar"):
        executable = shutil.which(candidate)
        if executable is None:
            continue
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if "GNU tar" in result.stdout:
            return executable
    return None


def iter_archive_paths(store: Path, entries: list[str]) -> list[Path]:
    paths: list[Path] = []
    for entry in entries:
        root = store / entry
        paths.append(root)
        if root.is_dir() and not root.is_symlink():
            paths.extend(sorted(root.rglob("*"), key=lambda path: str(path.relative_to(store))))
    return paths


def write_python_tar(
    store: Path,
    output: Path,
    layout_prefix: str,
    entries: list[str],
) -> None:
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT, dereference=False) as archive:
        for source in iter_archive_paths(store, entries):
            relative = source.relative_to(store)
            info = archive.gettarinfo(
                str(source),
                arcname=str(Path(layout_prefix) / relative),
            )
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.pax_headers = {}
            if info.isfile():
                with source.open("rb") as file:
                    archive.addfile(info, file)
            else:
                archive.addfile(info)


def write_gnu_tar(
    executable: str,
    store: Path,
    output: Path,
    layout_prefix: str,
    entries: list[str],
) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as list_file:
        list_file.write("\n".join(entries))
        list_file.write("\n")
        list_path = Path(list_file.name)
    try:
        command = [
            executable,
            "--create",
            "--file",
            str(output),
            "--directory",
            str(store),
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--format=posix",
            "--pax-option=delete=atime,delete=ctime",
            f"--transform=s,^,{layout_prefix}/,",
            "--files-from",
            str(list_path),
        ]
        subprocess.run(command, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def shard_specs(
    store: Path,
    data_files_per_shard: int,
    feature_episodes_per_shard: int,
) -> list[tuple[str, str, list[str]]]:
    specs: list[tuple[str, str, list[str]]] = []
    data = store / "data"
    if data.is_dir():
        files = sorted(
            str(path.relative_to(store))
            for path in data.rglob("*")
            if path.is_file() or path.is_symlink()
        )
        for index, group in enumerate(chunks(files, data_files_per_shard)):
            specs.append((f"data-{index:05d}.tar", "data", group))

    features = store / "features"
    if features.is_dir():
        episodes = sorted(
            (path for path in features.iterdir() if path.is_dir()),
            key=numeric_suffix,
        )
        relative = [str(path.relative_to(store)) for path in episodes]
        for index, group in enumerate(chunks(relative, feature_episodes_per_shard)):
            specs.append((f"features-{index:05d}.tar", "features", group))

    meta = store / "meta"
    if meta.exists():
        specs.append(("meta.tar", "meta", ["meta"]))

    extras = sorted(
        str(path.relative_to(store))
        for path in store.iterdir()
        if path.name not in {"data", "features", "meta"}
    )
    if extras:
        specs.append(("root.tar", "root", extras))
    return specs


def package_shard(
    store: Path,
    shard_dir: Path,
    layout_prefix: str,
    spec: tuple[str, str, list[str]],
) -> Shard:
    filename, kind, entries = spec
    output = shard_dir / filename
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.is_file() and sidecar.is_file():
        saved = json.loads(sidecar.read_text())
        if output.stat().st_size == saved["bytes"]:
            return Shard(**saved)

    shard_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    try:
        gnu_tar = find_gnu_tar()
        if gnu_tar is None:
            write_python_tar(store, partial, layout_prefix, entries)
        else:
            write_gnu_tar(gnu_tar, store, partial, layout_prefix, entries)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)

    result = Shard(
        path=f"shards/{layout_prefix}/{filename}",
        kind=kind,
        entries=len(entries),
        bytes=output.stat().st_size,
        sha256=sha256(output),
    )
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary_sidecar.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    os.replace(temporary_sidecar, sidecar)
    return result


def main() -> None:
    args = parse_args()
    store = args.store.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not store.is_dir():
        raise FileNotFoundError(store)
    if output == store or store in output.parents:
        raise ValueError("--output must not be inside --store")
    if "/" in args.layout_prefix or args.layout_prefix in {"", ".", ".."}:
        raise ValueError("--layout-prefix must be one directory name")

    specs = shard_specs(
        store,
        args.data_files_per_shard,
        args.feature_episodes_per_shard,
    )
    if not specs:
        raise ValueError(f"no store content found under {store}")

    shard_dir = output / "shards" / args.layout_prefix
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(package_shard, store, shard_dir, args.layout_prefix, spec)
            for spec in specs
        ]
        shards = [future.result() for future in futures]
    shards.sort(key=lambda shard: shard.path)

    manifest = {
        "format_version": 1,
        "archive_format": "tar",
        "layout_prefix": args.layout_prefix,
        "store_id": args.store_id,
        "total_bytes": sum(shard.bytes for shard in shards),
        "shards": [asdict(shard) for shard in shards],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"manifest-{args.layout_prefix}.json"
    temporary_manifest = manifest_path.with_suffix(".json.partial")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(
        f"STORE_PACKAGED id={args.store_id} layout={args.layout_prefix} "
        f"shards={len(shards)} bytes={manifest['total_bytes']} output={output}"
    )


if __name__ == "__main__":
    main()
