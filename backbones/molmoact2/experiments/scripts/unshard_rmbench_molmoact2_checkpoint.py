#!/usr/bin/env python3
"""Convert a distributed MolmoAct2 checkpoint into a model-only release."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from olmo.train.distributed_checkpointing import unshard_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory containing config.yaml and model_and_optim/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source = checkpoint / "model_and_optim"
    config = checkpoint / "config.yaml"

    if not source.is_dir():
        raise SystemExit(f"Missing distributed model state: {source}")
    if not config.is_file():
        raise SystemExit(f"Missing checkpoint configuration: {config}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output}")

    output.mkdir(parents=True, exist_ok=True)
    unshard_checkpoint(
        source,
        output,
        optim=False,
        save_overwrite=args.overwrite,
        quiet=False,
    )
    shutil.copy2(config, output / "config.yaml")

    optimizer_paths = [
        output / "optim.pt",
        output / "train",
        output / "model_and_optim",
    ]
    if any(path.exists() for path in optimizer_paths):
        raise SystemExit("Model-only conversion unexpectedly produced optimizer state.")
    print(output / "model.pt")


if __name__ == "__main__":
    main()
