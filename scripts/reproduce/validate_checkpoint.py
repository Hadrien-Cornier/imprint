#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--architecture", choices=("perceiver_v1", "gated_delta_hybrid_v1"), required=True)
    parser.add_argument("--num-latents", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.stage1, allow_pickle=True) as checkpoint:
        if "metadata/memory_architecture" in checkpoint.files:
            architecture = str(np.asarray(checkpoint["metadata/memory_architecture"]).item())
        else:
            architecture = (
                "gated_delta_hybrid_v1"
                if any(key.startswith("bottleneck/delta_blocks/") for key in checkpoint.files)
                else "perceiver_v1"
            )
        if architecture != args.architecture:
            raise ValueError(f"Stage-1 architecture {architecture!r} != {args.architecture!r}")
        latents = checkpoint["bottleneck/latents"]
        if latents.shape != (args.num_latents, 1024):
            raise ValueError(f"unexpected Stage-1 latent shape: {latents.shape}")

    required = (
        args.stage2 / "_CHECKPOINT_METADATA",
        args.stage2 / "params/_METADATA",
        args.stage2 / "params/manifest.ocdbt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-2 checkpoint is incomplete: {missing}")
    print(
        f"CHECKPOINT_VALID architecture={architecture} "
        f"latents={latents.shape} stage2={args.stage2.resolve()}"
    )


if __name__ == "__main__":
    main()
