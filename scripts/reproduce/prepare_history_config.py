#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path)
    parser.add_argument("--latent-cap", type=int, required=True)
    parser.add_argument("--max-recent-tokens", type=int, required=True)
    parser.add_argument("--max-step-tokens", type=int, required=True)
    parser.add_argument("--num-latents", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.source)
    memory = config.cosmo_bottleneck_memory

    expected = {
        "latent_cap": args.latent_cap,
        "max_recent_tokens": args.max_recent_tokens,
        "max_step_tokens": args.max_step_tokens,
        "num_latents": args.num_latents,
    }
    actual = {key: int(memory[key]) for key in expected}
    if actual != expected:
        raise ValueError(f"history config caps do not match release profile: {actual} != {expected}")

    if args.feature_store is not None:
        feature_store = args.feature_store.expanduser().resolve()
        if not (feature_store / "features").is_dir():
            raise FileNotFoundError(f"feature store has no features directory: {feature_store}")
        memory.feature_store_path = str(feature_store)
    elif "feature_store_path" in memory:
        del memory["feature_store_path"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
