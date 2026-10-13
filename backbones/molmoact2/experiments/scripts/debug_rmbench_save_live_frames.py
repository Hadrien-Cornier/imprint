"""Persist one initial RMBench observation for offline feature parity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sapien.core as sapien_core

from debug_rmbench_live_input_parity import build_eval_args
from rmbench_molmoact2_adapter import extract_rmbench_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sapien_core.SapienRenderer = lambda: sapien_core.render.SapienRenderer(
        sapien_core.Device("cuda:0")
    )

    from envs.cover_blocks import cover_blocks

    expert = cover_blocks()
    try:
        expert.setup_demo(
            now_ep_num=0,
            seed=args.seed,
            is_test=True,
            **build_eval_args(),
        )
        expert.play_once()
    finally:
        expert.close_env()

    environment = cover_blocks()
    try:
        environment.setup_demo(
            now_ep_num=0,
            seed=args.seed,
            is_test=True,
            **build_eval_args(),
        )
        frames, state = extract_rmbench_observation(environment.get_obs())
        np.save(output_dir / "live_frames.npy", frames)
        np.save(output_dir / "live_state.npy", state)
        print(f"frames={frames.shape} dtype={frames.dtype}")
        print(f"state={state.tolist()}")
    finally:
        environment.close_env()


if __name__ == "__main__":
    main()
