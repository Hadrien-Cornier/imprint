"""Compare the evaluator's live RMBench input with a Stage-2 training row."""

from __future__ import annotations

import pickle
import os
from pathlib import Path

import numpy as np
import sapien.core as sapien_core
import yaml


RMBENCH_ROOT = Path(os.environ.get("RMBENCH_ROOT", "/path/to/RMBench"))
STAGE2_SAMPLE = Path(
    os.environ.get("RMBENCH_STAGE2_SAMPLE", "/path/to/rmbench/stage2/data/sample.pkl")
)


def _load_robot_config(path: str) -> dict:
    with (Path(path) / "config.yml").open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def build_eval_args() -> dict:
    from envs import CONFIGS_PATH

    with (RMBENCH_ROOT / "task_config/demo_clean.yml").open(encoding="utf-8") as handle:
        args = yaml.load(handle, Loader=yaml.FullLoader)
    with Path(f"{CONFIGS_PATH}_embodiment_config.yml").open(encoding="utf-8") as handle:
        embodiments = yaml.load(handle, Loader=yaml.FullLoader)
    with Path(f"{CONFIGS_PATH}_camera_config.yml").open(encoding="utf-8") as handle:
        cameras = yaml.load(handle, Loader=yaml.FullLoader)

    args["task_name"] = "cover_blocks"
    args["task_config"] = "demo_clean"
    args["ckpt_setting"] = "molmoact2_stage2"
    args["head_camera_h"] = cameras[args["camera"]["head_camera_type"]]["h"]
    args["head_camera_w"] = cameras[args["camera"]["head_camera_type"]]["w"]
    robot_file = embodiments[args["embodiment"][0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = _load_robot_config(robot_file)
    args["right_embodiment_config"] = _load_robot_config(robot_file)
    return args


def main() -> None:
    # Match the evaluator's explicit render-device selection before constructing
    # the task environment.
    sapien_core.SapienRenderer = lambda: sapien_core.render.SapienRenderer(
        sapien_core.Device("cuda:0")
    )

    from envs.cover_blocks import cover_blocks
    from rmbench_molmoact2_adapter import extract_rmbench_observation

    environment = cover_blocks()
    try:
        environment.setup_demo(
            now_ep_num=0,
            seed=100000,
            is_test=True,
            **build_eval_args(),
        )
        observation = environment.get_obs()
        live_frames, live_state = extract_rmbench_observation(observation)
        with STAGE2_SAMPLE.open("rb") as handle:
            target = pickle.load(handle)

        print("LIVE_CAMERA_KEYS", tuple(observation["observation"]))
        print("LIVE_NATIVE_STATE", observation["joint_action"]["vector"])
        print("LIVE_MODEL_STATE", live_state)
        print("TARGET_MODEL_STATE", target["state"])
        print("STATE_MAX_ABS", float(np.abs(live_state - target["state"]).max()))

        for index, key in enumerate(("image", "wrist_image", "right_image")):
            camera = ("head_camera", "left_camera", "right_camera")[index]
            raw = observation["observation"][camera]["rgb"]
            target_frame = target[key]
            difference = np.abs(live_frames[index].astype(np.int16) - target_frame.astype(np.int16))
            print(
                "CAMERA",
                key,
                "RAW",
                raw.shape,
                raw.dtype,
                int(raw.min()),
                int(raw.max()),
                float(raw.mean()),
            )
            print(
                "CAMERA",
                key,
                "MODEL",
                live_frames[index].shape,
                live_frames[index].dtype,
                int(live_frames[index].min()),
                int(live_frames[index].max()),
                float(live_frames[index].mean()),
                "TARGET",
                int(target_frame.min()),
                int(target_frame.max()),
                float(target_frame.mean()),
                "MAX_ABS",
                int(difference.max()),
                "MEAN_ABS",
                float(difference.mean()),
                "EQUAL",
                bool(np.array_equal(live_frames[index], target_frame)),
            )
        print("ENV_INSTRUCTION_BEFORE_SET", repr(environment.get_instruction()))
    finally:
        environment.close_env()


if __name__ == "__main__":
    main()
