"""Run one exact live RMBench policy request without executing an action."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import sapien.core as sapien_core
import yaml


RMBENCH_ROOT = Path(os.environ.get("RMBENCH_ROOT", "/path/to/RMBench"))
POLICY_CLIENT_PATH = Path(
    os.environ.get("RMBENCH_POLICY_CLIENT_PATH", "outputs/eval_policy_client.py")
)
POLICY_PORT = 8042


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _robot_config(path: str) -> dict:
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
    args["left_embodiment_config"] = _robot_config(robot_file)
    args["right_embodiment_config"] = _robot_config(robot_file)
    return args


def main() -> None:
    sapien_core.SapienRenderer = lambda: sapien_core.render.SapienRenderer(
        sapien_core.Device("cuda:0")
    )

    from envs.cover_blocks import cover_blocks
    from generate_episode_instructions import generate_episode_descriptions

    client_module = _load_module(POLICY_CLIENT_PATH, "_rmbench_policy_client")
    model = client_module.ModelClient(port=POLICY_PORT)
    args = build_eval_args()

    expert = cover_blocks()
    try:
        expert.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
        episode_info = expert.play_once()
    finally:
        expert.close_env()

    environment = cover_blocks()
    try:
        environment.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
        descriptions = generate_episode_descriptions(
            args["task_name"],
            [episode_info["info"]],
            1,
        )
        instruction = np.random.choice(descriptions[0]["unseen"])
        environment.set_instruction(instruction=instruction)
        observation = environment.get_obs()

        model.call(func_name="reset_model")
        actions = model.call(
            func_name="eval",
            obs={"prompt": instruction, "observation": observation},
        )
        actions = np.asarray(actions, dtype=np.float32)
        initial_target = np.zeros((16,), dtype=np.float32)
        initial_target[14:] = 1.0
        initial_error = np.abs(actions[0] - initial_target)

        print("LIVE_PROMPT", repr(instruction))
        print("LIVE_CAMERA_KEYS", tuple(observation["observation"]))
        print("LIVE_NATIVE_STATE", observation["joint_action"]["vector"])
        for camera in ("head_camera", "left_camera", "right_camera"):
            frame = observation["observation"][camera]["rgb"]
            print(
                "LIVE_CAMERA",
                camera,
                frame.shape,
                frame.dtype,
                int(frame.min()),
                int(frame.max()),
                float(frame.mean()),
            )
        print("POLICY_ACTIONS_SHAPE", actions.shape)
        print("POLICY_ACTION_0", actions[0])
        print("POLICY_ACTION_0_ABS_MEAN_TO_INITIAL_TARGET", float(initial_error.mean()))
        print("POLICY_ACTION_0_ABS_MAX_TO_INITIAL_TARGET", float(initial_error.max()))
    finally:
        model.close()
        environment.close_env()


if __name__ == "__main__":
    main()
