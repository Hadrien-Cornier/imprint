#!/usr/bin/env python3
from pathlib import Path

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.config import get_config


EXPECTED = {
    "imprint_rmbench_full_attention_memft": (50, 100_000, 4, 1244, "dinov3"),
    "imprint_rmbench_recurrent_memft": (50, 100_000, 4, 1244, "dinov3"),
    "imprint_robocasa_full_attention_memft": (20, 150_000, 1, 1776, "dinov3"),
    "imprint_robocasa_recurrent_memft": (20, 150_000, 4, 1776, "dinov3"),
}

PAPER_STAGE1_SCHEDULES = {
    "configs/paper/rmbench/full_attention/experiment.env": ("4096", "0", "1"),
    "configs/paper/rmbench/recurrent/experiment.env": (
        "256,512,1024,2048,4096",
        "1",
        "1",
    ),
    "configs/paper/robocasa/full_attention/experiment.env": (
        "256,512,1024,2048,4096",
        "0",
        "1",
    ),
    "configs/paper/robocasa/recurrent/experiment.env": (
        "256,512,1024,2048,4096",
        "1",
        "1",
    ),
}


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise AssertionError((path, "invalid env line", raw_line))
        values[key] = value
    return values


def main() -> None:
    for name, expected in EXPECTED.items():
        config = get_config(name)
        history = get_history_config(config.model.history_config).cosmo_bottleneck_memory
        actual = (
            config.model.action_horizon,
            config.num_train_steps,
            config.fsdp_devices,
            history.max_step_tokens,
            history.visual_feature_stream,
        )
        if actual != expected:
            raise AssertionError((name, actual, expected))
        if history.num_latents != 256:
            raise AssertionError((name, "num_latents", history.num_latents))
        if history.latent_cap != 4096:
            raise AssertionError((name, "latent_cap", history.latent_cap))
        if history.max_recent_tokens != 768:
            raise AssertionError((name, "max_recent_tokens", history.max_recent_tokens))

    root = Path(__file__).resolve().parents[2]
    for relative_path, expected in PAPER_STAGE1_SCHEDULES.items():
        values = read_env(root / relative_path)
        actual = (
            values.get("IMPRINT_STAGE1_LATENT_BUCKETS"),
            values.get("IMPRINT_LENGTH_BUCKETED_BATCHES"),
            values.get("IMPRINT_DETERMINISTIC_TRAINING"),
        )
        if actual != expected:
            raise AssertionError((relative_path, actual, expected))
    print("RELEASE_CONFIGS_OK")


if __name__ == "__main__":
    main()
