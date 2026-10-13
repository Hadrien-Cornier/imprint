#!/usr/bin/env python3
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.config import get_config


EXPECTED = {
    "imprint_rmbench_full_attention_memft": (50, 100_000, 4, 1244, "dinov3"),
    "imprint_rmbench_recurrent_memft": (50, 100_000, 4, 1244, "dinov3"),
    "imprint_robocasa_full_attention_memft": (20, 150_000, 1, 1776, "dinov3"),
    "imprint_robocasa_recurrent_memft": (20, 150_000, 4, 1776, "dinov3"),
}


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
    print("RELEASE_CONFIGS_OK")


if __name__ == "__main__":
    main()
