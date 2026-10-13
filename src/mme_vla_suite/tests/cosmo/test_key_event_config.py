import numpy as np

from mme_vla_suite.policies.robodojo_policy import RoboDojoInputs
from mme_vla_suite.training.config import get_config


def test_key_event_config_is_isolated_from_late_half_weighting():
    key = get_config("mme_wan_stage2_robodojo_base_h30_keyevents3w5_60k")
    late = get_config("mme_wan_stage2_robodojo_base_h30_late50w3_60k")

    assert key.model.action_horizon == 30
    assert key.model.key_event_loss_weight == 3.0
    assert key.model.key_event_window_radius == 5
    assert key.model.key_event_manifest_path.endswith(
        "cover_blocks_key_events_v1.json"
    )
    assert key.model.late_episode_loss_weight == 1.0
    assert key.num_train_steps == 60_000
    assert key.save_interval == 10_000
    assert key.keep_period == 1

    assert late.model.key_event_manifest_path is None
    assert late.model.late_episode_loss_weight == 3.0


def test_robodojo_inputs_preserves_sample_loss_weight():
    config = get_config("mme_wan_stage2_robodojo_base_h30_keyevents3w5_60k")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transformed = RoboDojoInputs(model_type=config.model.model_type)(
        {
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/right_image": image,
            "observation/state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((30, 14), dtype=np.float32),
            "sample_loss_weight": np.asarray(3.0, dtype=np.float32),
        }
    )

    assert transformed["sample_loss_weight"] == np.float32(3.0)
