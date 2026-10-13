import dataclasses

import numpy as np

from mme_vla_suite.policies.robomme_policy import RoboMMEInputs
from mme_vla_suite.training.config import get_config
from mme_vla_suite.training.dataset import pool_square_offline_latents


def test_robomme_imitation4_pi0_stage2_contract():
    config = get_config(
        "mme_wan_stage2_pi0_robomme_imitation4_memft_freeze_language"
    )
    assert config.model.pi05 is False
    assert config.model.action_horizon == 50
    assert config.batch_size == 64
    assert config.fsdp_devices == 4
    assert config.num_train_steps == 100_000
    assert "robomme-imitation4" in config.model.history_config


def test_robomme_imitation4_pi05_stage2_contract():
    config = get_config(
        "mme_wan_stage2_pi05_robomme_imitation4_memft_freeze_language"
    )
    assert config.model.pi05 is True
    assert config.model.action_horizon == 20
    assert config.model.discrete_state_input is False
    assert config.batch_size == 64
    assert config.fsdp_devices == 4
    assert config.num_train_steps == 100_000
    assert "robomme-imitation4" in config.model.history_config


def test_two_view_input_omits_third_pi0_camera():
    transform = RoboMMEInputs(
        model_type=get_config(
            "mme_wan_stage2_pi0_robomme_imitation4_memft_freeze_language"
        ).model.model_type
    )
    output = transform(
        {
            "observation/image": np.zeros((32, 32, 3), np.uint8),
            "observation/wrist_image": np.zeros((32, 32, 3), np.uint8),
            "observation/state": np.zeros((8,), np.float32),
            "actions": np.zeros((50, 8), np.float32),
            "prompt": "test",
        }
    )
    assert set(output["image"]) == {"base_0_rgb", "left_wrist_0_rgb"}
    assert set(output["image_mask"]) == {"base_0_rgb", "left_wrist_0_rgb"}


def test_two_view_input_supports_pi05():
    transform = RoboMMEInputs(
        model_type=get_config(
            "mme_wan_stage2_pi05_robomme_imitation4_memft_freeze_language"
        ).model.model_type
    )
    output = transform(
        {
            "observation/image": np.zeros((32, 32, 3), np.uint8),
            "observation/wrist_image": np.zeros((32, 32, 3), np.uint8),
            "observation/state": np.zeros((8,), np.float32),
            "actions": np.zeros((20, 8), np.float32),
            "prompt": "test",
        }
    )
    assert set(output["image"]) == {"base_0_rgb", "left_wrist_0_rgb"}
    assert set(output["image_mask"]) == {"base_0_rgb", "left_wrist_0_rgb"}


def test_joint_data_config_applies_delta_actions():
    config = get_config(
        "mme_wan_stage2_pi0_robomme_imitation4_memft_freeze_language"
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    state = np.arange(8, dtype=np.float32)
    absolute = np.tile(state, (50, 1))
    absolute[:, 7] = -1.0
    sample = {
        "image": np.zeros((32, 32, 3), np.uint8),
        "wrist_image": np.zeros((32, 32, 3), np.uint8),
        "state": state,
        "actions": absolute,
        "prompt": "test",
        "static_image_emb": None,
        "static_pos_emb": None,
        "static_state_emb": None,
        "static_mask": None,
        "static_time": None,
        "recur_image_emb": None,
        "recur_pos_emb": None,
        "recur_state_emb": None,
        "recur_mask": None,
        "cosmo_latent_emb": None,
        "cosmo_mem_mask": None,
        "cosmo_recent_emb": None,
        "cosmo_recent_mask": None,
        "cosmo_action_emb": None,
        "cosmo_action_mask": None,
        "cosmo_proprio_emb": None,
        "cosmo_proprio_mask": None,
        "cosmo_lang_emb": None,
        "simple_subgoal": None,
        "grounded_subgoal": None,
    }
    transformed = sample
    for transform in data_config.repack_transforms.inputs:
        transformed = transform(transformed)
    for transform in data_config.data_transforms.inputs:
        transformed = transform(transformed)
    np.testing.assert_allclose(transformed["actions"][:, :7], 0.0)
    np.testing.assert_allclose(transformed["actions"][:, 7], -1.0)


def test_pi05_joint_data_config_applies_delta_actions():
    config = get_config(
        "mme_wan_stage2_pi05_robomme_imitation4_memft_freeze_language"
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    state = np.arange(8, dtype=np.float32)
    absolute = np.tile(state, (20, 1))
    absolute[:, 7] = 1.0
    sample = {
        "image": np.zeros((32, 32, 3), np.uint8),
        "wrist_image": np.zeros((32, 32, 3), np.uint8),
        "state": state,
        "actions": absolute,
        "prompt": "test",
        "static_image_emb": None,
        "static_pos_emb": None,
        "static_state_emb": None,
        "static_mask": None,
        "static_time": None,
        "recur_image_emb": None,
        "recur_pos_emb": None,
        "recur_state_emb": None,
        "recur_mask": None,
        "cosmo_latent_emb": None,
        "cosmo_mem_mask": None,
        "cosmo_recent_emb": None,
        "cosmo_recent_mask": None,
        "cosmo_action_emb": None,
        "cosmo_action_mask": None,
        "cosmo_proprio_emb": None,
        "cosmo_proprio_mask": None,
        "cosmo_lang_emb": None,
        "simple_subgoal": None,
        "grounded_subgoal": None,
    }
    transformed = sample
    for transform in data_config.repack_transforms.inputs:
        transformed = transform(transformed)
    for transform in data_config.data_transforms.inputs:
        transformed = transform(transformed)
    np.testing.assert_allclose(transformed["actions"][:, :7], 0.0)
    np.testing.assert_allclose(transformed["actions"][:, 7], 1.0)


def test_square_latent_pooling_preserves_prepooled_store():
    prepooled = np.arange(3 * 8 * 48, dtype=np.float32).reshape(3, 8, 48)
    np.testing.assert_array_equal(
        pool_square_offline_latents(prepooled, pool_hw=2),
        prepooled,
    )

    unpooled = np.ones((3, 2 * 64, 48), dtype=np.float32)
    pooled = pool_square_offline_latents(unpooled, pool_hw=2)
    assert pooled.shape == (3, 2 * 4, 48)
    np.testing.assert_array_equal(pooled, 1.0)
