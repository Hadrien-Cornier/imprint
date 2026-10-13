import jax
import jax.numpy as jnp

from mme_vla_suite.models.integration import history_pi0


def test_current_obs_dropout_disabled_preserves_images_and_masks():
    images = {
        "base_0_rgb": jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
        "left_wrist_0_rgb": jnp.full((2, 4, 4, 3), 2.0, dtype=jnp.float32),
    }
    masks = {
        "base_0_rgb": jnp.array([True, True]),
        "left_wrist_0_rgb": jnp.array([True, False]),
    }

    dropped_images, dropped_masks = history_pi0.apply_current_obs_dropout(
        images,
        masks,
        rng=jax.random.PRNGKey(0),
        prob=0.0,
        fill_value=0.0,
    )

    assert dropped_masks == masks
    for name in images:
        assert jnp.array_equal(dropped_images[name], images[name])


def test_current_obs_dropout_full_probability_blanks_valid_views_only():
    images = {
        "base_0_rgb": jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
        "left_wrist_0_rgb": jnp.full((2, 4, 4, 3), 2.0, dtype=jnp.float32),
        "right_wrist_0_rgb": jnp.full((2, 4, 4, 3), 3.0, dtype=jnp.float32),
    }
    masks = {
        "base_0_rgb": jnp.array([True, True]),
        "left_wrist_0_rgb": jnp.array([True, False]),
        "right_wrist_0_rgb": jnp.array([True, True]),
    }

    dropped_images, dropped_masks = history_pi0.apply_current_obs_dropout(
        images,
        masks,
        rng=jax.random.PRNGKey(0),
        prob=1.0,
        fill_value=0.0,
    )

    assert dropped_masks == masks
    assert jnp.array_equal(dropped_images["base_0_rgb"], jnp.zeros_like(images["base_0_rgb"]))
    assert jnp.array_equal(dropped_images["left_wrist_0_rgb"], images["left_wrist_0_rgb"])
    assert jnp.array_equal(dropped_images["right_wrist_0_rgb"], images["right_wrist_0_rgb"])
