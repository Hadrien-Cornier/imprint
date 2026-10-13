import pathlib
import sys
import importlib.util

import jax.numpy as jnp

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(_OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENPI_ROOT))

from openpi.models.model import Observation
from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
    preprocess_observation,
)
from mme_vla_suite.training.dataset import late_episode_loss_weight

_TRAIN_MME_PATH = _OPENPI_ROOT / "scripts" / "train_mme.py"
_SPEC = importlib.util.spec_from_file_location("train_mme_for_test", _TRAIN_MME_PATH)
_TRAIN_MME = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN_MME)
_apply_sample_loss_weight = _TRAIN_MME._apply_sample_loss_weight


def _base_observation():
    return Observation(
        images={
            "base_0_rgb": jnp.zeros((1, 8, 8, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.zeros((1, 8, 8, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.zeros((1, 8, 8, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.array([True]),
            "left_wrist_0_rgb": jnp.array([True]),
            "right_wrist_0_rgb": jnp.array([True]),
        },
        state=jnp.zeros((1, 14), dtype=jnp.float32),
    )


def test_preprocess_preserves_precomputed_cosmo_mem_tokens_and_sample_weight():
    tokens = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)
    weight = jnp.array([3.0], dtype=jnp.float32)
    obs = HistAugObservation.from_base_obs(
        _base_observation(),
        cosmo_mem_tokens=tokens,
        sample_loss_weight=weight,
    )

    out = preprocess_observation(None, obs, train=False)

    assert jnp.array_equal(out.cosmo_mem_tokens, tokens)
    assert jnp.array_equal(out.sample_loss_weight, weight)


def test_late_episode_loss_weight_switches_at_fraction_boundary():
    assert late_episode_loss_weight(4, 10, late_fraction=0.5, late_weight=3.0) == 1.0
    assert late_episode_loss_weight(5, 10, late_fraction=0.5, late_weight=3.0) == 3.0


def test_apply_sample_loss_weight_broadcasts_over_action_chunks():
    loss = jnp.ones((2, 3, 4), dtype=jnp.float32)
    weight = jnp.array([1.0, 3.0], dtype=jnp.float32)

    weighted = _apply_sample_loss_weight(loss, weight)

    assert jnp.allclose(weighted[0], 1.0)
    assert jnp.allclose(weighted[1], 3.0)
