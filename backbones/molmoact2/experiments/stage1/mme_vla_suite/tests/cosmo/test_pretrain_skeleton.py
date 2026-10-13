"""Skeleton smoke test: the Stage-1 trainer imports and exposes main()."""
import importlib
import logging

import jax
import numpy as np
import optax
import pytest
from flax import nnx


def test_pretrain_bottleneck_importable():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    assert hasattr(mod, "main") and callable(mod.main)
    assert hasattr(mod, "parse_args")


def test_pretrain_bottleneck_help():
    # argparse builds without touching the (unbuilt) store.
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    args = mod.parse_args(["--config", "x.yaml"])
    assert args.config == "x.yaml"
    assert args.recent_feature_source == "siglip"


def test_pretrain_bottleneck_accepts_dinov3_recent_features():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")

    args = mod.parse_args(["--config", "x.yaml", "--recent_feature_source", "dinov3"])

    assert args.recent_feature_source == "dinov3"


def test_reference_caps_and_resume_modes_are_supported():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    args = mod.parse_args([
        "--config", "x.yaml",
        "--max_recent_tokens", "768",
        "--max_step_tokens", "1244",
        "--resume_full_state", "full_state/10000",
        "--lr_decay_steps", "160000",
    ])
    assert args.max_recent_tokens == 768
    assert args.max_step_tokens == 1244
    assert args.resume_full_state == "full_state/10000"
    assert args.lr_decay_steps == 160000

    with pytest.raises(SystemExit):
        mod.parse_args([
            "--config", "x.yaml",
            "--resume_stage1_ckpt", "weights.npz",
            "--resume_full_state", "full_state/10000",
        ])


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ([12, 34], (12, 34)),
        (
            {
                "episode_index": 12,
                "global_episode_index": 112,
                "sample_index": 99,
                "step_index": 34,
            },
            (12, 34),
        ),
    ],
)
def test_sample_location_supports_legacy_and_compact_records(sample, expected):
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    assert mod._sample_location(sample) == expected


def test_training_state_checkpoint_round_trip(tmp_path):
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    _, state = nnx.split(model)
    tx = optax.adamw(1e-3)
    opt_state = tx.init(state)
    grads = jax.tree.map(jax.numpy.ones_like, state)
    updates, opt_state = tx.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)

    checkpoint_path = mod._save_training_state(tmp_path, 7, state, opt_state)

    target_model = nnx.Linear(2, 3, rngs=nnx.Rngs(1))
    _, target_state = nnx.split(target_model)
    target_opt_state = tx.init(target_state)
    restored_state, restored_opt_state, restored_step = mod._restore_training_state(
        checkpoint_path, target_state, target_opt_state
    )

    assert restored_step == 7
    for expected, actual in zip(
        jax.tree.leaves(state), jax.tree.leaves(restored_state), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(
        jax.tree.leaves(opt_state), jax.tree.leaves(restored_opt_state), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)


def test_format_progress_reports_elapsed_rate_and_eta():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    line = mod._format_progress(
        step=25,
        total_steps=100,
        elapsed_seconds=50.0,
        recent_elapsed_seconds=10.0,
        recent_steps=5,
    )
    assert line == (
        "PROGRESS 25/100 (25.0%) "
        "elapsed=00:00:50 recent=2.000s/step avg=2.000s/step eta=00:02:30"
    )


def test_configure_logging_enables_trainer_info_messages():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    mod.logger.setLevel(logging.WARNING)
    mod._configure_logging()
    assert mod.logger.isEnabledFor(logging.INFO)


def test_place_training_state_uses_requested_sharding():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = {"weight": jax.numpy.arange(4, dtype=jax.numpy.float32)}
    opt_state = {"moment": jax.numpy.ones(4, dtype=jax.numpy.float32)}

    placed_state, placed_opt_state = mod._place_training_state(
        state, opt_state, replicated
    )

    assert placed_state["weight"].sharding == replicated
    assert placed_opt_state["moment"].sharding == replicated
