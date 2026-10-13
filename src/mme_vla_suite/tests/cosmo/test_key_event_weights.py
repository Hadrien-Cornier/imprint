import json

import pytest

from mme_vla_suite.training.key_event_weights import KeyEventWeightLookup


def _write_manifest(path, **overrides):
    payload = {
        "schema_version": 1,
        "training_store_path": "/store",
        "episode_count": 1,
        "window_radius": 5,
        "action_loss_weight": 3.0,
        "episodes": {"0": {"weighted_steps": [4, 5, 6]}},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def test_lookup_returns_key_and_default_weights(tmp_path):
    path = tmp_path / "labels.json"
    _write_manifest(path)

    lookup = KeyEventWeightLookup.load(
        path,
        expected_store_path="/store",
        expected_episode_count=1,
        expected_radius=5,
        expected_weight=3.0,
    )

    assert lookup.weight_for(0, 5) == 3.0
    assert lookup.weight_for(0, 7) == 1.0


def test_lookup_rejects_stale_metadata(tmp_path):
    path = tmp_path / "labels.json"
    _write_manifest(path, window_radius=4)

    with pytest.raises(ValueError, match="window_radius"):
        KeyEventWeightLookup.load(
            path,
            expected_store_path="/store",
            expected_episode_count=1,
            expected_radius=5,
            expected_weight=3.0,
        )


def test_lookup_rejects_duplicate_steps(tmp_path):
    path = tmp_path / "labels.json"
    _write_manifest(
        path,
        episodes={"0": {"weighted_steps": [4, 5, 5]}},
    )

    with pytest.raises(ValueError, match="sorted and unique"):
        KeyEventWeightLookup.load(
            path,
            expected_store_path="/store",
            expected_episode_count=1,
            expected_radius=5,
            expected_weight=3.0,
        )
