import numpy as np


def test_variant_local_causal_action_overrides_global_indexed_actions(tmp_path):
    from mme_vla_suite.training.dataset import load_variant_causal_action

    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    global_causal = np.arange(60, dtype=np.float32).reshape(5, 12)
    local_causal = np.full((2, 12), -7.0, dtype=np.float32)
    np.save(episode_dir / "action_causal_v3.npy", local_causal)

    actual = load_variant_causal_action(
        str(episode_dir),
        variant=3,
        full_variant_frame_idx=np.array([1, 3], dtype=np.int64),
        selected_variant_frame_idx=np.array([1, 3], dtype=np.int64),
        global_causal_actions=global_causal,
    )

    assert np.array_equal(actual, local_causal)


def test_variant_local_causal_action_rejects_wrong_length(tmp_path):
    from mme_vla_suite.training.dataset import load_variant_causal_action

    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    np.save(episode_dir / "action_causal_v1.npy", np.zeros((1, 12), dtype=np.float32))

    try:
        load_variant_causal_action(
            str(episode_dir),
            variant=1,
            full_variant_frame_idx=np.array([0, 2], dtype=np.int64),
            selected_variant_frame_idx=np.array([0], dtype=np.int64),
            global_causal_actions=np.zeros((3, 12), dtype=np.float32),
        )
    except ValueError as exc:
        assert "action_causal_v1.npy" in str(exc)
    else:
        raise AssertionError("expected variant action length validation to fail")
