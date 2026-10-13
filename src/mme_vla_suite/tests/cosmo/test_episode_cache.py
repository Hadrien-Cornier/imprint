import json

import numpy as np


def test_episode_store_cache_reuses_entry_for_repeated_episode(tmp_path):
    from mme_vla_suite.training.episode_cache import EpisodeStoreCache

    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    (episode_dir / "chunk_index.json").write_text(json.dumps({"chunk_frames": 4}))
    (episode_dir / "variant_index.json").write_text(
        json.dumps({"v0": {"frame_idx": [0, 1, 2, 3]}})
    )
    np.save(
        episode_dir / "action_proprio.npy",
        {"action": np.zeros((4, 12), dtype=np.float32), "proprio": np.zeros((4, 16), dtype=np.float32)},
    )
    np.save(episode_dir / "recent_siglip.npy", np.zeros((4, 3, 64, 8), dtype=np.float32))
    np.save(episode_dir / "latents_v0.npy", np.zeros((1, 12, 48), dtype=np.float32))

    cache = EpisodeStoreCache(capacity=1)
    first = cache.get(episode_dir)
    second = cache.get(episode_dir)

    assert second is first
    assert second.get_latents(0) is first.get_latents(0)
