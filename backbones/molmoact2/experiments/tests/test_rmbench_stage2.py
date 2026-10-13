import json
import pickle

import numpy as np

from olmo.data.rmbench_stage2 import (
    CompactRMBenchFrameDataset,
    CompactRMBenchMemoryAssembler,
    RMBenchMemoryLayout,
)


def _write_feature_store(tmp_path) -> None:
    episode_dir = tmp_path / "features" / "episode_7"
    episode_dir.mkdir(parents=True)
    np.save(
        episode_dir / "latents_v0.npy",
        np.arange(2 * 4 * 48, dtype=np.float32).reshape(2, 4, 48),
    )
    np.save(
        episode_dir / "dinov3.npy",
        np.arange(3 * 1 * 1 * 768, dtype=np.float32).reshape(3, 1, 1, 768),
    )
    np.save(
        episode_dir / "action_causal_v0.npy",
        np.arange(3 * 16, dtype=np.float32).reshape(3, 16),
    )
    np.save(
        episode_dir / "action_proprio.npy",
        {"proprio": np.arange(3 * 16, dtype=np.float32).reshape(3, 16)},
        allow_pickle=True,
    )
    np.save(episode_dir / "lang_emb.npy", np.arange(2048, dtype=np.float32))
    (episode_dir / "variant_index.json").write_text(
        json.dumps({"v0": {"frame_idx": [0, 1, 2]}}),
        encoding="utf-8",
    )
    (episode_dir / "chunk_index.json").write_text(
        json.dumps({"chunk_frames": 2}),
        encoding="utf-8",
    )


def test_offline_memory_uses_stored_pi05_language_embedding_and_fixed_masks(tmp_path) -> None:
    _write_feature_store(tmp_path)
    layout = RMBenchMemoryLayout(
        latent_cap=8,
        max_recent_tokens=4,
        max_step_tokens=5,
        latent_pool_h=None,
        latent_pool_w=None,
    )
    assembler = CompactRMBenchMemoryAssembler(
        feature_root=tmp_path / "features",
        layout=layout,
        n_variants=1,
    )

    memory = assembler.gather(episode_index=7, step_index=2, rng=np.random.default_rng(0))

    assert memory["cosmo_latent_emb"].shape == (8, 48)
    assert memory["cosmo_mem_mask"].tolist() == [True, True, True, True, False, False, False, False]
    np.testing.assert_array_equal(
        memory["cosmo_latent_emb"][:4],
        np.arange(4 * 48, dtype=np.float32).reshape(4, 48),
    )
    assert memory["cosmo_recent_emb"].shape == (4, 768)
    assert memory["cosmo_recent_mask"].tolist() == [True, False, False, False]
    np.testing.assert_array_equal(
        memory["cosmo_recent_emb"][0],
        np.arange(2 * 768, 3 * 768, dtype=np.float32),
    )
    assert memory["cosmo_action_mask"].tolist() == [True, True, True, False, False]
    assert memory["cosmo_proprio_mask"].tolist() == [True, True, True, False, False]
    np.testing.assert_array_equal(memory["cosmo_lang_emb"], np.arange(2048, dtype=np.float32))


def test_compact_dataset_maps_pkl_row_and_pi05_memory_fields(tmp_path) -> None:
    _write_feature_store(tmp_path)
    data_root = tmp_path / "stage2"
    (data_root / "data").mkdir(parents=True)
    (data_root / "meta").mkdir()
    (data_root / "meta" / "samples.json").write_text(
        json.dumps([{"sample_index": 3, "episode_index": 7, "step_index": 2}]),
        encoding="utf-8",
    )
    with (data_root / "data" / "3.pkl").open("wb") as f:
        pickle.dump(
            {
                "image": np.zeros((4, 5, 3), dtype=np.uint8),
                "wrist_image": np.ones((4, 5, 3), dtype=np.uint8),
                "right_image": np.full((4, 5, 3), 2, dtype=np.uint8),
                "state": np.arange(16, dtype=np.float32),
                "actions": np.zeros((50, 16), dtype=np.float32),
                "prompt": "close the drawer",
            },
            f,
        )

    dataset = CompactRMBenchFrameDataset(
        data_root=data_root,
        feature_root=tmp_path / "features",
    )
    frame = dataset[0]

    assert frame["task"] == "close the drawer"
    assert frame["action"].shape == (50, 16)
    assert frame["observation.state"].shape == (16,)
    assert frame["cosmo_recent_emb"].shape == (768, 768)
    assert frame["cosmo_action_emb"].shape == (1244, 16)
    np.testing.assert_array_equal(frame["cosmo_lang_emb"], np.arange(2048, dtype=np.float32))
