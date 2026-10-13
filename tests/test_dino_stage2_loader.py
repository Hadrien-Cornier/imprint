import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mme_vla_suite.training.dataset import RoboMMEDataset, resolve_feature_dir


def _write_dino_episode(store: Path) -> None:
    episode = store / "features" / "episode_0"
    episode.mkdir(parents=True)
    (store / "meta").mkdir()
    (store / "meta" / "dino_store_manifest.json").write_text(
        json.dumps(
            {
                "extractor": {
                    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                    "feature_dim": 768,
                    "tokens_per_view": 64,
                },
                "replacement_streams": {
                    "recent": {"file": "dinov3.npy", "shape": ["T", "V", 64, 768]},
                    "alignment": {"file": "dinov3.npy", "shape": ["T", "V*P", 768]},
                },
            }
        )
    )
    (episode / "chunk_index.json").write_text(json.dumps({"chunk_frames": 4}))
    (episode / "variant_index.json").write_text(
        json.dumps({"v0": {"frame_idx": [0, 1, 2, 3, 4]}})
    )
    np.save(
        episode / "dinov3.npy",
        np.arange(5 * 3 * 64 * 768, dtype=np.float16).reshape(5, 3, 64, 768),
    )
    action = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
    np.save(
        episode / "action_proprio.npy",
        {"action": action, "action_causal": action + 1, "proprio": action + 2},
        allow_pickle=True,
    )
    np.save(episode / "action_causal_v0.npy", action + 1)
    np.save(episode / "latents_v0.npy", np.zeros((1, 12, 48), np.float16))


def test_offline_dino_stage2_loader_uses_canonical_dino_stream(tmp_path: Path) -> None:
    _write_dino_episode(tmp_path)
    cfg = SimpleNamespace(
        visual_feature_stream="dinov3",
        latent_source="offline",
        n_variants=1,
        num_query_samples=2,
        pool_hw=0,
        pool_h=2,
        pool_w=2,
        wan_h=256,
        wan_w=256,
        siglip_dim=768,
        lang_emb_dim=2048,
        action_encoding="raw",
        proprio_encoding="raw",
        latent_cap=64,
        max_recent_tokens=768,
    )
    dataset = RoboMMEDataset.__new__(RoboMMEDataset)
    dataset.feature_dir = tmp_path / "features"
    dataset.history_config = SimpleNamespace(cosmo_bottleneck_memory=cfg)
    dataset._episode_store_cache = None
    dataset.need_aux = True

    sample = dataset._gather_cosmo_bottleneck(epis_idx=0, step_idx=4)

    assert sample["cosmo_recent_emb"].shape == (768, 768)
    assert sample["cosmo_recent_mask"].sum() == 192
    assert sample["aux_align_target"].shape == (2, 192, 768)
    assert sample["cosmo_action_emb"][0].tolist() == (np.arange(16) + 1).tolist()


def test_dino_stage2_can_use_separate_published_feature_store(tmp_path: Path) -> None:
    dino_store = tmp_path / "published_dino_store"
    cfg = SimpleNamespace(feature_store_path=str(dino_store))
    history_config = SimpleNamespace(cosmo_bottleneck_memory=cfg)

    assert resolve_feature_dir(tmp_path / "h50_stage2_data", history_config) == (
        dino_store / "features"
    )
