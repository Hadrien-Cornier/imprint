import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.dataset import _next_seeded_generator
from mme_vla_suite.training.dataset import resolve_feature_dir


def _write_dino_episode(store: Path, *, n_variants: int = 1) -> None:
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
        json.dumps(
            {
                f"v{variant}": {"frame_idx": [0, 1, 2, 3, 4]}
                for variant in range(n_variants)
            }
        )
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
    for variant in range(n_variants):
        np.save(episode / f"action_causal_v{variant}.npy", action + variant + 1)
        np.save(
            episode / f"latents_v{variant}.npy",
            np.full((1, 12, 48), variant, np.float16),
        )


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


def test_per_sample_generator_respects_numpy_worker_seed() -> None:
    def draw_sequence() -> list[int]:
        return [
            int(_next_seeded_generator().integers(0, 1_000_000))
            for _ in range(16)
        ]

    np.random.seed(42)
    first = draw_sequence()
    np.random.seed(42)
    second = draw_sequence()

    assert first == second
    assert len(set(first)) > 1


def test_offline_variant_sequence_respects_numpy_worker_seed(tmp_path: Path) -> None:
    _write_dino_episode(tmp_path, n_variants=8)
    cfg = SimpleNamespace(
        visual_feature_stream="dinov3",
        latent_source="offline",
        n_variants=8,
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
    dataset.need_aux = False

    def draw_sequence() -> list[int]:
        return [
            int(dataset._gather_cosmo_bottleneck(0, 4)["cosmo_latent_emb"][0, 0])
            for _ in range(24)
        ]

    np.random.seed(42)
    first = draw_sequence()
    np.random.seed(42)
    second = draw_sequence()

    assert first == second
    assert len(set(first)) > 1
