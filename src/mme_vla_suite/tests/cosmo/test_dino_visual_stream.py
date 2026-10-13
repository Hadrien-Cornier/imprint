import numpy as np
import pytest


def test_dino_manifest_exposes_only_supported_visual_geometry(tmp_path):
    from mme_vla_suite.shared.visual_streams import load_dino_manifest

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "dino_store_manifest.json").write_text(
        """{
  "extractor": {
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "feature_dim": 768,
    "tokens_per_view": 64
  },
  "replacement_streams": {
    "recent": {"file": "dinov3.npy", "shape": ["T", "V", 64, 768]},
    "alignment": {"file": "dinov3.npy", "shape": ["T", "V*P", 768]}
  }
}"""
    )

    assert load_dino_manifest(tmp_path) == {
        "feature_dim": 768,
        "tokens_per_view": 64,
        "num_views": 3,
    }


def test_dino_manifest_rejects_wrong_stream_or_geometry(tmp_path):
    from mme_vla_suite.shared.visual_streams import load_dino_manifest

    meta = tmp_path / "meta"
    meta.mkdir()
    manifest = meta / "dino_store_manifest.json"
    manifest.write_text(
        """{
  "extractor": {
    "model_id": "google/siglip",
    "feature_dim": 768,
    "tokens_per_view": 64
  },
  "replacement_streams": {
    "recent": {"file": "dinov3.npy", "shape": ["T", "V", 64, 768]},
    "alignment": {"file": "dinov3.npy", "shape": ["T", "V*P", 768]}
  }
}"""
    )
    with pytest.raises(ValueError, match="model_id"):
        load_dino_manifest(tmp_path)

    manifest.write_text(
        """{
  "extractor": {
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "feature_dim": 1024,
    "tokens_per_view": 64
  },
  "replacement_streams": {
    "recent": {"file": "dinov3.npy", "shape": ["T", "V", 64, 1024]},
    "alignment": {"file": "dinov3.npy", "shape": ["T", "V*P", 1024]}
  }
}"""
    )
    with pytest.raises(ValueError, match="feature_dim"):
        load_dino_manifest(tmp_path)


def test_dino_visual_stream_uses_canonical_tensor_for_recent_and_alignment(tmp_path):
    from mme_vla_suite.shared.visual_streams import load_visual_stream

    episode = tmp_path / "episode_0"
    episode.mkdir()
    dino = np.arange(5 * 3 * 4 * 6, dtype=np.float32).reshape(5, 3, 4, 6)
    np.save(episode / "dinov3.npy", dino)

    recent, align = load_visual_stream(
        episode,
        stream="dinov3",
        recent_idx=np.array([1, 3]),
        align_idx=np.array([2, 4]),
    )

    assert recent.shape == (2 * 3 * 4, 6)
    assert align.shape == (2, 3 * 4, 6)
    np.testing.assert_array_equal(recent, dino[[1, 3]].reshape(-1, 6))
    np.testing.assert_array_equal(align, dino[[2, 4]].reshape(2, 12, 6))


def test_dino_visual_stream_rejects_missing_or_malformed_tensor(tmp_path):
    from mme_vla_suite.shared.visual_streams import load_visual_stream

    with pytest.raises(FileNotFoundError, match="dinov3.npy"):
        load_visual_stream(
            tmp_path,
            stream="dinov3",
            recent_idx=np.array([], dtype=np.int64),
            align_idx=np.array([], dtype=np.int64),
        )

    np.save(tmp_path / "dinov3.npy", np.zeros((4, 64, 768), np.float16))
    with pytest.raises(ValueError, match="rank-4"):
        load_visual_stream(
            tmp_path,
            stream="dinov3",
            recent_idx=np.array([0]),
            align_idx=np.array([0]),
        )
