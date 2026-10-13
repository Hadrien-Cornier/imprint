import numpy as np
import pytest

from mme_vla_suite.shared.recent_features import (
    load_recent_feature_array,
    recent_feature_filename,
    validate_recent_feature_store,
)


def test_siglip_source_resolves_legacy_feature_file(tmp_path):
    features = np.zeros((3, 4, 1152), dtype=np.float32)
    np.save(tmp_path / "recent_siglip.npy", features)

    assert recent_feature_filename("siglip") == "recent_siglip.npy"
    loaded = load_recent_feature_array(tmp_path, "siglip")

    assert loaded.shape == features.shape


def test_dinov3_source_requires_768_dimensional_features(tmp_path):
    features = np.zeros((3, 4, 768), dtype=np.float32)
    np.save(tmp_path / "dinov3.npy", features)

    assert recent_feature_filename("dinov3") == "dinov3.npy"
    loaded = load_recent_feature_array(tmp_path, "dinov3")

    assert loaded.shape == features.shape


def test_dinov3_source_rejects_wrong_feature_dimension(tmp_path):
    np.save(tmp_path / "dinov3.npy", np.zeros((3, 4, 1024), dtype=np.float32))

    with pytest.raises(ValueError, match="expected 768"):
        load_recent_feature_array(tmp_path, "dinov3")


def test_missing_selected_feature_reports_source_and_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="source 'dinov3'.*dinov3.npy"):
        load_recent_feature_array(tmp_path, "dinov3")


def test_store_validation_checks_every_episode_directory(tmp_path):
    features_root = tmp_path / "features"
    first_episode = features_root / "episode_0"
    second_episode = features_root / "episode_1"
    first_episode.mkdir(parents=True)
    second_episode.mkdir()
    np.save(first_episode / "dinov3.npy", np.zeros((1, 2, 768), dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="episode_1.*dinov3.npy"):
        validate_recent_feature_store(tmp_path, "dinov3")


def test_store_validation_rejects_config_dimension_mismatch(tmp_path):
    episode_dir = tmp_path / "features" / "episode_0"
    episode_dir.mkdir(parents=True)
    np.save(episode_dir / "dinov3.npy", np.zeros((1, 2, 768), dtype=np.float32))

    with pytest.raises(ValueError, match="configured input dim 2048"):
        validate_recent_feature_store(tmp_path, "dinov3", expected_dim=2048)


def test_store_validation_rejects_dinov3_reconstruction_token_mismatch(tmp_path):
    episode_dir = tmp_path / "features" / "episode_0"
    episode_dir.mkdir(parents=True)
    np.save(episode_dir / "dinov3.npy", np.zeros((1, 3, 64, 768), dtype=np.float32))

    with pytest.raises(ValueError, match="configured reconstruction tokens 12"):
        validate_recent_feature_store(tmp_path, "dinov3", expected_tokens=12)
