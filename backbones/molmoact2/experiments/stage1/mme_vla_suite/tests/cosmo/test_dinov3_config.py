from pathlib import Path


def test_dinov3_stage1_config_uses_768_dimensional_recent_features():
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "rmbench_all9_compact_nl256_dinov3.yaml"
    )

    config = config_path.read_text()

    assert "input_dim: 768" in config
    assert "siglip_dim: 768" in config
    assert "align_dim: 768" in config
    assert "align_num_patches: 192" in config
    assert "latent_source: offline" in config
