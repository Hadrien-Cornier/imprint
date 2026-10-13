from mme_vla_suite.models.config.utils import get_history_config


def test_wan_variant_loads():
    cfg = get_history_config("cosmo-wan-bottleneck-modul.yaml")
    assert cfg.representation_type == "cosmo_bottleneck"
    assert cfg.cosmo_bottleneck_memory.variant == "Wan2.2-TI2V-5B"
    assert cfg.cosmo_bottleneck_memory.temporal_factor == 4
    assert cfg.cosmo_bottleneck_memory.cosmos_latent_dim == 48


def test_wan_train_config_resolves():
    from mme_vla_suite.training.config import get_config
    c = get_config("mme_vla_suite_wan")
    assert c.model.history_config == "cosmo-wan-bottleneck-modul.yaml"
