from mme_vla_suite.models.config.utils import get_history_config

def test_cosmo_variant_loads():
    cfg = get_history_config("cosmo-bottleneck-modul.yaml")
    assert cfg.representation_type == "cosmo_bottleneck"
    assert cfg.integration_type == "modulation"
    assert cfg.memory_token_dim == 1024
    assert cfg.budget == 256
    assert cfg.cosmo_bottleneck_memory.num_latents == 256
    assert cfg.cosmo_bottleneck_memory.cosmos_latent_dim == 16
    assert cfg.cosmo_bottleneck_memory.action_in_dim == 24
    assert cfg.cosmo_bottleneck_memory.proprio_in_dim == 32
    assert cfg.cosmo_bottleneck_memory.lang_emb_dim == 2048
