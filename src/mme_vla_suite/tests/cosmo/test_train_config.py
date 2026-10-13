def test_cosmo_train_config_resolves():
    from mme_vla_suite.training.config import get_config
    c = get_config("mme_vla_suite_cosmo")
    assert c.model.history_config == "cosmo-bottleneck-modul.yaml"
    assert c.model.use_history is True
    assert c.model.pi05 is True


def test_optimized_gated_delta_stage2_config_is_opt_in():
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.config import get_config

    original = get_config("mme_wan_stage2_pi05_gated_delta_memft")
    optimized = get_config("mme_wan_stage2_pi05_gated_delta_memft_optimized")
    history = get_history_config(optimized.model.history_config)

    assert original.length_bucketed_batches is False
    assert optimized.length_bucketed_batches is True
    assert history.cosmo_bottleneck_memory.memory_compute_dtype == "bfloat16"
