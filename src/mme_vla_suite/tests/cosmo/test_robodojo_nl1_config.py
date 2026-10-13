from omegaconf import OmegaConf

from mme_vla_suite.models.config.utils import get_history_config


SOURCE_CONFIG = "cosmo-wan-bottleneck-modul-robodojo-s2nojitter.yaml"
NL1_CONFIG = "cosmo-wan-bottleneck-modul-robodojo-s2nojitter-nl1.yaml"


def test_robodojo_nl1_config_only_changes_num_latents():
    source = OmegaConf.to_container(
        get_history_config(SOURCE_CONFIG), resolve=True
    )
    nl1 = OmegaConf.to_container(get_history_config(NL1_CONFIG), resolve=True)

    source["cosmo_bottleneck_memory"]["num_latents"] = 1
    assert nl1 == source
