import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tests.cosmo import shapes as S


def test_factory_constructs_cosmo_bottleneck():
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
    cfg = HistoryPi0Config(use_history=True, pi05=True,
                           history_config="cosmo-bottleneck-modul.yaml")
    # `create` loads the YAML history_config into a DictConfig and builds the model.
    model = cfg.create(jax.random.key(0))
    assert model.representation_type == "cosmo_bottleneck"
    assert model.mem_encoder.__class__.__name__ == "CosmoBottleneckMemory"


def test_observation_carries_cosmo_fields():
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    # the dataclass must accept the new cosmo fields and round-trip via to_dict
    b = 1
    obs = HistAugObservation(
        images={}, image_masks={}, state=jnp.zeros((b, 16)),
        tokenized_prompt=None, tokenized_prompt_mask=None,
        token_ar_mask=None, token_loss_mask=None,
        cosmo_latent_emb=jnp.zeros((b, 4, S.COSMOS_LATENT_DIM)),
        cosmo_recent_emb=jnp.zeros((b, 8, S.SIGLIP_DIM)),
        cosmo_mem_mask=jnp.ones((b, 4), bool),
        cosmo_action_emb=jnp.zeros((b, 4, S.ACTION_IN_DIM)),
        cosmo_proprio_emb=jnp.zeros((b, 4, S.PROPRIO_IN_DIM)),
        cosmo_lang_emb=jnp.zeros((b, S.LANG_EMB_DIM)),
    )
    d = obs.to_dict()
    assert d["cosmo_latent_emb"].shape == (b, 4, S.COSMOS_LATENT_DIM)
    assert d["cosmo_lang_emb"].shape == (b, S.LANG_EMB_DIM)
