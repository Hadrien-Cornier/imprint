import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from tests.cosmo import shapes as S


def _cfg():
    return OmegaConf.create({"memory_token_dim": S.MEMORY_TOKEN_DIM,
        "cosmo_decoder": {"num_heads": 8, "num_blocks": 1,
                          "align_dim": S.SIGLIP_DIM, "action_dim": 12, "proprio_dim": 16,
                          "max_query_steps": S.MAX_HISTORY_STEPS}})


def test_decoder_reconstructs_per_step_targets():
    from mme_vla_suite.models.representation.cosmo_decoder import CosmoMemoryDecoder
    d = CosmoMemoryDecoder(config=_cfg(), rngs=nnx.Rngs(0), dtype=jnp.float32)
    b, M, dim = 2, S.BOTTLENECK_TOKENS, S.MEMORY_TOKEN_DIM
    mem = jnp.asarray(np.random.randn(b, M, dim), jnp.float32)   # 256 memory tokens
    steps = jnp.asarray(np.tile(np.arange(5), (b, 1)))          # (b,5) queried step ids
    out = d(mem, steps)                                         # NO language input
    assert out["align"].shape == (b, 5, S.SIGLIP_DIM)
    assert out["action"].shape == (b, 5, 12)
    assert out["proprio"].shape == (b, 5, 16)
    assert out["reward"].shape == (b, 5)
    # positional query works: different queried steps -> different outputs
    o0 = d(mem, steps[:, :1]); o1 = d(mem, steps[:, 1:2])
    assert not np.allclose(np.asarray(o0["align"]), np.asarray(o1["align"]), atol=1e-4)
    # reward reads from memory (task-relativity via language-conditioned memory upstream):
    mem2 = jnp.asarray(np.random.randn(b, M, dim), jnp.float32)
    assert not np.allclose(np.asarray(out["reward"]), np.asarray(d(mem2, steps)["reward"]), atol=1e-4)
