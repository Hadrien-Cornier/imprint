import jax, jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from tests.cosmo import shapes as S

def _cfg():
    return OmegaConf.create({
        "budget": S.BOTTLENECK_TOKENS, "memory_token_dim": S.MEMORY_TOKEN_DIM,
        "cosmo_bottleneck_memory": {"num_latents": S.BOTTLENECK_TOKENS,
                                    "num_cross_blocks": 2, "num_heads": 8,
                                    "cosmos_latent_dim": S.COSMOS_LATENT_DIM,
                                    "siglip_dim": S.SIGLIP_DIM,
                                    "proprio_in_dim": S.PROPRIO_IN_DIM,
                                    "action_in_dim": S.ACTION_IN_DIM,
                                    "lang_emb_dim": S.LANG_EMB_DIM,
                                    "action_tokenizer_hidden": 512,
                                    "proprio_tokenizer_hidden": 512},
    })

def test_bottleneck_output_shape_and_mask_invariance():
    from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory
    m = CosmoBottleneckMemory(config=_cfg(), rngs=nnx.Rngs(0), dtype=jnp.float32)
    b, Lc, Lr, La, Lp = 2, 40, 8, 16, 16
    cosmo = jnp.asarray(np.random.randn(b, Lc, S.COSMOS_LATENT_DIM), jnp.float32)
    recent = jnp.asarray(np.random.randn(b, Lr, S.SIGLIP_DIM), jnp.float32)
    act = jnp.asarray(np.random.randn(b, La, S.ACTION_IN_DIM), jnp.float32)
    prop = jnp.asarray(np.random.randn(b, Lp, S.PROPRIO_IN_DIM), jnp.float32)
    lang = jnp.asarray(np.random.randn(b, S.LANG_EMB_DIM), jnp.float32)
    # last 5 consolidated latents are padding in BOTH calls (shared mask); the
    # invariance property is that the VALUES of masked/padding positions must
    # not change the output. (A prior draft compared all-valid vs 5-masked,
    # which is unsatisfiable for any softmax-attention bottleneck: attending to
    # 40 vs 35 valid keys fundamentally differs. See Task 4 report.)
    mask = jnp.ones((b, Lc), bool).at[:, -5:].set(False)
    tokens, out_mask, stats = m(cosmo, recent, act, prop, mask, lang)
    assert tokens.shape == (b, S.BOTTLENECK_TOKENS, S.MEMORY_TOKEN_DIM)
    # masked (padding) consolidated latents must not change the output
    cosmo2 = cosmo.at[:, -5:, :].set(999.0)
    mask2 = mask
    tokens2, _, _ = m(cosmo2, recent, act, prop, mask2, lang)
    np.testing.assert_allclose(np.asarray(tokens), np.asarray(tokens2), rtol=1e-4, atol=1e-4)

def test_language_conditioning_changes_output_and_dropout_is_identity():
    from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory
    m = CosmoBottleneckMemory(config=_cfg(), rngs=nnx.Rngs(0), dtype=jnp.float32)
    b, Lc = 2, 40
    cosmo = jnp.asarray(np.random.randn(b, Lc, S.COSMOS_LATENT_DIM), jnp.float32)
    recent = jnp.asarray(np.random.randn(b, 8, S.SIGLIP_DIM), jnp.float32)
    act = jnp.asarray(np.random.randn(b, 16, S.ACTION_IN_DIM), jnp.float32)
    prop = jnp.asarray(np.random.randn(b, 16, S.PROPRIO_IN_DIM), jnp.float32)
    mask = jnp.ones((b, Lc), bool)
    lang = jnp.asarray(np.random.randn(b, S.LANG_EMB_DIM), jnp.float32)
    t_lang, _, _ = m(cosmo, recent, act, prop, mask, lang)
    t_none, _, _ = m(cosmo, recent, act, prop, mask, None)   # language dropout -> FiLM identity
    assert not np.allclose(np.asarray(t_lang), np.asarray(t_none), rtol=1e-3, atol=1e-3)
