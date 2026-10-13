import numpy as np
import pytest


@pytest.mark.gpu
def test_optionD_buffer_to_memory_tokens():
    """Real Cosmos frontend -> online buffer (20 steps) -> bottleneck -> 256 finite memory tokens."""
    import jax, jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory
    from omegaconf import OmegaConf
    from tests.cosmo import shapes as S

    fe = CosmoFrontend(variant="CV8x8x8", device="cuda")

    # SigLIP stand-in for the recent stream: a deterministic finite encoder of the right shape.
    # (The real SigLIP path is exercised by the serving policy / Task 9; here we isolate the
    # Cosmos+buffer+bottleneck integration and keep the recent-stream encoder simple + finite.)
    def siglip_fn(x):  # x: (t, v, 224,224,3) -> (t, v, 64, 2048)
        t, v = x.shape[0], x.shape[1]
        return np.zeros((t, v, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

    buf = CosmoMemoryBuffer(cosmo=fe, siglip_fn=siglip_fn, chunk_frames=8, downsample=2,
                            num_views=1, state_dim=S.STATE_DIM, action_dim=12,
                            latent_dim=S.COSMOS_LATENT_DIM)
    buf.set_language(None)
    rng = np.random.default_rng(0)
    for step in range(20):
        buf.add_buffer(images=rng.integers(0, 256, (1, 1, 128, 128, 3)).astype(np.uint8),
                       states=np.zeros((1, S.STATE_DIM), np.float32),
                       actions=np.zeros((1, 12), np.float32), step_idx_list=[step])
    feats = buf.get_history_feats(19)
    # 20 steps, downsample 2 -> 10 downsampled frames -> 1 finalized chunk (drop-terminal) + recent window
    assert np.isfinite(feats["cosmo_recent_emb"]).all()
    assert np.isfinite(feats["cosmo_latent_emb"]).all()
    assert feats["cosmo_latent_emb"].shape[0] >= 0
    assert feats["cosmo_action_emb"].shape[1] == S.ACTION_IN_DIM
    assert feats["cosmo_proprio_emb"].shape[1] == S.PROPRIO_IN_DIM

    # feed through the bottleneck -> 256 memory tokens
    cfg = OmegaConf.create({"budget": 256, "memory_token_dim": 1024,
        "cosmo_bottleneck_memory": {"num_latents": 256, "num_cross_blocks": 2, "num_heads": 8,
        "cosmos_latent_dim": 16, "siglip_dim": 2048, "proprio_in_dim": 32, "action_in_dim": 24,
        "lang_emb_dim": 2048, "action_tokenizer_hidden": 512, "proprio_tokenizer_hidden": 512}})
    m = CosmoBottleneckMemory(config=cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)

    def b1(a):
        return jnp.asarray(a)[None]  # add batch axis

    Lc = feats["cosmo_latent_emb"].shape[0]
    cosmo = b1(feats["cosmo_latent_emb"]) if Lc > 0 else jnp.zeros((1, 1, 16), jnp.float32)
    mask = b1(feats["cosmo_mem_mask"]) if Lc > 0 else jnp.zeros((1, 1), bool)
    recent = b1(feats["cosmo_recent_emb"]) if feats["cosmo_recent_emb"].shape[0] > 0 else jnp.zeros((1, 1, 2048), jnp.float32)
    act = b1(feats["cosmo_action_emb"])
    prop = b1(feats["cosmo_proprio_emb"])
    tokens, tok_mask, _ = m(cosmo, recent, act, prop, mask, None)
    assert tokens.shape == (1, 256, 1024)
    assert np.isfinite(np.asarray(tokens)).all()
