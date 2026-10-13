import jax, jax.numpy as jnp, numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf
from tests.cosmo import shapes as S

def _cfg():
    return OmegaConf.create({
        "budget": S.BOTTLENECK_TOKENS, "memory_token_dim": 1024,   # modulation seam width
        "cosmo_bottleneck_memory": {"num_latents": S.BOTTLENECK_TOKENS, "num_cross_blocks": 2,
            "num_heads": 8, "cosmos_latent_dim": S.WAN_LATENT_DIM, "siglip_dim": S.SIGLIP_DIM,
            "proprio_in_dim": S.PROPRIO_IN_DIM, "action_in_dim": S.ACTION_IN_DIM,
            "lang_emb_dim": S.LANG_EMB_DIM, "action_tokenizer_hidden": 512, "proprio_tokenizer_hidden": 512},
        "cosmo_decoder": {"num_heads": 8, "num_blocks": 1, "align_dim": S.SIGLIP_DIM,
            "action_dim": 12, "proprio_dim": 16, "max_query_steps": S.MAX_HISTORY_STEPS}})

def _fake_batch(b=2, Lc=40, Lr=8, La=16, Lp=16, Q=6):
    rng = np.random.default_rng(0)
    f = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)
    return {
        "cosmo_latent_emb": f(b, Lc, S.WAN_LATENT_DIM),
        "cosmo_recent_emb": f(b, Lr, S.SIGLIP_DIM),
        "cosmo_action_emb": f(b, La, S.ACTION_IN_DIM),
        "cosmo_proprio_emb": f(b, Lp, S.PROPRIO_IN_DIM),
        "cosmo_mem_mask": jnp.ones((b, Lc), bool),
        "cosmo_lang_emb": f(b, S.LANG_EMB_DIM),
        "query_steps": jnp.asarray(np.tile(np.arange(Q), (b,1))),
        "align_target": f(b, Q, S.SIGLIP_DIM),
        "action_target": f(b, Q, 12),
        "proprio_target": f(b, Q, 16),
    }

def test_composite_loss_decreases_on_overfit():
    from mme_vla_suite.training.pretrain_loss import make_stage1_modules, composite_alignment_loss
    cfg = _cfg(); modules = make_stage1_modules(cfg, seed=0)
    batch = _fake_batch()
    weights = {"align": 1.0, "action": 1.0, "proprio": 1.0}   # reward off (no labels)
    # nnx training: build an optimizer over both modules and run a few steps.
    graphdef, state = nnx.split((modules["bottleneck"], modules["decoder"]))
    tx = optax.adam(1e-3); opt_state = tx.init(state)
    def loss_of(state):
        bmk, dec = nnx.merge(graphdef, state)
        m = {"bottleneck": bmk, "decoder": dec}
        total, _ = composite_alignment_loss(m, batch, weights)
        return total
    l0 = float(loss_of(state))
    for _ in range(30):
        g = jax.grad(loss_of)(state)
        updates, opt_state = tx.update(g, opt_state, state)
        state = optax.apply_updates(state, updates)
    l1 = float(loss_of(state))
    assert l1 < 0.9 * l0, f"loss did not drop: {l0} -> {l1}"

def test_reward_term_active():
    from mme_vla_suite.training.pretrain_loss import make_stage1_modules, composite_alignment_loss
    cfg = _cfg(); modules = make_stage1_modules(cfg, seed=0)
    batch = _fake_batch()
    batch = dict(batch); batch["reward_target"] = jnp.asarray(np.random.default_rng(1).standard_normal((2,6)), jnp.float32)
    _, parts_off = composite_alignment_loss(modules, batch, {"align":1.,"action":1.,"proprio":1.})
    _, parts_on  = composite_alignment_loss(modules, batch, {"align":1.,"action":1.,"proprio":1.,"reward":1.})
    assert "reward" not in parts_off        # w_reward absent -> reward term skipped
    assert "reward" in parts_on and float(parts_on["reward"]) >= 0.0
