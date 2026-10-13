"""Stage-1 composite pretraining objective (doc §6.4), FOUR objectives.

L = w_align * align_loss + w_action * MSE(action) + w_proprio * MSE(proprio)
    + w_reward * MSE(reward)

align_loss: per-token LayerNorm (EgoWAM/RAE DINO-head recipe) on BOTH the decoder output
  and the precomputed VLM-prefix target, then MSE over the full (Q, P=64, 2048) grid.
  Per-token LN makes the loss scale-invariant across patches and tames gemma_2b prefix
  magnitude imbalance; the target LN is under stop_gradient (target is a constant).

All losses are MEAN-reduced (per element), so each objective is O(1) regardless of its
dimensionality (align has 64*2048 vs action's 12); weights express INTENT, not scale
correction. action/proprio targets are unit-scaled upstream; reward target is the
standardized MEAN of the 3 RoboDopamine modes.

Stage-1 = OPTION B: bottleneck FiLM ON (memory language-conditioned); decoder language-free.
Only bottleneck+decoder train; targets are precomputed & treated as constants.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory
from mme_vla_suite.models.representation.cosmo_decoder import CosmoMemoryDecoder


def make_stage1_modules(config, seed=0):
    rngs = nnx.Rngs(seed)
    return {"bottleneck": CosmoBottleneckMemory(config, rngs=rngs),
            "decoder": CosmoMemoryDecoder(config, rngs=rngs)}


def _layernorm_lastdim(x, eps=1e-6):
    """Parameter-free per-token LayerNorm over the feature (last) axis."""
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps)


def composite_alignment_loss(modules, batch, weights):
    # FIX(bug2): thread the per-stream validity masks so padded recent/action/proprio
    # slots are excluded from the bottleneck cross-attention.
    # FIX(bug4): pass the real per-episode language embedding -> FiLM path trains.
    mem, _, _ = modules["bottleneck"](
        batch["cosmo_latent_emb"], batch["cosmo_recent_emb"],
        batch["cosmo_action_emb"], batch["cosmo_proprio_emb"],
        batch["cosmo_mem_mask"], lang_emb=batch.get("cosmo_lang_emb"),
        recent_mask=batch.get("cosmo_recent_mask"),
        action_mask=batch.get("cosmo_action_mask"),
        proprio_mask=batch.get("cosmo_proprio_mask"))   # FiLM ON
    out = modules["decoder"](mem, batch["query_steps"])              # no language

    def mse(a, b):
        return jnp.mean((a - b) ** 2)

    # FIX(bug3): mask padded queries. The loader pads Q to a fixed tier; padded query
    # rows carry zero targets that would otherwise pull the reconstruction toward 0.
    # Reduce each per-query error to a scalar-per-query, then average over REAL queries
    # only (query_mask: (b,Q) True=real). Guard the denominator against an all-pad batch.
    qm = batch.get("query_mask")

    def masked_mean(err):
        # err: (b, Q, ...) elementwise squared error -> mean over feature axes -> (b, Q)
        feat_axes = tuple(range(2, err.ndim))
        per_q = jnp.mean(err, axis=feat_axes) if feat_axes else err
        if qm is None:
            return jnp.mean(per_q)
        m = qm.astype(per_q.dtype)
        return jnp.sum(per_q * m) / jnp.clip(jnp.sum(m), 1.0, None)

    # --- align: per-token LayerNorm on pred + target, then masked-MSE over (b,Q,P,2048) ---
    pred_a = _layernorm_lastdim(out["align"])
    tgt_a = jax.lax.stop_gradient(_layernorm_lastdim(batch["align_target"]))
    l_align = masked_mean((pred_a - tgt_a) ** 2)

    l_action = masked_mean((out["action"] - batch["action_target"]) ** 2)
    l_prop = masked_mean((out["proprio"] - batch["proprio_target"]) ** 2)
    parts = {"align": l_align, "action": l_action, "proprio": l_prop}
    total = weights["align"] * l_align + weights["action"] * l_action + weights["proprio"] * l_prop

    if weights.get("reward", 0.0) > 0.0 and "reward_target" in batch:
        # reward is a SINGLE current-step scalar in (0,1); pred is sigmoid-bounded so
        # this MSE is <= 1 and cannot explode. NO standardization (target already in (0,1)).
        l_reward = mse(out["reward"], batch["reward_target"])   # both (b,)
        parts["reward"] = l_reward
        total = total + weights["reward"] * l_reward
    return total, parts
