"""Learned 256-token Perceiver bottleneck (Option D).

Four KV streams: consolidated Cosmos latents, recent per-frame SigLIP tokens,
action tokens (executed action + Δa), proprio tokens (state + Δp). The 256
learned queries are language-conditioned via FiLM (doc §5.3); action/proprio
are embedded by small MLP tokenizers (doc §5.1). No type/time embeddings on the
memory tokens (doc `no_type_time`) — modality identity comes only from the
separate per-stream projections.
"""
import jax.numpy as jnp
import numpy as np
from flax import nnx
import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init


class _MlpTokenizer(nnx.Module):
    """Small MLP that embeds a per-step action/proprio vector into the bottleneck dim."""
    def __init__(self, in_dim, hidden, out_dim, rngs, dtype):
        self.l1 = nnx.Linear(in_dim, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.l2 = nnx.Linear(hidden, out_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, x):
        return self.l2(nnx.gelu(self.l1(x)))


class _CrossBlock(nnx.Module):
    def __init__(self, dim, num_heads, rngs, dtype):
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=dim,
                                           qkv_features=dim, out_features=dim,
                                           decode=False, rngs=rngs, dtype=dtype)
        self.self_attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=dim,
                                                qkv_features=dim, out_features=dim,
                                                decode=False, rngs=rngs, dtype=dtype)
        self.norm_q = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.norm_s = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.ffn = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, q, kv, kv_mask):
        # cross-attend queries -> kv (kv_mask: (b, Lkv) True=valid). flax MHA
        # takes a boolean mask broadcast to (b, heads, q, Lkv), True=attend.
        m = kv_mask[:, None, None, :]  # (b,1,1,Lkv) broadcast over heads,q
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), mask=m)
        q = q + self.self_attn(self.norm_s(q), self.norm_s(q))
        q = q + jnp.tanh(self.ffn(q))
        return q


class CosmoBottleneckMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        cb = config.cosmo_bottleneck_memory
        dim = int(config.memory_token_dim)
        self.num_latents = int(cb.num_latents)
        # per-stream projections into the shared bottleneck dim
        self.proj_cosmo = nnx.Linear(int(cb.cosmos_latent_dim), dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.proj_recent = nnx.Linear(int(cb.siglip_dim), dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        # MLP tokenizers for action/proprio (doc §5.1)
        self.tok_action = _MlpTokenizer(int(cb.action_in_dim), int(cb.action_tokenizer_hidden), dim, rngs, dtype)
        self.tok_proprio = _MlpTokenizer(int(cb.proprio_in_dim), int(cb.proprio_tokenizer_hidden), dim, rngs, dtype)
        # language -> FiLM(gamma,beta) over the query latents (doc §5.3).
        # Small-nonzero kernel (not zeros): a random lang embedding produces
        # nonzero γ,β so language actually conditions the queries, while the
        # kernel is small enough to stay near-identity in expectation at init.
        # Bias stays zero-init so the lang_emb=None (language-dropout) path is
        # exactly identity: no FiLM is applied at all.
        # Use the module-level `kernel_init` SINGLETON (also normal stddev=0.02), NOT a fresh
        # inline nnx.initializers.normal(...): an inline closure is a new object each
        # cost.model.create(), and nnx embeds the initializer in the graphdef's static
        # metadata by identity. Two init builds (eval_shape out_shardings vs the jitted
        # init) would then produce unequal graphdefs -> jax.jit "different pytree metadata
        # at out_shardings.model_def". A shared singleton keeps both graphdefs identical.
        self.lang_film = nnx.Linear(int(cb.lang_emb_dim), 2 * dim, rngs=rngs, dtype=dtype,
                                    kernel_init=kernel_init)
        self.latents = nnx.Param(0.02 * jnp.asarray(
            np.random.default_rng(0).standard_normal((self.num_latents, dim)), dtype))
        # Plain Python list -> nnx assigns INTEGER child keys (blocks/0/..., blocks/1/...),
        # matching openpi's own module-list convention. This is REQUIRED for Stage-2:
        # flax's State.replace_by_pure_dict (used by train.py init) runs int() on every
        # numeric path component, so the native state MUST key blocks by int, not str
        # (an nnx.Dict with string "0"/"1" keys makes replace_by_pure_dict raise
        # "key not available in state"). The Stage-1 checkpoint stores string keys
        # "bottleneck/blocks/0/..."; Stage1BottleneckWeightLoader remaps them to
        # int-keyed tuples on load (int-key-safe tuple flatten/unflatten), so no retrain.
        self.num_cross_blocks = int(cb.num_cross_blocks)
        self.blocks = [_CrossBlock(dim, int(cb.num_heads), rngs, dtype)
                       for _ in range(self.num_cross_blocks)]
        self.dtype = dtype

    def __call__(self, cosmo_emb, recent_emb, action_emb, proprio_emb, cosmo_mask,
                 lang_emb=None, recent_mask=None, action_mask=None, proprio_mask=None):
        b = cosmo_emb.shape[0]
        dim = self.latents.value.shape[-1]
        kv = jnp.concatenate([
            self.proj_cosmo(cosmo_emb), self.proj_recent(recent_emb),
            self.tok_action(action_emb), self.tok_proprio(proprio_emb)], axis=1)
        # FIX(bug2): each stream carries its OWN validity mask. Padded recent/action/proprio
        # slots (the loader pads to fixed MAX_RECENT/MAX_STEPS) are NOT real memory and MUST
        # be masked out of the cross-attention -- otherwise the zero-filled pads (the majority
        # of the KV, since real recent ~[192..576] vs cap 2048) dilute the true memory tokens.
        # Absent mask (e.g. inference passes dense streams) -> all-valid, matching old behavior.
        def _m(msk, L):
            return msk if msk is not None else jnp.ones((b, L), bool)
        kv_mask = jnp.concatenate([
            cosmo_mask,
            _m(recent_mask, recent_emb.shape[1]),
            _m(action_mask, action_emb.shape[1]),
            _m(proprio_mask, proprio_emb.shape[1])], axis=1)
        q = jnp.broadcast_to(self.latents.value[None], (b, self.num_latents, dim))
        # language-conditioned queries via FiLM; lang_emb=None -> identity (language dropout ablation)
        if lang_emb is not None:
            gamma, beta = jnp.split(self.lang_film(lang_emb.astype(self.dtype)), 2, axis=-1)
            q = q * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        for blk in self.blocks:
            q = blk(q, kv, kv_mask)
        return q, jnp.ones((b, self.num_latents), bool), None
