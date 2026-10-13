"""Perceiver-IO reconstruction decoder for bottleneck pretraining (Stage 1, doc §6.1/§6.2/§6.3).

Design (EgoWAM-inspired, RAE-DINO-head lineage):
  * We reconstruct the FULL spatial VLM-prefix grid per step (P=64 patch tokens, each
    2048-d), NOT a pooled global vector -- spatial layout ("where") is the recall signal
    memory must preserve. Queries are FACTORIZED: q[step,patch] = step_emb(step) +
    patch_emb(patch), plus one extra per-step SUMMARY token (step_emb + summary_emb) that
    feeds the per-step aux heads (action/proprio/reward).
  * Each of `num_blocks` blocks: (1) self-attention AMONG the P+1 tokens of the SAME step
    (a coherent feature map forms, patches + summary talk to each other, steps don't leak),
    (2) cross-attention from all query tokens into the 256 memory tokens (KV), (3) a WIDE
    FFN (ffn_mult x). Pre-norm residual throughout.
  * Positional info lives ONLY here (learned step + patch embeddings); the encoder memory
    stays time-agnostic. NO language input -- memory is language-conditioned upstream
    (Stage-1 bottleneck FiLM on). All heads are discarded at Stage-2 deployment.

Reconstruction is DETERMINISTIC (the memory already saw these frames -> single correct
target), so we regress with MSE rather than EgoWAM's flow-matching diffusion (which they
need only because they predict the multi-modal FUTURE frame).
"""
import jax
import jax.numpy as jnp
from flax import nnx
from mme_vla_suite.models.representation.utils import kernel_init


class _DecoderBlock(nnx.Module):
    """self-attn over the (P+1) tokens within each step -> cross-attn into memory -> wide FFN."""

    def __init__(self, dim, num_heads, ffn_inner, rngs, dtype):
        self.n_self = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, qkv_features=dim, out_features=dim,
            decode=False, rngs=rngs, dtype=dtype)
        self.n_cross_q = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.n_cross_kv = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, qkv_features=dim, out_features=dim,
            decode=False, rngs=rngs, dtype=dtype)
        self.n_ffn = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.ffn_in = nnx.Linear(dim, ffn_inner, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.ffn_out = nnx.Linear(ffn_inner, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, q, mem_kv):
        # q: (b, Q, T, dim)  T = P+1 tokens per step ; mem_kv: (b, M, dim)
        b, Q, T, d = q.shape
        # (1) self-attention within each step (batch steps into the leading axis)
        qs = self.n_self(q).reshape(b * Q, T, d)
        qs = self.self_attn(qs).reshape(b, Q, T, d)
        q = q + qs
        # (2) cross-attention: all query tokens attend the memory tokens
        qc = self.n_cross_q(q).reshape(b, Q * T, d)
        kv = self.n_cross_kv(mem_kv)
        qc = self.cross_attn(qc, kv).reshape(b, Q, T, d)
        q = q + qc
        # (3) wide FFN
        h = self.ffn_out(nnx.gelu(self.ffn_in(self.n_ffn(q))))
        q = q + h
        return q


class CosmoMemoryDecoder(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype=jnp.float32):
        cd = config.cosmo_decoder
        dim = int(config.memory_token_dim)
        self.dtype = dtype
        self.dim = dim
        self.num_patches = int(getattr(cd, "align_num_patches", 64))   # P (8x8 prefix grid)
        num_blocks = int(getattr(cd, "num_blocks", 2))
        num_heads = int(cd.num_heads)
        ffn_inner = int(getattr(cd, "ffn_inner", 4 * dim))

        self.step_emb = nnx.Embed(int(cd.max_query_steps), dim, rngs=rngs, dtype=dtype)
        self.patch_emb = nnx.Embed(self.num_patches, dim, rngs=rngs, dtype=dtype)
        # a dedicated per-step SUMMARY query (aggregates the step for action/proprio heads)
        self.summary_emb = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (dim,), dtype=dtype)
        )

        self.blocks = [
            _DecoderBlock(dim, num_heads, ffn_inner, rngs, dtype) for _ in range(num_blocks)
        ]
        self.out_norm = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)

        self.head_align = nnx.Linear(dim, int(cd.align_dim), rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.head_action = nnx.Linear(dim, int(cd.action_dim), rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.head_proprio = nnx.Linear(dim, int(cd.proprio_dim), rngs=rngs, dtype=dtype, kernel_init=kernel_init)

        # --- REWARD: a SINGLE global scalar in (0,1) for the CURRENT step, read from
        # memory via a dedicated learned query (NOT a per-step reconstruction). Reward
        # is the objective that genuinely NEEDS memory: the current frame alone is
        # ambiguous (a fruit back in its container looks the same washed or not) --
        # only the memory knows task progress. sigmoid bounds it to (0,1) so the MSE
        # can never explode. Its own cross-attn readout keeps it independent of the
        # per-step queries. ---
        self.reward_query = nnx.Param(0.02 * jax.random.normal(rngs.params(), (dim,), dtype=dtype))
        self.reward_norm_q = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.reward_norm_kv = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.reward_cross = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, qkv_features=dim, out_features=dim,
            decode=False, rngs=rngs, dtype=dtype)
        self.head_reward = nnx.Linear(dim, 1, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, mem_tokens, query_steps):
        # mem_tokens: (b, M, dim) language-conditioned ; query_steps: (b, Q) int
        b, Q = query_steps.shape
        P = self.num_patches
        dim = self.dim

        step = self.step_emb(query_steps)                       # (b, Q, dim)
        patch = self.patch_emb(jnp.arange(P))                   # (P, dim)
        summary = self.summary_emb.value[None, :]               # (1, dim)
        token_pos = jnp.concatenate([patch, summary], axis=0)   # (P+1, dim)

        # q[b,s,j] = step_emb(step) + token_pos[j]
        q = step[:, :, None, :] + token_pos[None, None, :, :]   # (b, Q, P+1, dim)

        for blk in self.blocks:
            q = blk(q, mem_tokens)
        q = self.out_norm(q)

        patch_tok = q[:, :, :P, :]        # (b, Q, P, dim) -> per-patch align target grid
        summ_tok = q[:, :, P, :]          # (b, Q, dim)    -> per-step action/proprio heads

        # reward: one query token per batch elem -> cross-attend memory -> sigmoid scalar.
        rq = jnp.broadcast_to(self.reward_query.value[None, None, :], (b, 1, dim))
        rq = rq + self.reward_cross(self.reward_norm_q(rq), self.reward_norm_kv(mem_tokens))
        reward = jax.nn.sigmoid(self.head_reward(rq)[:, 0, 0])   # (b,) in (0,1)

        return {
            "align": self.head_align(patch_tok),        # (b, Q, P, align_dim)
            "action": self.head_action(summ_tok),       # (b, Q, action_dim)
            "proprio": self.head_proprio(summ_tok),      # (b, Q, proprio_dim)
            "reward": reward,                            # (b,)  single current-step scalar
        }
