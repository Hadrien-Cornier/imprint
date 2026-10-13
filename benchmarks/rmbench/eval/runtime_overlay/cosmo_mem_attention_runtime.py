"""Cosmo bottleneck runtime with read-only cross-attention tracing.

The policy action path still uses ``CosmoBottleneckMemory.__call__``. The
``attention_trace`` method is a separate deterministic forward pass used only
by instrumented evaluations.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention_weights

from mme_vla_suite.models.representation.utils import kernel_init


class _MlpTokenizer(nnx.Module):
    def __init__(self, in_dim, hidden, out_dim, rngs, dtype):
        self.l1 = nnx.Linear(
            in_dim,
            hidden,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.l2 = nnx.Linear(
            hidden,
            out_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(self, x):
        return self.l2(nnx.gelu(self.l1(x)))


class _CrossBlock(nnx.Module):
    def __init__(self, dim, num_heads, rngs, dtype):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
            dtype=dtype,
        )
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
            dtype=dtype,
        )
        self.norm_q = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.norm_s = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.ffn = nnx.Linear(
            dim,
            dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(self, q, kv, kv_mask):
        mask = kv_mask[:, None, None, :]
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), mask=mask)
        q = q + self.self_attn(self.norm_s(q), self.norm_s(q))
        q = q + jnp.tanh(self.ffn(q))
        return q

    def cross_attention_weights(self, q, kv, kv_mask):
        """Return pre-dropout cross-attention weights and the normal block output."""
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        query = self.attn.query(q_norm)
        key = self.attn.key(kv_norm)
        if self.attn.normalize_qk:
            query = self.attn.query_ln(query)
            key = self.attn.key_ln(key)
        mask = kv_mask[:, None, None, :]
        weights = dot_product_attention_weights(
            query,
            key,
            mask=mask,
            deterministic=True,
            dtype=self.attn.dtype,
            precision=self.attn.precision,
        )
        q = q + self.attn(q_norm, kv_norm, mask=mask)
        q = q + self.self_attn(self.norm_s(q), self.norm_s(q))
        q = q + jnp.tanh(self.ffn(q))
        return q, weights


class CosmoBottleneckMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        cb = config.cosmo_bottleneck_memory
        dim = int(config.memory_token_dim)
        self.num_latents = int(cb.num_latents)
        self.proj_cosmo = nnx.Linear(
            int(cb.cosmos_latent_dim),
            dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.proj_recent = nnx.Linear(
            int(cb.siglip_dim),
            dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.tok_action = _MlpTokenizer(
            int(cb.action_in_dim),
            int(cb.action_tokenizer_hidden),
            dim,
            rngs,
            dtype,
        )
        self.tok_proprio = _MlpTokenizer(
            int(cb.proprio_in_dim),
            int(cb.proprio_tokenizer_hidden),
            dim,
            rngs,
            dtype,
        )
        self.lang_film = nnx.Linear(
            int(cb.lang_emb_dim),
            2 * dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.latents = nnx.Param(
            0.02
            * jnp.asarray(
                np.random.default_rng(0).standard_normal(
                    (self.num_latents, dim)
                ),
                dtype,
            )
        )
        self.num_cross_blocks = int(cb.num_cross_blocks)
        self.blocks = [
            _CrossBlock(dim, int(cb.num_heads), rngs, dtype)
            for _ in range(self.num_cross_blocks)
        ]
        self.dtype = dtype

    @staticmethod
    def _mask_or_ones(mask, batch_size, length):
        if mask is not None:
            return mask
        return jnp.ones((batch_size, length), dtype=jnp.bool_)

    def _prepare_inputs(
        self,
        cosmo_emb,
        recent_emb,
        action_emb,
        proprio_emb,
        cosmo_mask,
        lang_emb,
        recent_mask,
        action_mask,
        proprio_mask,
    ):
        batch_size = cosmo_emb.shape[0]
        dim = self.latents.value.shape[-1]
        kv = jnp.concatenate(
            [
                self.proj_cosmo(cosmo_emb),
                self.proj_recent(recent_emb),
                self.tok_action(action_emb),
                self.tok_proprio(proprio_emb),
            ],
            axis=1,
        )
        kv_mask = jnp.concatenate(
            [
                cosmo_mask,
                self._mask_or_ones(
                    recent_mask, batch_size, recent_emb.shape[1]
                ),
                self._mask_or_ones(
                    action_mask, batch_size, action_emb.shape[1]
                ),
                self._mask_or_ones(
                    proprio_mask, batch_size, proprio_emb.shape[1]
                ),
            ],
            axis=1,
        )
        q = jnp.broadcast_to(
            self.latents.value[None],
            (batch_size, self.num_latents, dim),
        )
        if lang_emb is not None:
            gamma, beta = jnp.split(
                self.lang_film(lang_emb.astype(self.dtype)),
                2,
                axis=-1,
            )
            q = q * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return q, kv, kv_mask

    def __call__(
        self,
        cosmo_emb,
        recent_emb,
        action_emb,
        proprio_emb,
        cosmo_mask,
        lang_emb=None,
        recent_mask=None,
        action_mask=None,
        proprio_mask=None,
    ):
        q, kv, kv_mask = self._prepare_inputs(
            cosmo_emb,
            recent_emb,
            action_emb,
            proprio_emb,
            cosmo_mask,
            lang_emb,
            recent_mask,
            action_mask,
            proprio_mask,
        )
        for block in self.blocks:
            q = block(q, kv, kv_mask)
        return q, jnp.ones((q.shape[0], self.num_latents), bool), None

    def attention_trace(
        self,
        cosmo_emb,
        recent_emb,
        action_emb,
        proprio_emb,
        cosmo_mask,
        lang_emb=None,
        recent_mask=None,
        action_mask=None,
        proprio_mask=None,
    ):
        """Return cross-attention summaries without altering the policy output."""
        q, kv, kv_mask = self._prepare_inputs(
            cosmo_emb,
            recent_emb,
            action_emb,
            proprio_emb,
            cosmo_mask,
            lang_emb,
            recent_mask,
            action_mask,
            proprio_mask,
        )
        block_weights = []
        for block in self.blocks:
            q, weights = block.cross_attention_weights(q, kv, kv_mask)
            block_weights.append(weights.astype(jnp.float32))
        # Full shape before reduction: B, blocks, heads, latent queries, KV.
        weights = jnp.stack(block_weights, axis=1)
        return {
            "token_attention": jnp.mean(weights, axis=(1, 2, 3)),
            "block_token_attention": jnp.mean(weights, axis=(2, 3)),
            "head_token_attention": jnp.mean(weights, axis=(1, 3)),
            "kv_mask": kv_mask,
        }
