"""Time2Vec encoding (Kazemi et al. 2019, arXiv:1907.05321).

A learnable vector representation of a scalar time/elapsed value:

    t2v(tau)[0]   = w0 * tau + b0                 # linear term (absolute elapsed time)
    t2v(tau)[1:k] = sin(w_i * tau + b_i)          # periodic terms

Used as an ADD-ON for the memory encoder so each remembered frame can carry *when*
it happened. Gated by `memory_use_time` in the history config; when off, this module
is never constructed and behavior is identical to the original architecture.
"""
import jax.numpy as jnp
from flax import nnx

import openpi.shared.array_typing as at


class Time2Vec(nnx.Module):
    def __init__(self, out_dim: int, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        assert out_dim >= 2, "time2vec out_dim must be >= 2 (1 linear + >=1 periodic)"
        self.out_dim = out_dim
        # weights/biases for the linear term (index 0) and periodic terms (1..out_dim-1)
        self.w = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (out_dim,), dtype))
        self.b = nnx.Param(nnx.initializers.zeros(rngs.params(), (out_dim,), dtype))

    def __call__(self, tau: at.Float[at.Array, "*b 1"]) -> at.Float[at.Array, "*b d"]:
        # tau: (..., 1) scalar elapsed value (e.g. step_idx / (episode_len - 1) in [0,1])
        x = tau * self.w.value + self.b.value          # (..., out_dim) broadcast
        linear = x[..., :1]
        periodic = jnp.sin(x[..., 1:])
        return jnp.concatenate([linear, periodic], axis=-1)
