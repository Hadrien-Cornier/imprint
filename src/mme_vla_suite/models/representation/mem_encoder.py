import jax.numpy as jnp
from flax import nnx

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init
from mme_vla_suite.shared.time2vec import Time2Vec


class FeatureEncoder(nnx.Module):
    def __init__(
        self,
        rngs,
        dtype,
        image_input_dim,
        pos_input_dim,
        pos_output_dim,
        state_input_dim,
        state_output_dim,
        output_dim_for_percep=None,
        ouput_dim_for_recur=None,
        use_pos_emb=True,
        use_state_emb=False,
        use_time=False,
        time_out_dim=32,
        time_hidden_dim=128,
    ):
        self.use_pos_emb = use_pos_emb
        self.use_state_emb = use_state_emb
        self.use_time = use_time
        input_dim = image_input_dim
        if use_state_emb:
            self.state_proj = nnx.Linear(
                state_input_dim,
                state_output_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
            input_dim += state_output_dim
        if use_pos_emb:
            self.pos_proj = nnx.Linear(
                pos_input_dim,
                pos_output_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
            input_dim += pos_output_dim
        # time-aware add-on (gated): Time2Vec(elapsed) -> silu proj -> concat into each token
        if use_time:
            self.time2vec = Time2Vec(time_out_dim, rngs=rngs, dtype=dtype)
            self.time_proj = nnx.Linear(
                time_out_dim,
                time_hidden_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
            input_dim += time_hidden_dim

        if output_dim_for_percep is not None:
            self.encoder_static = nnx.Linear(
                input_dim,
                output_dim_for_percep,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
        else:
            self.encoder_static = None
        if ouput_dim_for_recur is not None:
            self.encoder_recur = nnx.Linear(
                input_dim,
                ouput_dim_for_recur,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
        else:
            self.encoder_recur = None

    def _add_pos_emb(
        self,
        base_emb,
        pos_emb,
    ):
        pos_emb = nnx.silu(self.pos_proj(pos_emb))
        base_emb = jnp.concatenate([base_emb, pos_emb], axis=-1)
        return base_emb

    def _add_state_emb(
        self,
        base_emb,
        state_emb,
    ):
        if base_emb.ndim == 5:
            _, _, v, p, _ = base_emb.shape
            state_emb = nnx.silu(self.state_proj(state_emb))
            state_emb = jnp.tile(state_emb[:, :, None, None, :], (1, 1, v, p, 1))
            base_emb = jnp.concatenate([base_emb, state_emb], axis=-1)
        else:
            state_emb = nnx.silu(self.state_proj(state_emb))
            base_emb = jnp.concatenate([base_emb, state_emb], axis=-1)

        return base_emb

    def _add_time_emb(
        self,
        base_emb,
        time_val,   # elapsed fraction, same per-frame layout as state: (..., 1)
    ):
        # Time2Vec(elapsed) -> silu proj, then concat with the same broadcast as state.
        time_emb = nnx.silu(self.time_proj(self.time2vec(time_val)))
        if base_emb.ndim == 5:
            _, _, v, p, _ = base_emb.shape
            time_emb = jnp.tile(time_emb[:, :, None, None, :], (1, 1, v, p, 1))
            base_emb = jnp.concatenate([base_emb, time_emb], axis=-1)
        else:
            base_emb = jnp.concatenate([base_emb, time_emb], axis=-1)
        return base_emb

    def _encode_memory(
        self,
        image_emb,
        pos_emb,
        state_emb,
        encoder_fn,
        time_val=None,
    ):
        input_emb = image_emb
        if self.use_pos_emb and pos_emb is not None:
            input_emb = self._add_pos_emb(input_emb, pos_emb)
        if self.use_state_emb and state_emb is not None:
            input_emb = self._add_state_emb(input_emb, state_emb)
        if self.use_time and time_val is not None:
            input_emb = self._add_time_emb(input_emb, time_val)
        input_emb = encoder_fn(input_emb)
        return input_emb

    def encode_perceptual_memory(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        static_time=None,
        *args,
        **kwargs,
    ):
        return self._encode_memory(
            static_image_emb,
            static_pos_emb,
            static_state_emb,
            self.encoder_static,
            time_val=static_time,
        )

    def encode_recurrent_memory(
        self,
        recurrent_image_emb: at.Float[at.Array, "b t v p d1"],
        recurrent_pos_emb: at.Float[at.Array, "b t v p d2"],
        recurrent_state_emb: at.Float[at.Array, "b t d3"],
        recurrent_time=None,
        *args,
        **kwargs,
    ):
        return self._encode_memory(
            recurrent_image_emb,
            recurrent_pos_emb,
            recurrent_state_emb,
            self.encoder_recur,
            time_val=recurrent_time,
        )