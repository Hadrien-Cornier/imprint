import flax.nnx as nnx
import jax.numpy as jnp


import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder


class PerceptualMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.mem_type = config.perceptual_memory.type

        # time-aware add-on (gated): read flags/dims with safe defaults so older configs
        # (without `memory_use_time` / `memory_feature.time`) behave exactly as before.
        use_time = bool(getattr(config, "memory_use_time", False))
        time_cfg = getattr(config.memory_feature, "time", None)
        time_out_dim = int(getattr(time_cfg, "out_dim", 32)) if time_cfg is not None else 32
        time_hidden_dim = int(getattr(time_cfg, "hidden_dim", 128)) if time_cfg is not None else 128
        self.use_time = use_time

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.memory_feature.img.input_dim,
            pos_input_dim=self.config.memory_feature.pos.input_dim,
            state_input_dim=self.config.memory_feature.state.input_dim,
            pos_output_dim=self.config.memory_feature.pos.hidden_dim,
            state_output_dim=self.config.memory_feature.state.hidden_dim,
            ouput_dim_for_recur=None,
            output_dim_for_percep=self.config.memory_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
            use_time=use_time,
            time_out_dim=time_out_dim,
            time_hidden_dim=time_hidden_dim,
        )

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        static_time=None,
    ):
        # get memory tokens using feature encoder
        assert static_image_emb.shape[1] == self.config.budget

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb, static_pos_emb, static_state_emb, static_time=static_time
        )

        return hidden_states, None, None
