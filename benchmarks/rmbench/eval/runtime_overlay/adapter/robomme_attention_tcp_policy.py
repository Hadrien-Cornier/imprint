"""Instrumented RoboMME adapter for memory-encoder attention visualization."""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

import robomme_tcp_policy as base


class AttentionRoboMMETCPServerModel(base.RoboMMETCPServerModel):
    def initialize_attention_capture(self) -> None:
        from openpi.shared import nnx_utils

        trace_dir = os.environ.get("RMBENCH_ATTENTION_TRACE_DIR")
        if not trace_dir:
            raise ValueError(
                "RMBENCH_ATTENTION_TRACE_DIR is required by the attention adapter"
            )
        self._attention_trace_dir = pathlib.Path(trace_dir)
        self._attention_trace_dir.mkdir(parents=True, exist_ok=True)
        encoder = self.policy._model.mem_encoder
        if not hasattr(encoder, "attention_trace"):
            raise RuntimeError(
                "instrumented CosmoBottleneckMemory.attention_trace is missing"
            )
        self._memory_attention = nnx_utils.module_jit(
            encoder.attention_trace
        )
        self._attention_call_index = 0

    def _attention_observation(
        self,
        frames: np.ndarray,
        state: np.ndarray,
    ):
        import jax
        import jax.numpy as jnp
        from mme_vla_suite.models.integration.history_observation import (
            HistAugObservation,
        )

        inputs = {
            "observation/image": frames[0],
            "observation/wrist_image": frames[1],
            "observation/right_image": frames[2],
            "observation/state": state,
            "prompt": self._prompt,
        }
        inputs = self.policy._prepare_history(inputs)
        inputs = self.policy._input_transform(inputs)
        return HistAugObservation.from_dict(
            jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
        )

    def _capture_memory_attention(self, observation) -> None:
        import jax

        frames, state = base.extract_observation(observation)
        model_obs = self._attention_observation(frames, state)
        trace = self._memory_attention(
            model_obs.cosmo_latent_emb,
            model_obs.cosmo_recent_emb,
            model_obs.cosmo_action_emb,
            model_obs.cosmo_proprio_emb,
            model_obs.cosmo_mem_mask,
            lang_emb=model_obs.cosmo_lang_emb,
            recent_mask=model_obs.cosmo_recent_mask,
            action_mask=model_obs.cosmo_action_mask,
            proprio_mask=model_obs.cosmo_proprio_mask,
        )
        trace = jax.tree.map(
            lambda value: np.asarray(jax.device_get(value))[0],
            trace,
        )
        buffer = self.policy.mem_buffer
        history_frames = np.stack(buffer._all_ds_frames_hi, axis=0)
        history_states = np.stack(buffer._states, axis=0)
        history_actions = np.stack(buffer._actions, axis=0)
        history_steps = (
            np.arange(history_frames.shape[0], dtype=np.int32)
            * int(buffer.downsample)
        )
        destination = (
            self._attention_trace_dir
            / f"call_{self._attention_call_index:03d}.npz"
        )
        np.savez_compressed(
            destination,
            token_attention=trace["token_attention"],
            block_token_attention=trace["block_token_attention"],
            head_token_attention=trace["head_token_attention"],
            kv_mask=trace["kv_mask"],
            latent_mask=np.asarray(model_obs.cosmo_mem_mask[0]),
            recent_mask=np.asarray(model_obs.cosmo_recent_mask[0]),
            action_mask=np.asarray(model_obs.cosmo_action_mask[0]),
            proprio_mask=np.asarray(model_obs.cosmo_proprio_mask[0]),
            current_frames=frames,
            current_state=state,
            history_frames=history_frames,
            history_states=history_states,
            history_actions=history_actions,
            history_steps=history_steps,
            generation_step=np.asarray(self.policy.step_idx, dtype=np.int32),
            consolidated_chunks=np.asarray(
                len(buffer._consolidated), dtype=np.int32
            ),
            recent_start_index=np.asarray(
                len(buffer._consolidated) * int(buffer.chunk_frames),
                dtype=np.int32,
            ),
            chunk_frames=np.asarray(buffer.chunk_frames, dtype=np.int32),
            downsample=np.asarray(buffer.downsample, dtype=np.int32),
            num_views=np.asarray(buffer.num_views, dtype=np.int32),
            visual_tokens_per_recent_frame=np.asarray(64, dtype=np.int32),
            wan_tokens_per_chunk_view=np.asarray(
                int(buffer.pool_h or buffer.pool_hw)
                * int(buffer.pool_w or buffer.pool_hw),
                dtype=np.int32,
            ),
        )
        metadata = {
            "call_index": self._attention_call_index,
            "generation_step": int(self.policy.step_idx),
            "attention_aggregation": {
                "token_attention": "mean(block, head, latent_query)",
                "block_token_attention": "mean(head, latent_query)",
                "head_token_attention": "mean(block, latent_query)",
            },
            "full_attention_shape": [
                int(trace["block_token_attention"].shape[0]),
                int(trace["head_token_attention"].shape[0]),
                int(self.policy._model.mem_encoder.num_latents),
                int(trace["token_attention"].shape[0]),
            ],
        }
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._attention_call_index += 1

    def _infer(self, observation):
        actions = super()._infer(observation)
        self._capture_memory_attention(observation)
        return actions


def get_model(usr_args):
    model = base.get_model(usr_args)
    model.__class__ = AttentionRoboMMETCPServerModel
    model.initialize_attention_capture()
    return model


eval = base.eval
reset_model = base.reset_model
