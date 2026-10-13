import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration import memory_pi0_gemma as _gemma
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_observation import preprocess_observation
from mme_vla_suite.models.integration.utils import Variant as MemoryExpertVariant
from mme_vla_suite.models.integration.utils import get_config as get_memory_expert_config
import openpi.models.gemma as _openpi_gemma
from openpi.models.model import Actions
from openpi.models.model import BaseModel
from openpi.models.pi0 import posemb_sincos
from openpi.models.pi0_config import Pi0Config
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("history-pi0")


def apply_current_obs_dropout(
    images: dict[str, at.Array],
    image_masks: dict[str, at.Array],
    *,
    rng: at.KeyArrayLike,
    prob: float,
    fill_value: float,
    target_image_names: tuple[str, ...] = ("base_0_rgb",),
) -> tuple[dict[str, at.Array], dict[str, at.Array]]:
    """Blank selected current observation images per sample while preserving masks."""
    prob = float(prob)
    if prob <= 0.0:
        return images, image_masks
    if prob > 1.0:
        raise ValueError(f"current_obs_dropout_prob must be in [0, 1], got {prob}")

    names = tuple(name for name in target_image_names if name in images)
    if not names:
        return images, image_masks

    keys = jax.random.split(rng, len(names))
    dropped = dict(images)
    for name, key in zip(names, keys, strict=True):
        image = images[name]
        valid = image_masks[name]
        drop = jax.random.bernoulli(key, prob, valid.shape) & valid
        drop = drop.reshape(drop.shape + (1,) * (image.ndim - drop.ndim))
        fill = jnp.full_like(image, fill_value)
        dropped[name] = jnp.where(drop, fill, image)
    return dropped, image_masks


def _parse_image_key_list(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    return tuple(name.strip() for name in str(value).split(",") if name.strip())


def make_attn_mask(input_mask, mask_ar, mask_na=None):
    """Adapted from pi0.py

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    mask_na: bool[B, N]
       [[0 0 0 1 1 1 0 0 0 ...]]: The 1s tokens can not attend to the first three 0s.
       this is used for vision tokens not attend to memory tokens, but action tokens can

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    if mask_na is not None:
        mask_na = jnp.broadcast_to(mask_na, input_mask.shape)
        mask_not_attend = jnp.logical_and(
            jnp.logical_or(mask_na[:, None, :], mask_na[:, :, None]),
            einops.repeat(jnp.cumsum(mask_na, axis=1) <= 0, "b k -> b s k", s=mask_na.shape[1]),
        )
        # print(mask_not_attend)
        return jnp.where(
            mask_not_attend,
            False,  # noqa: FBT003
            jnp.logical_and(attn_mask, valid_mask),
        )
    return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class MemoryPi0Config(Pi0Config):
    """Native pi0 action expert augmented with the existing memory interface.

    This is intentionally a separate model from HistoryPi0. The pi0 branch must
    use the continuous state token and action-time MLP from pi0_base; allowing a
    pi0.5 config here would make checkpoint mismatches too easy to miss.
    """

    pi05: bool = False
    discrete_state_input: bool = False
    memory_expert_variant: MemoryExpertVariant = "gemma_150m"

    use_history: bool = False  # Use history or not
    history_config: str | None = None  # history config
    max_token_len: int = 64

    # E2E+aux: auxiliary Stage-1 reconstruction losses (align/ap/reward) on the bottleneck tokens.
    # Default 0.0 => OFF => no decoder built, no aux loss, action-only behavior (all existing configs).
    # aux_weight is the schedule START; decays to aux_weight_end over training (bottleneck-only grads).
    aux_weight: float = 0.0
    aux_weight_end: float = 0.0
    aux_reward_w: float = 0.0  # per-term reward weight within the aux group (align/ap fixed 1.0)
    # aux lambda schedule: if False (default) lambda is a FIXED constant (single value when
    # aux_weight==aux_weight_end, else their midpoint -- the original behavior, wandb aux/lambda flat).
    # If True, lambda is LINEARLY ANNEALED aux_weight -> aux_weight_end over training using the
    # train_fraction (step/num_train_steps) threaded from train_step into compute_loss.
    aux_anneal: bool = False
    # Training-only regularizer: per-sample/per-view blanking of current observation images before
    # pi0 image tokenization. Memory streams and eval/sampling are unchanged.
    current_obs_dropout_prob: float = 0.0
    current_obs_dropout_fill_value: float = 0.0
    current_obs_dropout_image_keys: str = "base_0_rgb"
    # Optional training-only sample reweighting. When weight > 1, the dataloader
    # tags samples at or after late_episode_loss_fraction of their episode and
    # train_step multiplies their diffusion loss by this scalar.
    late_episode_loss_weight: float = 1.0
    late_episode_loss_fraction: float = 0.5
    key_event_manifest_path: str | None = None
    key_event_loss_weight: float = 1.0
    key_event_window_radius: int = 5

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.pi05:
            raise ValueError("MemoryPi0Config only supports native pi0 (pi05=False).")
        if self.discrete_state_input:
            raise ValueError("pi0 uses the continuous state suffix token, not discrete prompt state.")

    def create(self, rng: at.KeyArrayLike) -> "MemoryPi0":
        # Load the history config if it's specified
        if self.history_config is not None:
            loaded_config = get_history_config(self.history_config)
            # Create a new config with the loaded history config
            config_with_loaded_history = dataclasses.replace(self, history_config=loaded_config)

            max_token_len = self.max_token_len
            if loaded_config.representation_type == "symbolic":
                if loaded_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"]:
                    max_token_len *= 2
                else:
                    raise ValueError(f"Not supported symbolic memory type: {loaded_config.symbolic_memory.type}")
                config_with_loaded_history = dataclasses.replace(
                    config_with_loaded_history, max_token_len=max_token_len
                )
                print("symbolic_memory_type: ", loaded_config.symbolic_memory.type)

            print("max_token_len: ", config_with_loaded_history.max_token_len)

            return MemoryPi0(config_with_loaded_history, rngs=nnx.Rngs(rng))
        return MemoryPi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Actions]:
        base_obs_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            if not self.use_history:
                observation_spec = base_obs_spec  # basic pi0
            elif self.history_config.representation_type == "symbolic":
                observation_spec = HistAugObservation.from_base_obs(
                    base_obs_spec,
                    symbolic_tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                    symbolic_tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                )
            elif self.history_config.representation_type == "perceptual":
                observation_spec = HistAugObservation.from_base_obs(
                    base_obs_spec,
                    static_image_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.budget,
                            self.history_config.memory_feature.img.input_dim,
                        ],
                        jnp.float32,
                    ),
                    static_mask=jax.ShapeDtypeStruct(
                        [batch_size, self.history_config.budget],
                        jnp.bool_,
                    ),
                    static_pos_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.budget,
                            self.history_config.memory_feature.pos.input_dim,
                        ],
                        jnp.float32,
                    ),
                    static_state_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.budget,
                            self.history_config.memory_feature.state.input_dim,
                        ],
                        jnp.float32,
                    ),
                    # time add-on: per-frame elapsed fraction (gated by memory_use_time)
                    static_time=(
                        jax.ShapeDtypeStruct([batch_size, self.history_config.budget, 1], jnp.float32)
                        if getattr(self.history_config, "memory_use_time", False)
                        else None
                    ),
                )
            elif self.history_config.representation_type == "recurrent":
                observation_spec = HistAugObservation.from_base_obs(
                    base_obs_spec,
                    recur_image_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.recurrent_memory.max_recur_steps,
                            self.history_config.num_views,
                            self.history_config.token_per_image,
                            self.history_config.memory_feature.img.input_dim,
                        ],
                        jnp.float32,
                    ),
                    recur_mask=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.recurrent_memory.max_recur_steps,
                        ],
                        jnp.bool_,
                    ),
                    recur_pos_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.recurrent_memory.max_recur_steps,
                            self.history_config.num_views,
                            self.history_config.token_per_image,
                            self.history_config.memory_feature.pos.input_dim,
                        ],
                        jnp.float32,
                    ),
                    recur_state_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.recurrent_memory.max_recur_steps,
                            self.history_config.memory_feature.state.input_dim,
                        ],
                        jnp.float32,
                    ),
                )

            elif self.history_config.representation_type == "cosmo_bottleneck":
                cbm = self.history_config.cosmo_bottleneck_memory
                num_latents = cbm.num_latents
                recent_len = self.history_config.streaming_obs_horizon * self.history_config.num_views
                observation_spec = HistAugObservation.from_base_obs(
                    base_obs_spec,
                    cosmo_latent_emb=jax.ShapeDtypeStruct(
                        [batch_size, num_latents, cbm.cosmos_latent_dim],
                        jnp.float32,
                    ),
                    cosmo_recent_emb=jax.ShapeDtypeStruct(
                        [batch_size, recent_len, cbm.siglip_dim],
                        jnp.float32,
                    ),
                    cosmo_mem_mask=jax.ShapeDtypeStruct(
                        [batch_size, num_latents],
                        jnp.bool_,
                    ),
                    cosmo_recent_mask=jax.ShapeDtypeStruct(
                        [batch_size, recent_len],
                        jnp.bool_,
                    ),
                    cosmo_action_mask=jax.ShapeDtypeStruct(
                        [batch_size, recent_len],
                        jnp.bool_,
                    ),
                    cosmo_proprio_mask=jax.ShapeDtypeStruct(
                        [batch_size, recent_len],
                        jnp.bool_,
                    ),
                    cosmo_action_emb=jax.ShapeDtypeStruct(
                        [batch_size, recent_len, cbm.action_in_dim],
                        jnp.float32,
                    ),
                    cosmo_proprio_emb=jax.ShapeDtypeStruct(
                        [batch_size, recent_len, cbm.proprio_in_dim],
                        jnp.float32,
                    ),
                    cosmo_lang_emb=jax.ShapeDtypeStruct(
                        [batch_size, cbm.lang_emb_dim],
                        jnp.float32,
                    ),
                )

            else:
                raise ValueError(f"Not supported representation type: {self.history_config.representation_type}")

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params and mem params
            filters.extend(
                [
                    nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                    nnx.Not(nnx_utils.PathRegex(".*mem.*")),
                ]
            )
            return nnx.Any(nnx.All(*filters), nnx_utils.PathRegex(".*img.*"))
        return nnx_utils.PathRegex(".*img.*")

    def get_stage2_frozen_bottleneck_freeze_filter(self) -> nnx.filterlib.Filter:
        """Stage-2 (memory injection) freeze recipe: freeze the perception/memory ENCODERS,
        train the memory->action injection + action expert + VLM transformer.

        Frozen (Any of):
          .*img.*        -- SigLIP visual encoder (ROBOMME recipe; keeps recent_siglip in-dist)
          .*embedder.*   -- gemma_2b token-embedding table. cosmo_lang_emb = embedder.encode(text),
                            so freezing it keeps the FROZEN bottleneck's language input identical
                            to Stage-1 (no drift). Shared with the VLM text prefix.
          .*mem_encoder.* -- the CosmoBottleneckMemory (Perceiver). Frozen so the 256 memory tokens
                            keep the meaning learned in Stage-1; only the DOWNSTREAM mapping adapts.

        Trainable (everything else): the modulation seam under llm/layers (mem_attn / mem_rms_norm_ffn),
        the action expert (llm *_1), and the VLM transformer. Verified split: 87 frozen / 33 train,
        no mem_encoder/embedder leakage into the trainable set.
        """
        return nnx.Any(
            nnx_utils.PathRegex(".*img.*"),
            nnx_utils.PathRegex(".*embedder.*"),
            nnx_utils.PathRegex(".*mem_encoder.*"),
        )

    def get_image_and_language_encoder_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the Pi0 visual tower and PaliGemma language branch.

        The native Pi0 action expert shares the scanned ``llm`` module with the
        PaliGemma branch, but its leaves carry the ``_1`` suffix. The memory
        modulation seam also lives under ``llm`` and must remain trainable.
        """
        action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
        memory_seam = nnx_utils.PathRegex(
            ".*(mem_attn|mem_rms_norm|q_einsum_mem|kv_einsum_mem|out_einsum_mem).*"
        )
        language_encoder = nnx.All(
            nnx_utils.PathRegex(".*llm.*"),
            nnx.Not(action_expert),
            nnx.Not(memory_seam),
        )
        return nnx.Any(
            nnx_utils.PathRegex(".*img.*"),
            language_encoder,
        )


class MemoryPi0(BaseModel):
    """OpenPI pi0 with 1024-wide memory modulation in its action expert."""

    def __init__(self, config: MemoryPi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        if config.pi05:
            raise ValueError("MemoryPi0 cannot construct a pi0.5 action expert.")
        paligemma_config = _openpi_gemma.get_config(config.paligemma_variant)
        action_expert_config = _openpi_gemma.get_config(config.action_expert_variant)
        self.action_expert_width = action_expert_config.width

        self.config = config
        self.use_history = config.use_history
        self.current_obs_dropout_prob = float(getattr(config, "current_obs_dropout_prob", 0.0) or 0.0)
        self.current_obs_dropout_fill_value = float(getattr(config, "current_obs_dropout_fill_value", 0.0) or 0.0)
        self.current_obs_dropout_image_keys = _parse_image_key_list(
            getattr(config, "current_obs_dropout_image_keys", "base_0_rgb")
        )
        if not 0.0 <= self.current_obs_dropout_prob <= 1.0:
            raise ValueError(f"current_obs_dropout_prob must be in [0, 1], got {self.current_obs_dropout_prob}")

        if self.use_history:
            self.history_config = config.history_config
            self.integration_type = config.history_config.integration_type
            self.representation_type = config.history_config.representation_type
            assert self.integration_type in ["context", "modulation", "expert"]
            assert self.representation_type in ["perceptual", "recurrent", "symbolic", "cosmo_bottleneck"]
            if self.integration_type == "modulation":
                memory_token_dim = int(config.history_config.memory_token_dim)
                if memory_token_dim != self.action_expert_width:
                    raise ValueError(
                        "Memory tokens must match the native pi0 action expert width: "
                        f"{memory_token_dim} != {self.action_expert_width}."
                    )

            if self.representation_type == "perceptual":
                from mme_vla_suite.models.representation.percep_mem import PerceptualMemory  # noqa: PLC0415

                self.mem_encoder = PerceptualMemory(
                    config=self.history_config,
                    rngs=rngs,
                    dtype=config.dtype,
                )
            elif self.representation_type == "recurrent":
                from mme_vla_suite.models.representation.recur_mem import RecurrentMemory  # noqa: PLC0415

                self.mem_encoder = RecurrentMemory(
                    config=self.history_config,
                    rngs=rngs,
                    dtype=config.dtype,
                )
            elif self.representation_type == "cosmo_bottleneck":
                from mme_vla_suite.models.representation.memory_factory import (  # noqa: PLC0415
                    create_cosmo_memory_encoder,
                )

                self.mem_encoder = create_cosmo_memory_encoder(
                    config=self.history_config,
                    rngs=rngs,
                    dtype=config.dtype,
                )
                # E2E+aux ONLY: reconstruction decoder (align/action/proprio/reward heads) that
                # regularizes the bottleneck. Built iff aux_weight>0; else absent -> zero behavior
                # change for action-only e2e and every other config.
                self._aux_weight = float(getattr(config, "aux_weight", 0.0) or 0.0)
                self._aux_weight_end = float(getattr(config, "aux_weight_end", 0.0) or 0.0)
                self._aux_reward_w = float(getattr(config, "aux_reward_w", 0.0) or 0.0)
                self._aux_anneal = bool(getattr(config, "aux_anneal", False))
                if self._aux_weight > 0.0:
                    from mme_vla_suite.models.representation.cosmo_decoder import CosmoMemoryDecoder  # noqa: PLC0415

                    self.aux_decoder = CosmoMemoryDecoder(
                        config=self.history_config,
                        rngs=rngs,
                        dtype=config.dtype,
                    )
            elif self.representation_type == "symbolic":
                self.integration_type = None  # if symbolic, we only use it as languge input
            else:
                raise ValueError(f"Not supported representation type: {self.representation_type}")

            print(
                f"====== Using History, Representation Type: {self.representation_type} , Integration Type: {self.integration_type} ======"
            )
            if self.representation_type == "perceptual":
                print(f"Perceptual Memory using {self.history_config.perceptual_memory.type} type\n")
            elif self.representation_type == "recurrent":
                print(f"Recurrent Memory using {self.history_config.recurrent_memory.type} type\n")
            else:
                print("\n")

            if self.integration_type == "expert":
                memory_expert_config = get_memory_expert_config(config.memory_expert_variant)
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[
                            memory_expert_config,
                            paligemma_config,
                            action_expert_config,
                        ],
                        embed_dtype=config.dtype,
                        integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(rngs=rngs, method="init", mem_mods=[False, False, False])
            else:
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[paligemma_config, action_expert_config],
                        embed_dtype=config.dtype,
                        integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(
                    rngs=rngs,
                    method="init",
                    mem_mods=[False, True] if self.integration_type == "modulation" else [False, False],
                )

        else:
            # safe setting
            self.history_config = self.integration_type = self.representation_type = None

            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    integration_type=self.integration_type,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", mem_mods=[False, False])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        # Native pi0 suffix: continuous state token plus per-action time mixing.
        # pi0.5 instead uses time_mlp_* and adaRMSNorm; neither exists in this model.
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_memory(self, obs: HistAugObservation):
        if self.representation_type == "perceptual":
            tokens, _, stats = self.mem_encoder(
                obs.static_image_emb,
                obs.static_pos_emb,
                obs.static_state_emb,
                static_time=obs.static_time,
            )
            input_mask = obs.static_mask
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        elif self.representation_type == "recurrent":
            (tokens, input_mask), _, stats = self.mem_encoder(
                obs.recur_image_emb, obs.recur_mask, obs.recur_pos_emb, obs.recur_state_emb
            )
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        elif self.representation_type == "cosmo_bottleneck":
            if obs.cosmo_mem_tokens is not None:
                # PRECOMPUTE PATH: frozen bottleneck tokens loaded from cache -> skip mem_encoder.
                tokens = obs.cosmo_mem_tokens
                if tokens.shape[-1] != self.action_expert_width:
                    raise ValueError(
                        "Precomputed memory token width does not match the pi0 action expert: "
                        f"{tokens.shape[-1]} != {self.action_expert_width}."
                    )
                input_mask = jnp.ones((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool_)
                stats = None
            else:
                tokens, input_mask, stats = self.mem_encoder(
                    obs.cosmo_latent_emb,
                    obs.cosmo_recent_emb,
                    obs.cosmo_action_emb,
                    obs.cosmo_proprio_emb,
                    obs.cosmo_mem_mask,
                    lang_emb=obs.cosmo_lang_emb,
                    recent_mask=obs.cosmo_recent_mask,
                    action_mask=obs.cosmo_action_mask,
                    proprio_mask=obs.cosmo_proprio_mask,
                )
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        else:
            tokens = None
            input_mask = None
            ar_mask = None
            na_mask = None
            stats = None
        return tokens, input_mask, ar_mask, na_mask, stats

    def init_delta_memory_state(self, batch_size=1):
        if not hasattr(self.mem_encoder, "init_incremental_state"):
            raise TypeError("configured memory encoder does not support incremental state")
        return self.mem_encoder.init_incremental_state(batch_size)

    def update_delta_memory_state(
        self,
        state,
        cosmo_chunk,
        action_chunk,
        proprio_chunk,
    ):
        if not hasattr(self.mem_encoder, "update_incremental"):
            raise TypeError("configured memory encoder does not support incremental updates")
        return self.mem_encoder.update_incremental(
            state,
            cosmo_chunk,
            action_chunk,
            proprio_chunk,
        )

    def read_delta_memory_state(
        self,
        state,
        recent_emb,
        action_recent,
        proprio_recent,
        lang_emb,
        recent_mask,
        action_mask,
        proprio_mask,
    ):
        if not hasattr(self.mem_encoder, "read_incremental"):
            raise TypeError("configured memory encoder does not support incremental reads")
        tokens, _, _ = self.mem_encoder.read_incremental(
            state,
            recent_emb,
            action_recent,
            proprio_recent,
            lang_emb=lang_emb,
            recent_mask=recent_mask,
            action_mask=action_mask,
            proprio_mask=proprio_mask,
        )
        return tokens

    @at.typecheck
    def embed_prefix(
        self, obs: HistAugObservation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        Any | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        if self.integration_type == "context":
            (
                mem_tokens,
                mem_input_mask,
                mem_ar_mask,
                mem_na_mask,
                stats,
            ) = self.embed_memory(obs)
            if mem_tokens is not None:
                tokens.append(mem_tokens)
                input_mask.append(mem_input_mask)
                ar_mask += mem_ar_mask
                na_mask += mem_na_mask
        else:
            stats = None

        # embed images
        for i, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            if i == 0:
                ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))
            else:
                ar_mask += [False] * image_tokens.shape[1]
            na_mask += [True] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if self.use_history and self.representation_type == "symbolic":
            tokenized_inputs = self.PaliGemma.llm(obs.symbolic_tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.symbolic_tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        elif obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)

        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_suffix(
        self,
        obs: HistAugObservation,
        noisy_actions: Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        # Native pi0 adds proprioception as one continuous suffix token.
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # Image/language inputs do not attend to state or actions.
        ar_mask += [True]
        na_mask += [False]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # Native pi0 mixes the flow timestep into every action token before the
        # transformer. It does not use pi0.5's adaptive RMSNorm conditioning.
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        action_expert_tokens = action_time_tokens
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        na_mask += [False] * self.action_horizon
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)
        return tokens, input_mask, ar_mask, na_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        actions: Actions,
        *,
        train: bool = False,
        train_fraction: float | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)
        if train and self.current_obs_dropout_prob > 0.0:
            images, image_masks = apply_current_obs_dropout(
                observation.images,
                observation.image_masks,
                rng=jax.random.fold_in(rng, 1701),
                prob=self.current_obs_dropout_prob,
                fill_value=self.current_obs_dropout_fill_value,
                target_image_names=self.current_obs_dropout_image_keys,
            )
            observation = observation.replace(images=images, image_masks=image_masks)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, stats = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask = self.embed_suffix(observation, x_t, time)

        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, stats = self.embed_memory(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            input_mask = jnp.concatenate([mem_input_mask, prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([mem_ar_mask, prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([mem_na_mask, prefix_na_mask, suffix_na_mask], axis=0)
        else:
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)

        if self.use_history and self.representation_type != "symbolic":
            attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
        else:
            attn_mask = make_attn_mask(input_mask, ar_mask)

        positions = jnp.cumsum(input_mask, axis=1) - 1

        if self.integration_type == "expert":
            (_mem_out, _prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [mem_tokens, prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
            )
        elif self.integration_type == "modulation":
            mem_seq, mem_mask, _, _, stats = self.embed_memory(observation)
            (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                mem_seq=[None, mem_seq],
                mem_mask=[None, mem_mask],
            )
        else:
            (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
            )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (*b, ah) PRIMARY

        # --- E2E+aux: auxiliary Stage-1 reconstruction loss on the bottleneck tokens ---
        # OPT-IN: only when aux_weight>0 AND decoder built AND targets present. Otherwise the
        # original action-only return runs verbatim (no behavior change for other configs).
        if (
            getattr(self, "_aux_weight", 0.0) > 0.0
            and hasattr(self, "aux_decoder")
            and getattr(observation, "aux_align_target", None) is not None
            and self.representation_type == "cosmo_bottleneck"
        ):
            # bottleneck tokens for the aux branch (same as mem_seq fed to the VLA). NOTE: aux grads
            # flow into mem_encoder + aux_decoder only (the VLA is a separate consumer) => bottleneck-only.
            aux_mem = mem_seq if self.integration_type == "modulation" else mem_tokens
            aux = self._aux_loss(aux_mem, observation)
            lam = self._aux_lambda(train_fraction)  # constant (default) or annealed if aux_anneal
            if stats is None:
                stats = {}
            stats = {
                **stats,
                "loss/action": jnp.mean(action_loss),
                "loss/aux_total": aux["total"],
                "loss/aux_align": aux["align"],
                "loss/aux_action": aux["action"],
                "loss/aux_proprio": aux["proprio"],
                "loss/aux_reward": aux.get("reward", 0.0),
                "aux/lambda": lam,
            }
            # add scalar aux (broadcast) to the per-element action loss so grads combine
            return action_loss + lam * aux["total"], stats

        return action_loss, stats

    def _aux_lambda(self, train_fraction=None):
        # Two modes (aux_anneal flag on the config):
        #  * aux_anneal=False (DEFAULT, unchanged): FIXED lambda. Single value when
        #    aux_weight==aux_weight_end, else their midpoint. (wandb aux/lambda flat, e.g. 0.15.)
        #  * aux_anneal=True: LINEAR anneal aux_weight -> aux_weight_end over training, driven by
        #    train_fraction = step/num_train_steps (threaded from train_step, jit-safe traced scalar).
        #    lambda(f) = w0 + (w_end - w0) * clip(f, 0, 1). Falls back to the fixed value if the
        #    caller did not supply train_fraction (e.g. eval/inference), so lambda stays well-defined.
        w0, w1 = self._aux_weight, self._aux_weight_end
        if getattr(self, "_aux_anneal", False) and train_fraction is not None and w1 != w0:
            f = jnp.clip(train_fraction, 0.0, 1.0)
            return w0 + (w1 - w0) * f
        if w1 > 0.0 and w1 != w0:
            return 0.5 * (w0 + w1)
        return w0

    def _aux_loss(self, mem, obs):
        # mirrors pretrain_loss.composite_alignment_loss (same LayerNorm-align recipe + masks)
        def _ln(x, eps=1e-6):
            mu = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
            return (x - mu) * jax.lax.rsqrt(var + eps)

        out = self.aux_decoder(mem, obs.aux_query_steps)
        qm = getattr(obs, "aux_query_mask", None)

        def masked_mean(err):
            feat = tuple(range(2, err.ndim))
            per_q = jnp.mean(err, axis=feat) if feat else err
            if qm is None:
                return jnp.mean(per_q)
            m = qm.astype(per_q.dtype)
            return jnp.sum(per_q * m) / jnp.clip(jnp.sum(m), 1.0, None)

        # upcast align target to fp32 before LayerNorm (it may be stored/transferred as fp16 to
        # halve dataloader IO + host->device bandwidth; the normalization wants full precision).
        _al_t = obs.aux_align_target.astype(jnp.float32)
        l_align = masked_mean((_ln(out["align"]) - jax.lax.stop_gradient(_ln(_al_t))) ** 2)
        l_action = masked_mean((out["action"] - obs.aux_action_target) ** 2)
        l_prop = masked_mean((out["proprio"] - obs.aux_proprio_target) ** 2)
        parts = {"align": l_align, "action": l_action, "proprio": l_prop}
        total = l_align + l_action + l_prop  # align/ap fixed weight 1.0 (within-group)
        if self._aux_reward_w > 0.0 and getattr(obs, "aux_reward_target", None) is not None:
            l_reward = jnp.mean((out["reward"] - obs.aux_reward_target) ** 2)
            parts["reward"] = l_reward
            total = total + self._aux_reward_w * l_reward
        parts["total"] = total
        return parts

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:

        observation = preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, _ = self.embed_memory(observation)
            vlm_tokens, vlm_mask, vlm_ar_mask, vlm_na_mask, _ = self.embed_prefix(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            prefix_mask = jnp.concatenate([mem_input_mask, vlm_mask], axis=1)
            prefix_ar_mask = jnp.concatenate([mem_ar_mask, vlm_ar_mask], axis=0)
            prefix_na_mask = jnp.concatenate([mem_na_mask, vlm_na_mask], axis=0)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [mem_tokens, vlm_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )

        elif self.integration_type == "modulation":
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            mem_seq, mem_mask, _, _, _ = self.embed_memory(observation)

        else:
            prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = self.embed_prefix(observation)
            if self.integration_type == "context":
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            else:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _ = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if self.integration_type == "expert":
                (mem_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                )
                assert mem_out is None
            elif self.integration_type == "modulation":
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    mem_seq=[None, mem_seq],
                    mem_mask=[None, mem_mask],
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def vision_encode(self, images: at.Float[at.Array, "k v 224 224 3"]):
        shape = images.shape
        assert shape[-3:] == (
            224,
            224,
            3,
        ), f"x.shape: {shape}, should be (B, 224, 224, 3)"

        x = einops.rearrange(images, "k v h w c -> (k v) h w c")
        out, _ = self.PaliGemma.img(x, train=False)
        return einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])
