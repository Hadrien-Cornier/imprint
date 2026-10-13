import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro


import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import numpy as np

from openpi.transforms import DataTransformFn, DataDict
import sentencepiece
import os

from mme_vla_suite.models.integration import history_pi0
from mme_vla_suite.models.integration import memory_pi0
from mme_vla_suite.policies.robomme_policy import RoboMMEInputs, RoboMMEJointOutputs, RoboMMEOutputs
from mme_vla_suite.policies.robodojo_policy import RoboDojoInputs, RoboDojoOutputs
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.pi0_weight_loader import (
    NativePi0Stage1BottleneckWeightLoader,
)


ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""



class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        subgoal: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            if subgoal is None:
                # This is original Pi05 format, where the state is part of the discrete language input.
                state = np.clip(state, -1, 1)
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}; State: {state_str};\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            else:
                # this is used only for real robot symbolic variant. We do not use proprioceptive states in our experiment
                subgoal = subgoal.strip().replace("_", " ").replace("\n", " ")
                full_prompt = f"Task: {cleaned_text}\nCurrent Subgoal: {subgoal}.\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)

        elif subgoal is not None:
            # This is the subgoal format, where the subgoal is part of the language input. for both simple and grounded.
            subgoal = subgoal.strip().replace("_", " ").replace("\n", " ")
            full_prompt = f"Task: {cleaned_text};\nCurrent Subgoal: {subgoal};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)

        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                # assert False, f"Token length ({len(tokens)}) exceeds max length ({self._max_len})!!!!!!!!!!!"
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)



@dataclasses.dataclass(frozen=True)
class TokenizePromptWithSymbolicMemory(DataTransformFn):
    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = True
    symbolic_memory_type: str | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()


        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        if self.symbolic_memory_type is None:
            data.pop("simple_subgoal")
            data.pop("grounded_subgoal")
            return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}

        if self.symbolic_memory_type == "simple_subgoal":
            simple_subgoal = data['simple_subgoal']
            symbolic_tokenized_prompt, symbolic_tokenized_prompt_mask = self.tokenizer.tokenize(prompt=prompt, subgoal=simple_subgoal, state=state)
        elif self.symbolic_memory_type == "grounded_subgoal":
            grounded_subgoal = data['grounded_subgoal']
            symbolic_tokenized_prompt, symbolic_tokenized_prompt_mask = self.tokenizer.tokenize(prompt=prompt, subgoal=grounded_subgoal, state=state)
        else:
            raise ValueError(f"Invalid symbolic memory type: {self.symbolic_memory_type}")

        data.pop("simple_subgoal")
        data.pop("grounded_subgoal")

        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks,
                "symbolic_tokenized_prompt": symbolic_tokenized_prompt,
                "symbolic_tokenized_prompt_mask": symbolic_tokenized_prompt_mask}


@dataclasses.dataclass(frozen=True)
class DropUnusedSymbolicMemory(DataTransformFn):
    """Remove string fields that cannot enter a numeric JAX training batch."""

    def __call__(self, data: DataDict) -> DataDict:
        data.pop("simple_subgoal", None)
        data.pop("grounded_subgoal", None)
        return data


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        DropUnusedSymbolicMemory(),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # This is MME-VLA set up
                symbolic_memory_type = None
                max_token_len = model_config.max_token_len

                if model_config.use_history and model_config.history_config is not None:
                    loaded_config = get_history_config(model_config.history_config)

                    if loaded_config.representation_type == "symbolic":
                        symbolic_memory_type = loaded_config.symbolic_memory.type
                        max_token_len *= 2 # it's enough for subgoals, no need to set into 512.

                print(f"max_token_len: {max_token_len}")

                return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            TokenizePromptWithSymbolicMemory(
                                PaligemmaTokenizer(max_token_len),
                                discrete_state_input=model_config.discrete_state_input,
                                symbolic_memory_type=symbolic_memory_type,
                            ),
                            _transforms.PadStatesAndActions(model_config.action_dim),
                        ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,  # pi05 base+toy trained with z-score (norm_stats have no quantiles); match that
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)



@dataclasses.dataclass(frozen=True)
class RoboMMEDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/right_image": "right_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # ---- New added keys ----
                        # perceptual memory
                        "static_image_emb": "static_image_emb", # (b, l, d1)
                        "static_pos_emb": "static_pos_emb", # (b, l, d2)
                        "static_state_emb": "static_state_emb", # (b, l, d3)
                        "static_mask": "static_mask", # (b, l)
                        "static_time": "static_time", # (b, l, 1) time add-on
                        # recurrent memory
                        "recur_image_emb": "recur_image_emb", # (b, t, v, p, d1)
                        "recur_pos_emb": "recur_pos_emb", # (b, t, v, p, d2)
                        "recur_state_emb": "recur_state_emb", # (b, t, d3)
                        "recur_mask": "recur_mask",
                        # symbolic memory
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                    }
                )
            ]
        )

        # PRECOMPUTE PATH: only forward cosmo_mem_tokens through the repack when a
        # frozen-bottleneck token cache is configured. RepackTransform requires EVERY listed
        # key to be present in each sample; the non-precompute (raw-stream) path never emits
        # cosmo_mem_tokens, so listing it unconditionally would KeyError all real Stage-2 runs.
        try:
            _hc = get_history_config(model_config.history_config) if model_config.history_config is not None else None
            _cbm = getattr(_hc, "cosmo_bottleneck_memory", None) if _hc is not None else None
            _precomp = bool(str(getattr(_cbm, "precompute_memory_dir", "") or "")) if _cbm is not None else False
        except Exception:
            _precomp = False
            _cbm = None
        # Only forward the cosmo_* streams through the repack when a cosmo_bottleneck
        # history config is active; the perceptual/recurrent/symbolic paths never emit
        # them, so listing them unconditionally KeyErrors those runs (RepackTransform
        # requires every listed key present in each sample).
        if _cbm is not None:
            for _k in ("cosmo_latent_emb", "cosmo_mem_mask", "cosmo_recent_emb",
                       "cosmo_recent_mask", "cosmo_action_emb", "cosmo_action_mask",
                       "cosmo_proprio_emb", "cosmo_proprio_mask", "cosmo_lang_emb"):
                repack_transform.inputs[0].structure[_k] = _k
        if _precomp:
            repack_transform.inputs[0].structure["cosmo_mem_tokens"] = "cosmo_mem_tokens"
        # E2E+aux: forward the aux_* reconstruction targets through the repack ONLY when the run
        # requests aux (model_config.aux_weight>0). Non-aux runs never emit these keys, so listing
        # them would KeyError (RepackTransform requires every listed key present in each sample).
        if float(getattr(model_config, "aux_weight", 0.0) or 0.0) > 0.0:
            for _k in ("aux_query_steps", "aux_query_mask", "aux_align_target",
                       "aux_action_target", "aux_proprio_target", "aux_reward_target"):
                repack_transform.inputs[0].structure[_k] = _k
        if (
            float(getattr(model_config, "late_episode_loss_weight", 1.0) or 1.0) > 1.0
            or bool(getattr(model_config, "key_event_manifest_path", None))
        ):
            repack_transform.inputs[0].structure["sample_loss_weight"] = "sample_loss_weight"

        data_transforms = _transforms.Group(
            inputs=[RoboMMEInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEOutputs()],
        )

        # NOTE: robocasa actions are ALREADY delta/relative (rel_pos etc.), so we do
        # NOT apply a DeltaActions transform — matching the working baseline
        # LeRobotRobocasaDataConfig (which leaves it commented out). The previous
        # DeltaActions(make_bool_mask(7,-1)) double-delta'd already-delta actions
        # against the absolute base_to_eef state, corrupting the action targets.

        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RoboMMEJointDataConfig(RoboMMEDataConfig):
    """Two-view RoboMME imitation data with joint-delta policy targets."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create(assets_dirs, model_config)
        structure = dict(config.repack_transforms.inputs[0].structure)
        structure.pop("observation/right_image")
        repack_transforms = _transforms.Group(
            inputs=[_transforms.RepackTransform(structure)]
        )
        data_transforms = _transforms.Group(
            inputs=[RoboMMEInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEJointOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        return dataclasses.replace(
            config,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
        )




@dataclasses.dataclass(frozen=True)
class LeRobotMMEVLARealRobotDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/left_shoulder_image": "left_shoulder_image",
                        "observation/right_shoulder_image": "right_shoulder_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # ---- New added keys ----
                        # perceptual memory
                        "static_image_emb": "static_image_emb", # (b, l, d1)
                        "static_pos_emb": "static_pos_emb", # (b, l, d2)
                        "static_state_emb": "static_state_emb", # (b, l, d3)
                        "static_mask": "static_mask", # (b, l)
                        "static_time": "static_time", # (b, l, 1) time add-on
                        # recurrent memory
                        "recur_image_emb": "recur_image_emb", # (b, t, v, p, d1)
                        "recur_pos_emb": "recur_pos_emb", # (b, t, v, p, d2)
                        "recur_state_emb": "recur_state_emb", # (b, t, d3)
                        "recur_mask": "recur_mask",
                        # symbolic memory
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[RoboMMERealRobotInputs(model_type=model_config.model_type)],
            outputs=[RoboMMERealRobotOutputs()],
        )

        # NOTE: robocasa actions are ALREADY delta/relative (rel_pos etc.), so we do
        # NOT apply a DeltaActions transform — matching the working baseline
        # LeRobotRobocasaDataConfig (which leaves it commented out). The previous
        # DeltaActions(make_bool_mask(7,-1)) double-delta'd already-delta actions
        # against the absolute base_to_eef state, corrupting the action targets.

        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

import openpi.shared.array_typing as at
from openpi.training.weight_loaders import WeightLoader, download, _merge_params
@dataclasses.dataclass(frozen=True)
class MMEVLAWeightLoader(WeightLoader):
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights. And our new weights
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Stage1BottleneckWeightLoader(WeightLoader):
    """Stage-2 weight transfer hook.

    (1) Load a base pi0.5 checkpoint (base OR washreturn-finetuned) into the VLM +
        action expert exactly like MMEVLAWeightLoader.
    (2) OVERLAY the Stage-1-trained bottleneck params into `mem_encoder/*`, so Stage-2
        finetunes an ALREADY-ALIGNED memory module (not a random init). The Stage-1
        decoder is loaded into ``aux_decoder/*`` only when the Stage-2 model builds
        the auxiliary reconstruction branch. The FiLM seam (mem_attn / mem_rms_norm)
        is NOT in the Stage-1 checkpoint -> keeps its fresh init and trains in Stage-2.

    stage1_ckpt is the .npz saved by scripts/pretrain_bottleneck.py, whose keys look like
    `bottleneck/<...>` and `decoder/<...>`. We remap `bottleneck/` -> `mem_encoder/` and
    remap `decoder/` -> `aux_decoder/` when that branch exists. Every `mem_encoder/*`
    reference param, and every enabled `aux_decoder/*` param, MUST get a match or we raise.
    Silent partial loading would leave part of memory random.
    """
    params_path: str          # base or finetuned pi0.5 params dir
    stage1_ckpt: str          # path to stage1_step*.npz
    strict: bool = True

    def load(self, params: at.Params) -> at.Params:
        import flax.traverse_util as _trav
        # Tuple-key flatten handles the int block indices from mem_encoder.blocks (now a
        # plain list -> int child keys). sep="/" would TypeError on int keys, and the
        # downstream train.py replace_by_pure_dict requires int-keyed blocks. So we do the
        # base-subset merge + Stage-1 overlay entirely in tuple-key space.
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = _trav.flatten_dict(params)            # tuple keys, int-safe
        flat_loaded = _trav.flatten_dict(loaded)
        merged = {}
        # step 1: base pi0.5 weights that are a subset of the reference param tree.
        for k, v in flat_loaded.items():
            if k in flat_ref:
                merged[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        # all remaining reference keys (memory/FiLM/fresh) come from the reference shapes.
        for k, v in flat_ref.items():
            if k not in merged:
                merged[k] = v
        # step 2: overlay Stage-1 bottleneck -> mem_encoder, remapping the string npz keys
        #         to tuples with numeric path components as int (to match list-indexed blocks).
        s1 = np.load(self.stage1_ckpt, allow_pickle=True)
        mem_keys = [k for k in merged if len(k) and k[0] == "mem_encoder"]
        expected_architecture = (
            "gated_delta_hybrid_v1"
            if any("delta_blocks" in tuple(str(p) for p in key) for key in mem_keys)
            else "perceiver_v1"
        )
        metadata_key = "metadata/memory_architecture"
        if metadata_key in s1.files:
            actual_architecture = str(np.asarray(s1[metadata_key]).item())
            if actual_architecture != expected_architecture:
                raise ValueError(
                    "Stage-1 checkpoint architecture mismatch: "
                    f"{actual_architecture!r} != {expected_architecture!r}"
                )
        elif expected_architecture != "perceiver_v1":
            raise ValueError(
                f"Stage-1 checkpoint {self.stage1_ckpt} has no architecture metadata; "
                f"refusing to load it into {expected_architecture!r}"
            )

        def _s1_to_tuple(key):   # "bottleneck/blocks/0/attn/kernel" -> ("mem_encoder","blocks",0,"attn","kernel")
            parts = key.split("/"); parts[0] = "mem_encoder"
            return tuple(int(p) if p.isdigit() else p for p in parts)
        def _decoder_to_tuple(key):
            parts = key.split("/"); parts[0] = "aux_decoder"
            return tuple(int(p) if p.isdigit() else p for p in parts)

        aux_keys = [k for k in merged if len(k) and k[0] == "aux_decoder"]
        n_set, n_decoder, n_skip = 0, 0, 0
        for k in s1.files:
            if k.startswith("bottleneck/"):
                tgt = _s1_to_tuple(k)
            elif k.startswith("decoder/") and aux_keys:
                tgt = _decoder_to_tuple(k)
            else:
                n_skip += 1
                continue
            if tgt in merged:
                v = s1[k]
                merged[tgt] = v.astype(merged[tgt].dtype) if v.dtype != merged[tgt].dtype else v
                if tgt[0] == "mem_encoder":
                    n_set += 1
                else:
                    n_decoder += 1
            elif self.strict:
                raise KeyError(f"Stage-1 param '{k}' -> {tgt} not found in Stage-2 param tree.")
        # guard: every mem_encoder param in the model must have been filled from Stage-1.
        if self.strict and n_set != len(mem_keys):
            raise KeyError(
                f"Stage-1 transfer incomplete: filled {n_set}/{len(mem_keys)} mem_encoder params.")
        if self.strict and aux_keys and n_decoder != len(aux_keys):
            raise KeyError(
                "Stage-1 auxiliary decoder transfer incomplete: "
                f"filled {n_decoder}/{len(aux_keys)} aux_decoder params."
            )
        print(
            f"[Stage1BottleneckWeightLoader] transferred {n_set} bottleneck params -> mem_encoder, "
            f"{n_decoder} decoder params -> aux_decoder (discarded {n_skip} unrelated entries)."
        )
        return _trav.unflatten_dict(merged)              # tuple-key unflatten -> int-keyed blocks


####################### RoboMME #######################



@dataclasses.dataclass(frozen=True)
class MMEVLAWeightLoaderE2E(WeightLoader):
    """E2E loader: load VLA params (VLM + action expert) into the subset of the reference
    tree that matches; leave EVERYTHING else (mem_encoder/*, FiLM seam) at fresh RANDOM init.
    No Stage-1 overlay. Merge done in TUPLE-key space so the integer block indices in
    mem_encoder.blocks (blocks/0/..., list-indexed) don't crash flatten_dict(sep="/")."""
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        import flax.traverse_util as _trav
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = _trav.flatten_dict(params)          # tuple keys, int-safe
        flat_loaded = _trav.flatten_dict(loaded)
        merged = {}
        for k, v in flat_loaded.items():
            if k in flat_ref:
                merged[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        for k, v in flat_ref.items():                  # mem_encoder/FiLM/fresh -> reference (random) init
            if k not in merged:
                merged[k] = v
        return _trav.unflatten_dict(merged)

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "runs/assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "runs/ckpts"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Opt-in exact static-shape buckets for offline Stage-2 memory streams.
    length_bucketed_batches: bool = False
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    resum_ckpt_id: int | None = None

    dataset_path: str = "data/robomme"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")

_CONFIGS = [
    TrainConfig(
        name="pi05_baseline",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=False,
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id=f"robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="mme_vla_suite",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=True,
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id=f"robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
]

_CONFIGS.append(
    TrainConfig(
        name="mme_vla_suite_cosmo",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=True,
            history_config="cosmo-bottleneck-modul.yaml",
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id=f"robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000, save_interval=10_000, keep_period=10_000,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

_CONFIGS.append(
    TrainConfig(
        name="mme_vla_suite_wan",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=True,
            history_config="cosmo-wan-bottleneck-modul.yaml",
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id=f"robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000, save_interval=10_000, keep_period=10_000,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

# ------------------------------------------------------------------------- #
# Stage-2 memory-VLA finetune: TWO paired experiments.                      #
#   Each loads a DIFFERENT base pi0.5 and overlays a MATCHING Stage-1        #
#   bottleneck (features must come from the same encoder as the frozen img).#
#   freeze_filter = .*img.* only (ROBOMME recipe: freeze SigLIP encoder,    #
#   finetune the VLM transformer + action expert + FiLM seam).              #
#   STAGE1_CKPT_* are placeholders -> point at the real Stage-1 .npz once   #
#   Stage-1 has trained on the matching store.                              #
# ------------------------------------------------------------------------- #
_STAGE1_CKPT_FT = os.environ.get(
    "STAGE1_CKPT_FT", "artifacts/checkpoints/stage1_ft/stage1_final.npz")
_STAGE1_CKPT_BASE = os.environ.get(
    "STAGE1_CKPT_BASE", "artifacts/checkpoints/stage1_base/stage1_final.npz")
_WASHRETURN_FT_PARAMS = "artifacts/checkpoints/pi05_washreturn_toy/step-10000/params"
_STAGE1_CKPT = os.environ.get(
    "STAGE1_CKPT", "artifacts/checkpoints/stage1_ft_all/stage1_step5000.npz"
)
_STAGE2_PI05_PARAMS = os.environ.get("PI05_PARAMS",
    "artifacts/checkpoints/pi05_washreturn_toy/step-10000/params")
_STAGE2_STORE = os.environ.get("STAGE2_STORE", "data/washreturn_store_var8")

# Experiment 1: memory on the FINETUNED pi0.5 (washreturn-10k features + ckpt).
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_finetuned",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-s2nojitter.yaml", discrete_state_input=False,  # jitter opt-in default OFF; swap to cosmo-wan-bottleneck-modul.yaml to enable
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,  # ROBOMME batch (fits single B200; toy finetune ran b=64)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),  # freeze img+embedder+mem_encoder; train seam+action expert+VLM
        weight_loader=Stage1BottleneckWeightLoader(params_path=_STAGE2_PI05_PARAMS, stage1_ckpt=_STAGE1_CKPT),
        num_train_steps=20_000, save_interval=2_500, keep_period=2_500,  # ~26 epochs over 101 eps -> save often, pick best by EVAL not train loss
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    )
)

# MEMFT: like finetuned but TRAINABLE bottleneck.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_finetuned_memft",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-s2nojitter.yaml", discrete_state_input=False,  # jitter opt-in default OFF; swap to cosmo-wan-bottleneck-modul.yaml to enable
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,  # ROBOMME batch (fits single B200; toy finetune ran b=64)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),  # MEMFT: trainable bottleneck (freeze only img)
        weight_loader=Stage1BottleneckWeightLoader(params_path=_STAGE2_PI05_PARAMS, stage1_ckpt=_STAGE1_CKPT),
        num_train_steps=20_000, save_interval=2_500, keep_period=2_500,  # ~26 epochs over 101 eps -> save often, pick best by EVAL not train loss
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    )
)

# Experiment 3: TRAINABLE memory encoder (finetune the bottleneck DURING VLA finetuning).
# Same as mme_wan_stage2_finetuned (offline precomputed Wan latents, jitter from the
# v0-v7 variant selection -- NO online Wan recompute) EXCEPT the freeze filter is the
# BASE one (freeze only .*img.*), so mem_encoder is TRAINABLE and finetunes alongside
# the VLA. Init from the ft_all Stage-1 bottleneck. 40k steps, FSDP 4 GPUs.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_memft",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-s2nojitter.yaml", discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=40_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),  # freeze ONLY img; mem_encoder TRAINABLE
        weight_loader=Stage1BottleneckWeightLoader(params_path=_STAGE2_PI05_PARAMS, stage1_ckpt=_STAGE1_CKPT),
        num_train_steps=40_000, save_interval=5_000, keep_period=5_000,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

# Experiment 2: memory on BASE pi0.5 (pi05_base features + ckpt).
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_base",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-s2nojitter.yaml", discrete_state_input=False,  # jitter opt-in default OFF; swap to cosmo-wan-bottleneck-modul.yaml to enable
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,  # ROBOMME batch (fits single B200; toy finetune ran b=64)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),  # freeze img+embedder+mem_encoder; train seam+action expert+VLM
        weight_loader=Stage1BottleneckWeightLoader(params_path=_STAGE2_PI05_PARAMS, stage1_ckpt=_STAGE1_CKPT),
        num_train_steps=20_000, save_interval=2_500, keep_period=2_500,  # ~26 epochs over 101 eps -> save often, pick best by EVAL not train loss
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
# E2E: memft-style TRAINABLE bottleneck but NO Stage-1 pretrain -- mem_encoder random, whole pipeline supervised by action loss only. 50k steps.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_e2e",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-s2nojitter.yaml", discrete_state_input=False,  # jitter opt-in default OFF; swap to cosmo-wan-bottleneck-modul.yaml to enable
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,  # ROBOMME batch (fits single B200; toy finetune ran b=64)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),  # MEMFT: trainable bottleneck (freeze only img); train mem_encoder+seam+action expert+VLM
        weight_loader=MMEVLAWeightLoaderE2E(params_path=_STAGE2_PI05_PARAMS),  # E2E: VLA only, mem_encoder RANDOM (tuple-key merge, int-block safe)
        num_train_steps=50_000, save_interval=5_000, keep_period=5_000,  # E2E from scratch, 50k steps
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    )
)
# E2E + AUX: same as mme_wan_stage2_e2e (from scratch, VLA-only weights, bottleneck+decoder
# RANDOM), PLUS the Stage-1 reconstruction losses (align/ap/reward) as auxiliary regularizers on
# the bottleneck tokens. action loss stays PRIMARY (weight 1.0); aux group weighted by aux_weight
# (effective ~0.15). Aux is bottleneck-only (grads reach mem_encoder+aux_decoder, not the VLA).
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_e2e_aux",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            # dedicated YAML adds `aux_weight: 0.25` so the DATASET (which reads the YAML) loads aux
            # targets; the model fields below drive the decoder + loss weighting.
            history_config="cosmo-wan-bottleneck-modul-s2nojitter-aux.yaml", discrete_state_input=False,
            aux_weight=0.25, aux_weight_end=0.05, aux_reward_w=0.1,  # -> effective lambda ~0.15 (midpoint)
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),  # MEMFT: trainable bottleneck+decoder
        weight_loader=MMEVLAWeightLoaderE2E(params_path=_STAGE2_PI05_PARAMS),
        num_train_steps=50_000, save_interval=5_000, keep_period=5_000,
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    ),
)

# E2E + AUX with LINEAR-ANNEALED lambda: identical to mme_wan_stage2_e2e_aux except aux_anneal=True,
# so the aux weight ramps aux_weight(0.25) -> aux_weight_end(0.05) over training (train_fraction =
# step/num_train_steps) instead of the fixed 0.15 midpoint. wandb aux/lambda will now decay 0.25->0.05.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_e2e_aux_anneal",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-s2nojitter-aux.yaml", discrete_state_input=False,
            aux_weight=0.25, aux_weight_end=0.05, aux_reward_w=0.1, aux_anneal=True,  # linear 0.25 -> 0.05
        ),
        data=RoboMMEDataConfig(repo_id="robomme", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoaderE2E(params_path=_STAGE2_PI05_PARAMS),
        num_train_steps=50_000, save_interval=5_000, keep_period=5_000,
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    ),
)


# ======================================================================================
# RoboDojo memory-benchmark configs — APPEND to mme_vla_suite/training/config.py
# (before the `_CONFIGS_DICT = {...}` line, alongside the robocasa configs).
#
# Requires at top of config.py:
#   from mme_vla_suite.policies.robodojo_policy import RoboDojoInputs, RoboDojoOutputs
#
# Mirrors the robocasa pipeline 1:1, retargeted to RoboDojo bimanual joint-space:
#   - 14-dim action/state (left_joint_0..6 + right_joint_0..6), padded to 32 by pi0.5
#   - 3 cameras: cam_high -> base_0_rgb, cam_left_wrist / cam_right_wrist -> wrist slots
#   - store built from the 6 memory tasks (task_index {3,11,16,25,32,33})
#
# THREE configs, all starting from pi05_base (per request):
#   (A) pi05_robodojo            — normal OpenPI no-memory finetune (baseline)
#   (B) mme_wan_stage2_robodojo_base — memory bottleneck (cosmo-wan) on base pi0.5
#   (C) mme_wan_stage2_robodojo_e2e  — E2E trainable bottleneck (no Stage-1), action-only
# ======================================================================================

# RoboDojo LeRobot store paths (env-overridable, mirrors _STAGE2_STORE convention).
_ROBODOJO_STORE = os.environ.get("ROBODOJO_STORE", "data/robodojo_mem_store")
_ROBODOJO_LEROBOT = os.environ.get("ROBODOJO_LEROBOT", "data/robodojo_mem_lerobot")
_ROBODOJO_KEY_EVENT_MANIFEST = os.environ.get(
    "ROBODOJO_KEY_EVENT_MANIFEST",
    "data/robodojo_key_event_labels/cover_blocks_key_events_v1.json",
)
_ROBODOJO_PI05_PARAMS = os.environ.get(
    "ROBODOJO_PI05_PARAMS",
    os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
)


@dataclasses.dataclass(frozen=True)
class RoboDojoDataConfig(DataConfigFactory):
    """RoboDojo bimanual-joint data factory. Identical structure to RoboMMEDataConfig
    (same memory-stream repack + aux gating) but wires RoboDojoInputs/Outputs and the
    3 RoboDojo camera keys instead of robocasa's."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # cam_high -> image, cam_left_wrist -> wrist_image, cam_right_wrist -> right_image
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/right_image": "right_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # perceptual memory
                        "static_image_emb": "static_image_emb",
                        "static_pos_emb": "static_pos_emb",
                        "static_state_emb": "static_state_emb",
                        "static_mask": "static_mask",
                        "static_time": "static_time",
                        # recurrent memory
                        "recur_image_emb": "recur_image_emb",
                        "recur_pos_emb": "recur_pos_emb",
                        "recur_state_emb": "recur_state_emb",
                        "recur_mask": "recur_mask",
                        # symbolic memory
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                    }
                )
            ]
        )

        # cosmo_bottleneck + precompute + aux stream gating — copied verbatim from RoboMMEDataConfig.
        try:
            _hc = get_history_config(model_config.history_config) if model_config.history_config is not None else None
            _cbm = getattr(_hc, "cosmo_bottleneck_memory", None) if _hc is not None else None
            _precomp = bool(str(getattr(_cbm, "precompute_memory_dir", "") or "")) if _cbm is not None else False
        except Exception:
            _precomp = False
            _cbm = None
        if _cbm is not None:
            for _k in ("cosmo_latent_emb", "cosmo_mem_mask", "cosmo_recent_emb",
                       "cosmo_recent_mask", "cosmo_action_emb", "cosmo_action_mask",
                       "cosmo_proprio_emb", "cosmo_proprio_mask", "cosmo_lang_emb"):
                repack_transform.inputs[0].structure[_k] = _k
        if _precomp:
            repack_transform.inputs[0].structure["cosmo_mem_tokens"] = "cosmo_mem_tokens"
        if float(getattr(model_config, "aux_weight", 0.0) or 0.0) > 0.0:
            for _k in ("aux_query_steps", "aux_query_mask", "aux_align_target",
                       "aux_action_target", "aux_proprio_target", "aux_reward_target"):
                repack_transform.inputs[0].structure[_k] = _k
        if (
            float(getattr(model_config, "late_episode_loss_weight", 1.0) or 1.0) > 1.0
            or bool(getattr(model_config, "key_event_manifest_path", None))
        ):
            repack_transform.inputs[0].structure["sample_loss_weight"] = "sample_loss_weight"

        data_transforms = _transforms.Group(
            inputs=[RoboDojoInputs(model_type=model_config.model_type)],
            outputs=[RoboDojoOutputs()],
        )
        # RoboDojo joint actions are absolute joint targets (not delta) -> no DeltaActions,
        # consistent with how robocasa's already-relative actions are left untouched.
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# (A) No-memory OpenPI finetune from pi05_base — the RoboDojo analog of `pi05_baseline`.
_CONFIGS.append(
    TrainConfig(
        name="pi05_robodojo",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=False,
            history_config=None, discrete_state_input=False,
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_ROBODOJO_STORE,  # no-mem trains on the SAME cosmo store (use_history=False ignores mem streams); store provides per-step image/state/action + normstats via train_mme
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoader(_ROBODOJO_PI05_PARAMS),
        num_train_steps=80_000, save_interval=10_000, keep_period=10_000,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

# (B) Memory bottleneck (cosmo-wan) on BASE pi0.5 — RoboDojo analog of mme_wan_stage2_base.
#     Requires _ROBODOJO_STORE built from the 6 memory tasks + a matching Stage-1 ckpt.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_robodojo_base",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-robodojo-s2nojitter.yaml", discrete_state_input=False,
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_ROBODOJO_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_ROBODOJO_PI05_PARAMS,
            stage1_ckpt=os.environ.get("ROBODOJO_STAGE1_CKPT", _STAGE1_CKPT_BASE)),
        num_train_steps=20_000, save_interval=10_000, keep_period=1,
        num_workers=4, ema_decay=0.999, fsdp_devices=1,
    )
)

# (B-memft) MEMFT: like (B) but memory encoder TRAINABLE (freeze only img).
#     Mirrors robocasa mme_wan_stage2_memft: init pi05_base + Stage-1 bottleneck ckpt,
#     then finetune the memory encoder alongside the VLA. 40k steps, fsdp 4.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_robodojo_memft",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-robodojo-s2nojitter.yaml", discrete_state_input=False,
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_ROBODOJO_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=40_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),  # MEMFT: trainable bottleneck (freeze only img); train mem_encoder+seam+action expert+VLM
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_ROBODOJO_PI05_PARAMS,
            stage1_ckpt=os.environ.get("ROBODOJO_STAGE1_CKPT", _STAGE1_CKPT_BASE)),
        num_train_steps=40_000, save_interval=10_000, keep_period=1,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

# (C) E2E trainable bottleneck (no Stage-1 pretrain), action-only — RoboDojo analog of
#     mme_wan_stage2_e2e. Use this if we skip Stage-1 / have no reward curve for RoboDojo.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_robodojo_e2e",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=20, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-robodojo-s2nojitter.yaml", discrete_state_input=False,
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_ROBODOJO_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoaderE2E(params_path=_ROBODOJO_PI05_PARAMS),
        num_train_steps=50_000, save_interval=10_000, keep_period=1,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

# Frozen stage-2 followups: 60k, horizon 30, precomputed 60k stage-1 memory tokens.
_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_robodojo_base_h30_headdrop05_60k",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=30, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-robodojo-s2nojitter-precomputed.yaml",
            discrete_state_input=False,
            current_obs_dropout_prob=0.5,
            current_obs_dropout_image_keys="base_0_rgb",
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_ROBODOJO_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=60_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_ROBODOJO_PI05_PARAMS,
            stage1_ckpt=os.environ.get("ROBODOJO_STAGE1_CKPT", _STAGE1_CKPT_BASE)),
        num_train_steps=60_000, save_interval=10_000, keep_period=1,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_robodojo_base_h30_late50w3_60k",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=30, use_history=True,
            history_config="cosmo-wan-bottleneck-modul-robodojo-s2nojitter-precomputed.yaml",
            discrete_state_input=False,
            late_episode_loss_weight=3.0,
            late_episode_loss_fraction=0.5,
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=_ROBODOJO_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=60_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_ROBODOJO_PI05_PARAMS,
            stage1_ckpt=os.environ.get("ROBODOJO_STAGE1_CKPT", _STAGE1_CKPT_BASE)),
        num_train_steps=60_000, save_interval=10_000, keep_period=1,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)


_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_robodojo_base_h30_keyevents3w5_60k",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=30,
            use_history=True,
            history_config="cosmo-wan-bottleneck-modul-robodojo-s2nojitter-precomputed.yaml",
            discrete_state_input=False,
            key_event_manifest_path=_ROBODOJO_KEY_EVENT_MANIFEST,
            key_event_loss_weight=3.0,
            key_event_window_radius=5,
        ),
        data=RoboDojoDataConfig(
            repo_id="robodojo",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_ROBODOJO_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=60_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_ROBODOJO_PI05_PARAMS,
            stage1_ckpt=os.environ.get(
                "ROBODOJO_STAGE1_CKPT", _STAGE1_CKPT_BASE
            ),
        ),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=1,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)


_CONFIGS.append(
    TrainConfig(
        name="robomme_robodojo_perceptual_h30_50k",
        project_name="RoboDojo",
        model=history_pi0.HistoryPi0Config(
            pi05=True, action_horizon=30, use_history=True,
            history_config="perceptual-framesamp-modul-robodojo-3view.yaml",
            discrete_state_input=False,
        ),
        data=RoboDojoDataConfig(repo_id="robodojo", base_config=DataConfig(prompt_from_task=True)),
        dataset_path=os.environ.get(
            "ROBODOJO_ROBOMME_STORE",
            "data/robomme_robodojo_cover_h30_3view",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=MMEVLAWeightLoader(_ROBODOJO_PI05_PARAMS),
        num_train_steps=50_000, save_interval=10_000, keep_period=1,
        num_workers=4, ema_decay=0.999, fsdp_devices=4,
    )
)

# Native pi0 Stage-2. The DINO Stage-1 bottleneck is VLA-independent: its
# 256x1024 tokens feed the 1024-wide pi0 action expert through the same memory
# modulation seam, while pi0_base supplies the native state/action-time path.
_PI0_BASE_PARAMS = os.environ.get(
    "PI0_PARAMS",
    "gs://openpi-assets/checkpoints/pi0_base/params",
)
_PI0_DINO_STAGE1_CKPT = os.environ.get(
    "PI0_STAGE1_CKPT",
    "artifacts/checkpoints/"
    "stage1_rmbench_all9_v2causal_dinov3_a1_action05_prop05_reward05_nl256_160k_20260819/"
    "stage1_step160000.npz",
)
_PI0_DINO_STAGE2_STORE = os.environ.get(
    "PI0_STAGE2_STORE",
    "data/rmbench_all9_pi05_base_var8_h50_stage2_compact_"
    "causal_action_initial_v2_dinov3",
)
_PI0_DINO_HISTORY_CONFIG = os.environ.get(
    "PI0_HISTORY_CONFIG",
    "cosmo-wan-bottleneck-modul-rmbench-all9-dinov3-frozen-precomputed.yaml",
)
_PI0_DINO_MEMFT_HISTORY_CONFIG = os.environ.get(
    "PI0_MEMFT_HISTORY_CONFIG",
    "cosmo-wan-bottleneck-modul-rmbench-all9-dinov3-memft.yaml",
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi0_frozen_precomputed",
        project_name="Robocasa",
        model=memory_pi0.MemoryPi0Config(
            action_horizon=50,
            max_token_len=64,
            use_history=True,
            history_config=_PI0_DINO_HISTORY_CONFIG,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_PI0_DINO_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=memory_pi0.MemoryPi0Config().get_stage2_frozen_bottleneck_freeze_filter(),
        weight_loader=NativePi0Stage1BottleneckWeightLoader(
            params_path=_PI0_BASE_PARAMS,
            stage1_ckpt=_PI0_DINO_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi0_memft",
        project_name="Robocasa",
        model=memory_pi0.MemoryPi0Config(
            action_horizon=50,
            max_token_len=64,
            use_history=True,
            history_config=_PI0_DINO_MEMFT_HISTORY_CONFIG,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_PI0_DINO_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Match o3rsq5cf: only the VLA image tower is frozen. The Stage-1
        # memory encoder is updated jointly with the action policy.
        freeze_filter=memory_pi0.MemoryPi0Config().get_freeze_filter(),
        weight_loader=NativePi0Stage1BottleneckWeightLoader(
            params_path=_PI0_BASE_PARAMS,
            stage1_ckpt=_PI0_DINO_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

_PI0_ROBOMME_STAGE1_CKPT = os.environ.get(
    "PI0_ROBOMME_STAGE1_CKPT",
    "artifacts/checkpoints/"
    "stage1_robomme_imitation4_v1causal_dinov3_a1_action05_prop05_reward05_"
    "nl256_accel_160k_20260904/stage1_step160000.npz",
)
_PI0_ROBOMME_STAGE2_STORE = os.environ.get(
    "PI0_ROBOMME_STAGE2_STORE",
    "data/robomme_imitation4_pi0_h50_stage2_causal_dinov3_20260904",
)
_PI0_ROBOMME_HISTORY_CONFIG = os.environ.get(
    "PI0_ROBOMME_HISTORY_CONFIG",
    "cosmo-wan-bottleneck-modul-robomme-imitation4-dinov3-memft.yaml",
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi0_robomme_imitation4_memft_freeze_language",
        project_name="Robocasa",
        model=memory_pi0.MemoryPi0Config(
            action_horizon=50,
            max_token_len=64,
            use_history=True,
            history_config=_PI0_ROBOMME_HISTORY_CONFIG,
        ),
        data=RoboMMEJointDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_PI0_ROBOMME_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=memory_pi0.MemoryPi0Config().get_image_and_language_encoder_freeze_filter(),
        weight_loader=NativePi0Stage1BottleneckWeightLoader(
            params_path=_PI0_BASE_PARAMS,
            stage1_ckpt=_PI0_ROBOMME_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

_PI05_ROBOMME_PARAMS = os.environ.get(
    "PI05_ROBOMME_PARAMS",
    os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
)
_PI05_ROBOMME_STAGE1_CKPT = os.environ.get(
    "PI05_ROBOMME_STAGE1_CKPT",
    "artifacts/checkpoints/"
    "stage1_robomme_imitation4_v1causal_dinov3_a1_action05_prop05_reward05_"
    "nl256_accel_160k_20260904/stage1_step160000.npz",
)
_PI05_ROBOMME_STAGE2_STORE = os.environ.get(
    "PI05_ROBOMME_STAGE2_STORE",
    "data/robomme_imitation4_pi05_h20_stage2_causal_dinov3_20260904",
)
_PI05_ROBOMME_HISTORY_CONFIG = os.environ.get(
    "PI05_ROBOMME_HISTORY_CONFIG",
    "cosmo-wan-bottleneck-modul-robomme-imitation4-dinov3-memft.yaml",
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi05_robomme_imitation4_memft_freeze_language",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            max_token_len=64,
            use_history=True,
            history_config=_PI05_ROBOMME_HISTORY_CONFIG,
            discrete_state_input=False,
        ),
        data=RoboMMEJointDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_PI05_ROBOMME_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_image_and_language_encoder_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_PI05_ROBOMME_PARAMS,
            stage1_ckpt=_PI05_ROBOMME_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

# Opt-in Gated DeltaNet Stage-2 profiles. These use separate history YAMLs and
# require a Stage-1 checkpoint carrying gated_delta_hybrid_v1 metadata, so they
# cannot silently consume a legacy Perceiver checkpoint.
_GATED_DELTA_PI05_PARAMS = os.environ.get(
    "GATED_DELTA_PI05_PARAMS",
    os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
)
_GATED_DELTA_STAGE1_CKPT = os.environ.get(
    "GATED_DELTA_STAGE1_CKPT",
    "artifacts/checkpoints/stage1_gated_delta_rmbench_all9_dinov3/"
    "stage1_step160000.npz",
)
_GATED_DELTA_STAGE2_STORE = os.environ.get(
    "GATED_DELTA_STAGE2_STORE",
    "data/rmbench_all9_pi05_base_var8_h50_stage2_compact_"
    "causal_action_initial_v2_dinov3",
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi05_gated_delta_memft",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=64,
            use_history=True,
            history_config="cosmo-wan-gated-delta-rmbench-all9-dinov3-memft.yaml",
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_GATED_DELTA_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_GATED_DELTA_PI05_PARAMS,
            stage1_ckpt=_GATED_DELTA_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi05_gated_delta_memft_optimized",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=64,
            use_history=True,
            history_config=(
                "cosmo-wan-gated-delta-rmbench-all9-dinov3-memft-optimized.yaml"
            ),
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_GATED_DELTA_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_GATED_DELTA_PI05_PARAMS,
            stage1_ckpt=_GATED_DELTA_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        length_bucketed_batches=True,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

_CONFIGS.append(
    TrainConfig(
        name="mme_wan_stage2_pi05_gated_delta_memft_aux",
        project_name="Robocasa",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=64,
            use_history=True,
            history_config="cosmo-wan-gated-delta-rmbench-all9-dinov3-memft-aux.yaml",
            discrete_state_input=False,
            aux_weight=0.25,
            aux_weight_end=0.25,
            aux_anneal=False,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        dataset_path=_GATED_DELTA_STAGE2_STORE,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=Stage1BottleneckWeightLoader(
            params_path=_GATED_DELTA_PI05_PARAMS,
            stage1_ckpt=_GATED_DELTA_STAGE1_CKPT,
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    )
)

# Canonical paper-release profiles. Paths are supplied through environment
# variables rather than fixed filesystem locations.
_IMPRINT_PI05_PARAMS = os.environ.get(
    "IMPRINT_PI05_PARAMS",
    "gs://openpi-assets/checkpoints/pi05_base/params",
)
_IMPRINT_RMBENCH_STORE = os.environ.get(
    "IMPRINT_RMBENCH_STORE",
    "data/rmbench-450",
)
_IMPRINT_ROBOCASA_STORE = os.environ.get(
    "IMPRINT_ROBOCASA_STORE",
    "data/robocasa-600",
)
_IMPRINT_RMBENCH_ATTENTION_STAGE1 = os.environ.get(
    "IMPRINT_RMBENCH_ATTENTION_STAGE1",
    "artifacts/checkpoints/rmbench-full-attention/stage1/stage1_step160000.npz",
)
_IMPRINT_RMBENCH_RECURRENT_STAGE1 = os.environ.get(
    "IMPRINT_RMBENCH_RECURRENT_STAGE1",
    "artifacts/checkpoints/rmbench-recurrent/stage1/stage1_step160000.npz",
)
_IMPRINT_ROBOCASA_ATTENTION_STAGE1 = os.environ.get(
    "IMPRINT_ROBOCASA_ATTENTION_STAGE1",
    "artifacts/checkpoints/robocasa-full-attention/stage1/stage1_step160000.npz",
)
_IMPRINT_ROBOCASA_RECURRENT_STAGE1 = os.environ.get(
    "IMPRINT_ROBOCASA_RECURRENT_STAGE1",
    "artifacts/checkpoints/robocasa-recurrent/stage1/stage1_step160000.npz",
)

_CONFIGS.extend(
    [
        TrainConfig(
            name="imprint_rmbench_full_attention_memft",
            project_name="Robocasa",
            model=history_pi0.HistoryPi0Config(
                pi05=True,
                action_horizon=50,
                max_token_len=64,
                use_history=True,
                history_config=(
                    "cosmo-wan-bottleneck-modul-rmbench-all9-dinov3-memft.yaml"
                ),
                discrete_state_input=False,
            ),
            data=RoboMMEDataConfig(
                repo_id="robomme",
                base_config=DataConfig(prompt_from_task=True),
            ),
            dataset_path=_IMPRINT_RMBENCH_STORE,
            batch_size=64,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=100_000,
                decay_lr=2.5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            freeze_filter=(
                history_pi0.HistoryPi0Config()
                .get_image_and_language_embedder_freeze_filter()
            ),
            weight_loader=Stage1BottleneckWeightLoader(
                params_path=_IMPRINT_PI05_PARAMS,
                stage1_ckpt=_IMPRINT_RMBENCH_ATTENTION_STAGE1,
            ),
            num_train_steps=100_000,
            save_interval=10_000,
            keep_period=10_000,
            num_workers=4,
            ema_decay=0.999,
            fsdp_devices=4,
        ),
        TrainConfig(
            name="imprint_rmbench_recurrent_memft",
            project_name="Robocasa",
            model=history_pi0.HistoryPi0Config(
                pi05=True,
                action_horizon=50,
                max_token_len=64,
                use_history=True,
                history_config=(
                    "cosmo-wan-gated-delta-rmbench-all9-dinov3-memft-optimized.yaml"
                ),
                discrete_state_input=False,
            ),
            data=RoboMMEDataConfig(
                repo_id="robomme",
                base_config=DataConfig(prompt_from_task=True),
            ),
            dataset_path=_IMPRINT_RMBENCH_STORE,
            batch_size=64,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=100_000,
                decay_lr=2.5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
            weight_loader=Stage1BottleneckWeightLoader(
                params_path=_IMPRINT_PI05_PARAMS,
                stage1_ckpt=_IMPRINT_RMBENCH_RECURRENT_STAGE1,
            ),
            num_train_steps=100_000,
            save_interval=10_000,
            keep_period=10_000,
            num_workers=4,
            length_bucketed_batches=True,
            ema_decay=0.999,
            fsdp_devices=4,
        ),
        TrainConfig(
            name="imprint_robocasa_full_attention_memft",
            project_name="Robocasa",
            model=history_pi0.HistoryPi0Config(
                pi05=True,
                action_horizon=20,
                max_token_len=64,
                use_history=True,
                history_config=(
                    "cosmo-wan-bottleneck-modul-robocasa-combined600-"
                    "dinov3-memft.yaml"
                ),
                discrete_state_input=False,
            ),
            data=RoboMMEDataConfig(
                repo_id="robomme",
                base_config=DataConfig(prompt_from_task=True),
            ),
            dataset_path=_IMPRINT_ROBOCASA_STORE,
            batch_size=64,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=150_000,
                decay_lr=2.5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
            weight_loader=Stage1BottleneckWeightLoader(
                params_path=_IMPRINT_PI05_PARAMS,
                stage1_ckpt=_IMPRINT_ROBOCASA_ATTENTION_STAGE1,
            ),
            num_train_steps=150_000,
            save_interval=10_000,
            keep_period=10_000,
            num_workers=4,
            ema_decay=0.999,
            fsdp_devices=1,
        ),
        TrainConfig(
            name="imprint_robocasa_recurrent_memft",
            project_name="Robocasa",
            model=history_pi0.HistoryPi0Config(
                pi05=True,
                action_horizon=20,
                max_token_len=64,
                use_history=True,
                history_config=(
                    "cosmo-wan-gated-delta-robocasa-combined600-"
                    "dinov3-memft-optimized.yaml"
                ),
                discrete_state_input=False,
            ),
            data=RoboMMEDataConfig(
                repo_id="robomme",
                base_config=DataConfig(prompt_from_task=True),
            ),
            dataset_path=_IMPRINT_ROBOCASA_STORE,
            batch_size=64,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=150_000,
                decay_lr=2.5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
            weight_loader=Stage1BottleneckWeightLoader(
                params_path=_IMPRINT_PI05_PARAMS,
                stage1_ckpt=_IMPRINT_ROBOCASA_RECURRENT_STAGE1,
            ),
            num_train_steps=150_000,
            save_interval=10_000,
            keep_period=10_000,
            num_workers=4,
            length_bucketed_batches=True,
            ema_decay=0.999,
            fsdp_devices=4,
        ),
    ]
)


_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
