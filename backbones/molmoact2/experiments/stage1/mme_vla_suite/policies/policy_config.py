import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import dataclasses

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config



def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.MME_VLA_Policy:

    repack_transforms = repack_transforms or transforms.Group()

    logging.info(f"Checking history config")
    history_config = None
    history_config_path = checkpoint_dir.parent / "history_config.txt"
    if history_config_path.exists():
        with open(history_config_path, "r") as f:
            history_config = f.read()

    if train_config.model.history_config != history_config:
        print(f" == You are using {train_config.model.history_config}, changing to {history_config} ==")
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, history_config=history_config, use_history=history_config is not None)
        )


    logging.info("Loading model...")
    _restored = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    # FIX(int-vs-str block keys): the model builds mem_encoder.blocks as a plain Python list
    # -> nnx assigns INTEGER child keys (0,1), but orbax serializes dict keys as STRINGS
    # ("0","1"). model.load's structural equality check fails on this metadata mismatch.
    # Coerce numeric-string keys back to int under mem_encoder/blocks before load.
    def _intify_numeric_keys(tree):
        if isinstance(tree, dict):
            return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intify_numeric_keys(v)
                    for k, v in tree.items()}
        return tree
    if isinstance(_restored, dict) and "mem_encoder" in _restored             and isinstance(_restored["mem_encoder"], dict) and "blocks" in _restored["mem_encoder"]:
        _restored["mem_encoder"]["blocks"] = _intify_numeric_keys(_restored["mem_encoder"]["blocks"])
    model = train_config.model.load(_restored)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return _policy.MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm
    )
