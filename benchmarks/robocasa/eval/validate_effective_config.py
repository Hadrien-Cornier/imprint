#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--history-name", required=True)
    parser.add_argument("--stage1", type=pathlib.Path, required=True)
    parser.add_argument("--store", type=pathlib.Path, required=True)
    parser.add_argument("--pi05", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-base", type=pathlib.Path, required=True)
    parser.add_argument("--exp-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["GATED_DELTA_PI05_PARAMS"] = str(args.pi05)
    os.environ["GATED_DELTA_STAGE1_CKPT"] = str(args.stage1)
    os.environ["GATED_DELTA_STAGE2_STORE"] = str(args.store)

    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training import config as train_config
    from openpi.models.model import ModelType

    sys.argv = [
        "train_mme.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        f"--dataset-path={args.store}",
        f"--model.history-config={args.history_name}",
        "--model.action-horizon=20",
        f"--weight-loader.params-path={args.pi05}",
        f"--weight-loader.stage1-ckpt={args.stage1}",
        "--batch-size=64",
        "--num-train-steps=150000",
        "--lr-schedule.warmup-steps=500",
        "--lr-schedule.decay-steps=150000",
        "--fsdp-devices=4",
        "--num-workers=4",
        "--save-interval=10000",
        "--keep-period=10000",
        f"--checkpoint-base-dir={args.checkpoint_base}",
        "--seed=42",
        "--ema-decay=0.999",
    ]
    config = train_config.cli()
    history = get_history_config(config.model.history_config)
    memory = history.cosmo_bottleneck_memory
    decoder = history.cosmo_decoder

    freeze_pattern = getattr(
        getattr(config.freeze_filter, "pattern", None),
        "pattern",
        None,
    )
    assert config.model.model_type is ModelType.PI05
    assert config.model.pi05 is True
    assert config.model.use_history is True
    assert config.model.action_horizon == 20
    assert config.model.history_config == args.history_name
    assert pathlib.Path(config.dataset_path) == args.store
    assert pathlib.Path(config.weight_loader.params_path) == args.pi05
    assert pathlib.Path(config.weight_loader.stage1_ckpt) == args.stage1
    assert config.batch_size == 64
    assert config.num_train_steps == 150_000
    assert config.lr_schedule.warmup_steps == 500
    assert config.lr_schedule.decay_steps == 150_000
    assert config.lr_schedule.peak_lr == 2.5e-5
    assert config.lr_schedule.decay_lr == 2.5e-6
    assert config.fsdp_devices == 4
    assert config.num_workers == 4
    assert config.length_bucketed_batches is True
    assert config.save_interval == 10_000
    assert config.keep_period == 10_000
    assert freeze_pattern == ".*img.*", freeze_pattern

    expected_memory = {
        "memory_encoder_type": "gated_delta_hybrid",
        "memory_architecture_version": "gated_delta_hybrid_v1",
        "memory_compute_dtype": "bfloat16",
        "visual_feature_stream": "dinov3",
        "latent_source": "offline",
        "num_latents": 256,
        "latent_cap": 4096,
        "max_recent_tokens": 768,
        "max_step_tokens": 1776,
        "max_stream_steps": 1776,
        "action_in_dim": 12,
        "proprio_in_dim": 16,
        "delta_full_scan_mode": "chunkwise_wy",
        "delta_wy_matmul_dtype": "bfloat16",
        "delta_event_order": "action_proprio_then_finalized_visual_v1",
    }
    actual_memory = {
        key: memory.get(key)
        for key in expected_memory
    }
    assert actual_memory == expected_memory, (actual_memory, expected_memory)
    assert list(memory.latent_buckets) == [256, 512, 1024, 2048, 4096]
    assert memory.n_variants == 8
    assert memory.num_query_samples == 16
    assert memory.downsample == 2
    assert memory.pool_hw == 2
    assert decoder.action_dim == 12
    assert decoder.proprio_dim == 16
    assert decoder.align_dim == 768
    assert decoder.align_num_patches == 192
    assert history.action_horizon == 20

    assert args.stage1.is_file()
    assert args.stage1.stat().st_size > 300_000_000
    with np.load(args.stage1, allow_pickle=True) as stage1:
        architecture = str(
            np.asarray(stage1["metadata/memory_architecture"]).item()
        )
        assert architecture == "gated_delta_hybrid_v1"
        assert stage1["bottleneck/latents"].shape == (256, 1024)
        delta_keys = [
            key for key in stage1.files
            if key.startswith("bottleneck/delta_blocks/")
        ]
        assert delta_keys

    assert (args.pi05 / "_METADATA").is_file()
    validation = json.loads(
        (args.store / "meta/combined600_store_validation.json").read_text()
    )
    compact = json.loads(
        (args.store / "meta/stage1_compact.json").read_text()
    )
    causal = json.loads(
        (args.store / "meta/causal_action_v1.json").read_text()
    )
    assert validation["status"] == "PASS"
    assert validation["episodes"] == 600
    assert validation["broad_tasks"] == 12
    assert compact["bounds"]["max_steps"] == 1776
    assert causal["status"] == "PASS"
    assert causal["episodes"] == 600
    assert causal["action_contract"]["action_causal_t"] == "action[t-1]"

    episode = args.store / "features/episode_0"
    action_proprio = np.load(
        episode / "action_proprio.npy",
        allow_pickle=True,
    ).item()
    action = np.asarray(action_proprio["action"])
    action_causal = np.asarray(action_proprio["action_causal"])
    proprio = np.asarray(action_proprio["proprio"])
    dino = np.load(episode / "dinov3.npy", mmap_mode="r")
    assert action.shape[-1] == 12
    assert action_causal.shape == action.shape
    assert np.allclose(action_causal[0], 0.0)
    assert np.allclose(action_causal[1:], action[:-1])
    assert proprio.shape[-1] == 16
    assert dino.shape[1:] == (3, 64, 768)

    provenance = {
        "effective_config": {
            "base": "pi05_base",
            "config_name": config.name,
            "history_config": config.model.history_config,
            "action_horizon": config.model.action_horizon,
            "batch_size": config.batch_size,
            "steps": config.num_train_steps,
            "warmup_steps": config.lr_schedule.warmup_steps,
            "decay_steps": config.lr_schedule.decay_steps,
            "fsdp_devices": config.fsdp_devices,
            "length_bucketed_batches": config.length_bucketed_batches,
            "freeze": "image_encoder_only",
        },
        "memory": actual_memory,
        "stage1": {
            "path": str(args.stage1),
            "sha256": sha256(args.stage1),
            "architecture": "gated_delta_hybrid_v1",
            "latent_shape": [256, 1024],
        },
        "store": {
            "path": str(args.store),
            "episodes": 600,
            "tasks": 12,
            "causal_actions": True,
        },
    }
    output = args.checkpoint_base.parent / args.exp_name / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(
        "ROBOCASA600_GATED_DELTA_STAGE2_PREFLIGHT_OK "
        "base=pi05 checkpoint=160k steps=150000 horizon=20 "
        "caps=4096/768/1776 action=causal "
        "memory=gated_delta_hybrid_v1 scan=chunkwise_wy "
        f"stage1_sha256={provenance['stage1']['sha256']}"
    )


if __name__ == "__main__":
    main()
