"""Benchmark a complete synthetic Stage-1 train step at production shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from flax import nnx
import jax
import numpy as np
from omegaconf import OmegaConf
import optax

import openpi.training.sharding as sharding
from mme_vla_suite.scripts.pretrain_bottleneck import build_optimizer
from mme_vla_suite.training.length_bucketing import max_steps_for_latent_bucket
from mme_vla_suite.training.pretrain_loss import (
    composite_alignment_loss,
    make_stage1_modules,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--latent_bucket", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fsdp_devices", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--wy_matmul_dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--module_compute_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


def make_batch(config, batch_size, latent_bucket):
    cb = config.cosmo_bottleneck_memory
    temporal_factor = int(cb.temporal_factor)
    pool_h = int(getattr(cb, "pool_h", 0) or getattr(cb, "pool_hw", 0) or 1)
    pool_w = int(getattr(cb, "pool_w", 0) or getattr(cb, "pool_hw", 0) or 1)
    tokens_per_chunk = int(config.num_views) * pool_h * pool_w
    max_steps = max_steps_for_latent_bucket(
        latent_bucket,
        temporal_factor=temporal_factor,
        tokens_per_chunk=tokens_per_chunk,
        global_cap=int(cb.max_step_tokens),
    )
    recent_tokens = int(cb.max_recent_tokens)
    queries = int(cb.num_query_samples)
    patches = int(config.cosmo_decoder.align_num_patches)
    align_dim = int(config.cosmo_decoder.align_dim)
    action_dim = int(config.cosmo_decoder.action_dim)
    proprio_dim = int(config.cosmo_decoder.proprio_dim)

    batch = {
        "cosmo_latent_emb": np.zeros(
            (batch_size, latent_bucket, int(cb.cosmos_latent_dim)),
            dtype=np.float32,
        ),
        "cosmo_mem_mask": np.ones(
            (batch_size, latent_bucket),
            dtype=bool,
        ),
        "cosmo_recent_emb": np.zeros(
            (batch_size, recent_tokens, int(cb.siglip_dim)),
            dtype=np.float16,
        ),
        "cosmo_recent_mask": np.ones(
            (batch_size, recent_tokens),
            dtype=bool,
        ),
        "cosmo_action_emb": np.zeros(
            (batch_size, max_steps, int(cb.action_in_dim)),
            dtype=np.float32,
        ),
        "cosmo_action_mask": np.ones(
            (batch_size, max_steps),
            dtype=bool,
        ),
        "cosmo_proprio_emb": np.zeros(
            (batch_size, max_steps, int(cb.proprio_in_dim)),
            dtype=np.float32,
        ),
        "cosmo_proprio_mask": np.ones(
            (batch_size, max_steps),
            dtype=bool,
        ),
        "cosmo_lang_emb": np.zeros(
            (batch_size, int(cb.lang_emb_dim)),
            dtype=np.float32,
        ),
        "query_steps": np.zeros((batch_size, queries), dtype=np.int32),
        "query_mask": np.ones((batch_size, queries), dtype=bool),
        "align_target": np.zeros(
            (batch_size, queries, patches, align_dim),
            dtype=np.float16,
        ),
        "action_target": np.zeros(
            (batch_size, queries, action_dim),
            dtype=np.float32,
        ),
        "proprio_target": np.zeros(
            (batch_size, queries, proprio_dim),
            dtype=np.float32,
        ),
        "reward_target": np.zeros((batch_size,), dtype=np.float32),
    }
    return batch, max_steps


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    cb = config.cosmo_bottleneck_memory
    cb.delta_full_scan_mode = "chunkwise_wy"
    cb.delta_wy_matmul_dtype = args.wy_matmul_dtype
    module_compute_dtype = {
        "float32": jax.numpy.float32,
        "bfloat16": jax.numpy.bfloat16,
    }[args.module_compute_dtype]
    modules = make_stage1_modules(
        config,
        seed=0,
        dtype=module_compute_dtype,
    )
    graphdef, state = nnx.split((modules["bottleneck"], modules["decoder"]))
    tx = build_optimizer(
        lr=1e-4,
        weight_decay=1e-4,
        warmup_steps=1000,
        num_steps=160000,
    )
    opt_state = tx.init(state)
    weights = {"align": 1.0, "action": 0.5, "proprio": 0.5, "reward": 0.5}

    def loss_fn(state, batch):
        bottleneck, decoder = nnx.merge(graphdef, state)
        return composite_alignment_loss(
            {"bottleneck": bottleneck, "decoder": decoder},
            batch,
            weights,
        )

    @jax.jit
    def train_step(state, opt_state, batch):
        (total, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state,
            batch,
        )
        updates, opt_state = tx.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, total, parts

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    host_batch, max_steps = make_batch(
        config,
        args.batch_size,
        args.latent_bucket,
    )
    host_batch_gib = sum(value.nbytes for value in host_batch.values()) / (1024**3)

    with sharding.set_mesh(mesh):
        batch = jax.device_put(host_batch, data_sharding)
        compile_start = time.perf_counter()
        state, opt_state, total, parts = train_step(state, opt_state, batch)
        jax.block_until_ready(total)
        compile_seconds = time.perf_counter() - compile_start

        for _ in range(args.warmup):
            state, opt_state, total, parts = train_step(state, opt_state, batch)
            jax.block_until_ready(total)

        timings = []
        for _ in range(args.steps):
            start = time.perf_counter()
            state, opt_state, total, parts = train_step(state, opt_state, batch)
            jax.block_until_ready(total)
            timings.append(time.perf_counter() - start)

    payload = {
        "latent_bucket": args.latent_bucket,
        "max_step_tokens": max_steps,
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "wy_matmul_dtype": args.wy_matmul_dtype,
        "module_compute_dtype": args.module_compute_dtype,
        "host_batch_gib": host_batch_gib,
        "compile_seconds": compile_seconds,
        "seconds_per_step_median": statistics.median(timings),
        "seconds_per_step_min": min(timings),
        "seconds_per_step_max": max(timings),
        "loss": float(total),
        "loss_parts": {key: float(value) for key, value in parts.items()},
        "devices": [str(device) for device in jax.devices()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
