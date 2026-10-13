"""Benchmark the recurrent and chunkwise-WY Gated DeltaNet kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from mme_vla_suite.models.representation.gated_delta_mem import (
    _chunkwise_wy_delta_scan,
    _recurrent_delta_scan,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("recurrent", "chunkwise_wy"), required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--length", type=int, default=6220)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--matmul_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    key = jax.random.key(17)
    keys = jax.random.split(key, 7)
    shape = (args.batch_size, args.length, args.heads, args.head_dim)
    matrix_shape = (
        args.batch_size,
        args.heads,
        args.head_dim,
        args.head_dim,
    )
    matrix = 0.01 * jax.random.normal(keys[0], matrix_shape, dtype=jnp.float32)
    q = jax.random.normal(keys[1], shape, dtype=jnp.float32)
    k = jax.random.normal(keys[2], shape, dtype=jnp.float32)
    q = q * jax.lax.rsqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(keys[3], shape, dtype=jnp.float32)
    beta = jax.nn.sigmoid(
        jax.random.normal(
            keys[4],
            (args.batch_size, args.length, args.heads),
            dtype=jnp.float32,
        )
        - 1.0
    )
    decay = jnp.exp(
        -jax.nn.softplus(
            jax.random.normal(
                keys[5],
                (args.batch_size, args.length, args.heads),
                dtype=jnp.float32,
            )
            - 4.0
        )
    )
    mask = jnp.ones((args.batch_size, args.length), dtype=jnp.bool_)
    matmul_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[args.matmul_dtype]

    def loss(scale, matrix, q, k, v, beta, decay, mask):
        scaled_q = q * scale
        scaled_k = k * scale
        scaled_v = v * scale
        if args.mode == "recurrent":
            final_matrix, output = _recurrent_delta_scan(
                matrix,
                scaled_q,
                scaled_k,
                scaled_v,
                beta,
                decay,
                mask,
                checkpoint_size=args.chunk_size,
            )
        else:
            final_matrix, output = _chunkwise_wy_delta_scan(
                matrix,
                scaled_q,
                scaled_k,
                scaled_v,
                beta,
                decay,
                mask,
                chunk_size=args.chunk_size,
                matmul_dtype=matmul_dtype,
            )
        return jnp.mean(jnp.square(final_matrix)) + jnp.mean(jnp.square(output))

    step = jax.jit(jax.value_and_grad(loss, argnums=0))
    arguments = (
        jnp.asarray(1.0, dtype=jnp.float32),
        matrix,
        q,
        k,
        v,
        beta,
        decay,
        mask,
    )
    compile_start = time.perf_counter()
    result = step(*arguments)
    jax.block_until_ready(result)
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(args.warmup):
        result = step(*arguments)
        jax.block_until_ready(result)

    timings = []
    for _ in range(args.steps):
        start = time.perf_counter()
        result = step(*arguments)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)

    payload = {
        "mode": args.mode,
        "matmul_dtype": args.matmul_dtype,
        "batch_size": args.batch_size,
        "length": args.length,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "chunk_size": args.chunk_size,
        "compile_seconds": compile_seconds,
        "seconds_per_step_median": statistics.median(timings),
        "seconds_per_step_min": min(timings),
        "seconds_per_step_max": max(timings),
        "loss": float(result[0]),
        "scale_gradient": float(result[1]),
        "devices": [str(device) for device in jax.devices()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
