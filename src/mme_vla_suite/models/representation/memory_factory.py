"""Construct a memory encoder from the history configuration."""

import jax.numpy as jnp

LEGACY_ENCODER = "perceiver"
GATED_DELTA_ENCODER = "gated_delta_hybrid"


def memory_encoder_type(config):
    cb = config.cosmo_bottleneck_memory
    return str(getattr(cb, "memory_encoder_type", LEGACY_ENCODER))


def memory_compute_dtype(config, fallback):
    """Resolve an opt-in memory-only compute dtype while preserving FP32 params."""
    cb = config.cosmo_bottleneck_memory
    dtype_name = getattr(cb, "memory_compute_dtype", None)
    if dtype_name is None:
        return fallback
    try:
        return {
            "float32": jnp.float32,
            "bfloat16": jnp.bfloat16,
        }[str(dtype_name)]
    except KeyError as error:
        raise ValueError(
            "memory_compute_dtype must be 'float32' or 'bfloat16', "
            f"got {dtype_name!r}"
        ) from error


def create_cosmo_memory_encoder(config, rngs, dtype=jnp.float32):
    encoder_type = memory_encoder_type(config)
    if encoder_type == LEGACY_ENCODER:
        from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory

        return CosmoBottleneckMemory(config=config, rngs=rngs, dtype=dtype)
    if encoder_type == GATED_DELTA_ENCODER:
        from mme_vla_suite.models.representation.gated_delta_mem import GatedDeltaBottleneckMemory

        return GatedDeltaBottleneckMemory(
            config=config,
            rngs=rngs,
            dtype=memory_compute_dtype(config, dtype),
        )
    raise ValueError(
        f"unsupported cosmo memory_encoder_type {encoder_type!r}; "
        f"expected {LEGACY_ENCODER!r} or {GATED_DELTA_ENCODER!r}"
    )
