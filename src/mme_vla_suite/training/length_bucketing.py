"""Exact static-shape batches for offline Stage-1 memory training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mme_vla_suite.shared.sample_locations import sample_location


DEFAULT_LATENT_BUCKETS = (256, 512, 1024, 2048, 4096)


def required_latent_tokens(
    step_index: int,
    *,
    temporal_factor: int,
    tokens_per_chunk: int,
) -> int:
    """Return a conservative, exact upper bound for any causal variant."""
    temporal_factor = int(temporal_factor)
    tokens_per_chunk = int(tokens_per_chunk)
    if temporal_factor <= 0 or tokens_per_chunk <= 0:
        raise ValueError("temporal_factor and tokens_per_chunk must be positive")
    available_frames = int(step_index) + 1
    finalized_chunks = max(0, available_frames // temporal_factor)
    return max(1, finalized_chunks * tokens_per_chunk)


def select_latent_bucket(required_tokens: int, buckets: Sequence[int]) -> int:
    normalized = tuple(sorted({int(bucket) for bucket in buckets}))
    if not normalized or normalized[0] <= 0:
        raise ValueError("latent buckets must contain positive integers")
    for bucket in normalized:
        if int(required_tokens) <= bucket:
            return bucket
    raise ValueError(
        f"required latent length {required_tokens} exceeds largest bucket "
        f"{normalized[-1]}"
    )


def max_steps_for_latent_bucket(
    latent_bucket: int,
    *,
    temporal_factor: int,
    tokens_per_chunk: int,
    global_cap: int,
) -> int:
    """Return the smallest step cap that cannot truncate this latent bucket."""
    latent_bucket = int(latent_bucket)
    temporal_factor = int(temporal_factor)
    tokens_per_chunk = int(tokens_per_chunk)
    global_cap = int(global_cap)
    if min(latent_bucket, temporal_factor, tokens_per_chunk, global_cap) <= 0:
        raise ValueError("bucket geometry and global cap must be positive")
    finalized_chunks = latent_bucket // tokens_per_chunk
    exact_cap = finalized_chunks * temporal_factor + temporal_factor - 1
    return min(global_cap, exact_cap)


def build_length_bucketed_epoch_batches(
    samples,
    *,
    batch_size: int,
    buckets: Sequence[int],
    temporal_factor: int,
    tokens_per_chunk: int,
    rng: np.random.Generator,
    sample_frame_counts: Sequence[int] | None = None,
):
    """Build one shuffled epoch while dropping only the global final remainder.

    Full batches stay within a single bucket. Per-bucket remainders are mixed
    together and assigned the largest required bucket in that mixed batch.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    normalized_buckets = tuple(sorted({int(bucket) for bucket in buckets}))
    if sample_frame_counts is not None and len(sample_frame_counts) != len(samples):
        raise ValueError(
            "sample_frame_counts must have one entry for every sample"
        )
    grouped = {bucket: [] for bucket in normalized_buckets}
    for sample_index, sample in enumerate(samples):
        if sample_frame_counts is None:
            _, step_index = sample_location(sample)
        else:
            step_index = int(sample_frame_counts[sample_index]) - 1
        required = required_latent_tokens(
            step_index,
            temporal_factor=temporal_factor,
            tokens_per_chunk=tokens_per_chunk,
        )
        bucket = select_latent_bucket(required, normalized_buckets)
        grouped[bucket].append(sample_index)

    batches = []
    leftovers = []
    for bucket in normalized_buckets:
        indices = np.asarray(grouped[bucket], dtype=np.int64)
        rng.shuffle(indices)
        full_count = (len(indices) // batch_size) * batch_size
        for start in range(0, full_count, batch_size):
            batches.append((bucket, indices[start : start + batch_size].copy()))
        leftovers.extend((int(index), bucket) for index in indices[full_count:])

    rng.shuffle(leftovers)
    leftover_count = (len(leftovers) // batch_size) * batch_size
    for start in range(0, leftover_count, batch_size):
        group = leftovers[start : start + batch_size]
        bucket = max(item[1] for item in group)
        indices = np.asarray([item[0] for item in group], dtype=np.int64)
        batches.append((bucket, indices))

    rng.shuffle(batches)
    return batches
