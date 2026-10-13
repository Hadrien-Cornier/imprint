import numpy as np

from mme_vla_suite.training.dataloader import LengthBucketedSampler
from mme_vla_suite.training.dataloader import crop_stage2_memory_batch
from mme_vla_suite.training.length_bucketing import DEFAULT_LATENT_BUCKETS
from mme_vla_suite.training.length_bucketing import build_length_bucketed_epoch_batches
from mme_vla_suite.training.length_bucketing import max_steps_for_latent_bucket
from mme_vla_suite.training.length_bucketing import required_latent_tokens
from mme_vla_suite.training.length_bucketing import select_latent_bucket


def test_bucket_boundaries_and_step_caps_are_exact():
    assert required_latent_tokens(
        0, temporal_factor=4, tokens_per_chunk=12
    ) == 1
    assert required_latent_tokens(
        3, temporal_factor=4, tokens_per_chunk=12
    ) == 12
    assert select_latent_bucket(252, DEFAULT_LATENT_BUCKETS) == 256
    assert select_latent_bucket(264, DEFAULT_LATENT_BUCKETS) == 512
    assert max_steps_for_latent_bucket(
        256,
        temporal_factor=4,
        tokens_per_chunk=12,
        global_cap=1244,
    ) == 87
    assert max_steps_for_latent_bucket(
        4096,
        temporal_factor=4,
        tokens_per_chunk=12,
        global_cap=1244,
    ) == 1244


def test_epoch_batches_never_assign_a_sample_to_a_short_bucket():
    steps = [0, 3, 4, 86, 87, 170, 342, 682, 900, 1000, 1100, 1200, 1243]
    samples = [
        {"episode_index": index // 3, "step_index": step}
        for index, step in enumerate(steps)
    ]
    batches = build_length_bucketed_epoch_batches(
        samples,
        batch_size=4,
        buckets=DEFAULT_LATENT_BUCKETS,
        temporal_factor=4,
        tokens_per_chunk=12,
        rng=np.random.default_rng(9),
    )

    seen = []
    for bucket, indices in batches:
        assert len(indices) == 4
        step_cap = max_steps_for_latent_bucket(
            bucket,
            temporal_factor=4,
            tokens_per_chunk=12,
            global_cap=1244,
        )
        for index in indices:
            seen.append(int(index))
            required = required_latent_tokens(
                steps[index],
                temporal_factor=4,
                tokens_per_chunk=12,
            )
            assert required <= bucket
            assert min(steps[index] + 1, 1244) <= step_cap

    assert len(seen) == 12
    assert len(set(seen)) == len(seen)


def test_variant_frame_counts_avoid_raw_timeline_overestimation():
    samples = [{"episode_index": 0, "step_index": 2435}]
    batches = build_length_bucketed_epoch_batches(
        samples,
        batch_size=1,
        buckets=DEFAULT_LATENT_BUCKETS,
        temporal_factor=4,
        tokens_per_chunk=12,
        rng=np.random.default_rng(3),
        sample_frame_counts=[1244],
    )

    assert len(batches) == 1
    bucket, indices = batches[0]
    assert bucket == 4096
    assert indices.tolist() == [0]
    assert required_latent_tokens(
        1243,
        temporal_factor=4,
        tokens_per_chunk=12,
    ) == 3732


def test_stage2_crop_synchronizes_latent_action_and_proprio_tiers():
    batch_size = 2
    latent_mask = np.zeros((batch_size, 4096), dtype=bool)
    latent_mask[0, :700] = True
    latent_mask[1, :620] = True
    step_mask = np.zeros((batch_size, 1244), dtype=bool)
    step_mask[0, :235] = True
    step_mask[1, :211] = True
    batch = {
        "cosmo_latent_emb": np.zeros((batch_size, 4096, 48), np.float32),
        "cosmo_mem_mask": latent_mask,
        "cosmo_action_emb": np.zeros((batch_size, 1244, 16), np.float32),
        "cosmo_action_mask": step_mask,
        "cosmo_proprio_emb": np.zeros((batch_size, 1244, 16), np.float32),
        "cosmo_proprio_mask": step_mask.copy(),
        "actions": np.zeros((batch_size, 50, 16), np.float32),
    }

    cropped = crop_stage2_memory_batch(
        batch,
        temporal_factor=4,
        tokens_per_chunk=12,
        max_step_tokens=1244,
    )

    assert cropped["cosmo_latent_emb"].shape == (batch_size, 1024, 48)
    assert cropped["cosmo_mem_mask"].shape == (batch_size, 1024)
    assert cropped["cosmo_action_emb"].shape == (batch_size, 343, 16)
    assert cropped["cosmo_proprio_emb"].shape == (batch_size, 343, 16)
    assert cropped["actions"].shape == (batch_size, 50, 16)
    assert int(cropped["cosmo_mem_mask"].sum()) == 1320
    assert int(cropped["cosmo_action_mask"].sum()) == 446


def test_stage2_sampler_keeps_full_batches_in_length_tiers():
    samples = [
        {"episode_index": index // 2, "step_index": index}
        for index in range(8)
    ]
    frame_counts = np.asarray([20, 30, 50, 60, 100, 110, 200, 210])
    sampler = LengthBucketedSampler(
        samples,
        frame_counts,
        batch_size=2,
        temporal_factor=4,
        tokens_per_chunk=12,
        seed=7,
    )

    indices = list(sampler)
    assert len(indices) == 8
    assert sorted(indices) == list(range(8))
    for start in range(0, len(indices), 2):
        batch_indices = indices[start : start + 2]
        buckets = {
            select_latent_bucket(
                required_latent_tokens(
                    int(frame_counts[index]) - 1,
                    temporal_factor=4,
                    tokens_per_chunk=12,
                ),
                DEFAULT_LATENT_BUCKETS,
            )
            for index in batch_indices
        }
        assert len(buckets) == 1
