"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue
"""

import bisect
import json
import logging
from pathlib import Path

import jax
import numpy as np
from omegaconf import DictConfig
import torch

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.shared.sample_locations import sample_location
from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.dataset import resolve_feature_dir
from mme_vla_suite.training.length_bucketing import DEFAULT_LATENT_BUCKETS
from mme_vla_suite.training.length_bucketing import build_length_bucketed_epoch_batches
from mme_vla_suite.training.length_bucketing import max_steps_for_latent_bucket
from mme_vla_suite.training.length_bucketing import select_latent_bucket
from openpi.models import model as _model
import openpi.training.config as _config
from openpi.training.data_loader import DataLoader
from openpi.training.data_loader import TorchDataLoader
from openpi.training.data_loader import transform_dataset


def load_stage2_variant_frame_counts(
    dataset_path: str,
    history_config: DictConfig,
    *,
    expected_samples: int,
) -> tuple[list, np.ndarray]:
    """Load exact per-sample causal lengths without opening every policy pickle."""
    samples_path = Path(dataset_path) / "meta" / "samples.json"
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"length-bucketed Stage-2 training requires {samples_path}"
        )
    with samples_path.open() as handle:
        samples = json.load(handle)
    if len(samples) != int(expected_samples):
        raise ValueError(
            "Stage-2 sample index does not match the policy dataset: "
            f"{len(samples)} != {expected_samples}"
        )

    cb = history_config.cosmo_bottleneck_memory
    n_variants = int(getattr(cb, "n_variants", 1))
    feature_dir = resolve_feature_dir(dataset_path, history_config)
    frame_cache: dict[int, tuple[list[int], ...]] = {}
    frame_counts = np.empty((len(samples),), dtype=np.int32)
    for sample_index, sample in enumerate(samples):
        episode_index, step_index = sample_location(sample)
        frame_lists = frame_cache.get(episode_index)
        if frame_lists is None:
            index_path = feature_dir / f"episode_{episode_index}" / "variant_index.json"
            with index_path.open() as handle:
                variant_index = json.load(handle)
            frame_lists = tuple(
                variant_index[f"v{variant}"]["frame_idx"]
                for variant in range(n_variants)
            )
            frame_cache[episode_index] = frame_lists
        frame_counts[sample_index] = max(
            bisect.bisect_right(frame_indices, step_index)
            for frame_indices in frame_lists
        )

    logging.info(
        "Stage-2 exact length index: samples=%d episodes=%d max_frames=%d",
        len(samples),
        len(frame_cache),
        int(frame_counts.max()) if len(frame_counts) else 0,
    )
    return samples, frame_counts


class LengthBucketedSampler(torch.utils.data.Sampler):
    """Yield shuffled, homogeneous length buckets on every epoch."""

    def __init__(
        self,
        samples,
        sample_frame_counts: np.ndarray,
        *,
        batch_size: int,
        buckets=DEFAULT_LATENT_BUCKETS,
        temporal_factor: int,
        tokens_per_chunk: int,
        seed: int,
    ):
        self._samples = samples
        self._sample_frame_counts = sample_frame_counts
        self._batch_size = int(batch_size)
        self._buckets = tuple(int(bucket) for bucket in buckets)
        self._temporal_factor = int(temporal_factor)
        self._tokens_per_chunk = int(tokens_per_chunk)
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self):
        return (len(self._samples) // self._batch_size) * self._batch_size

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        batches = build_length_bucketed_epoch_batches(
            self._samples,
            batch_size=self._batch_size,
            buckets=self._buckets,
            temporal_factor=self._temporal_factor,
            tokens_per_chunk=self._tokens_per_chunk,
            rng=rng,
            sample_frame_counts=self._sample_frame_counts,
        )
        for _, indices in batches:
            yield from (int(index) for index in indices)


def crop_stage2_memory_batch(
    batch,
    *,
    buckets=DEFAULT_LATENT_BUCKETS,
    temporal_factor: int,
    tokens_per_chunk: int,
    max_step_tokens: int,
):
    """Crop a fixed-cap Stage-2 batch to its smallest exact static-shape tier."""
    required = (
        "cosmo_latent_emb",
        "cosmo_mem_mask",
        "cosmo_action_emb",
        "cosmo_action_mask",
        "cosmo_proprio_emb",
        "cosmo_proprio_mask",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"length-bucketed Stage-2 batch is missing {missing}")

    max_valid_latents = max(
        1,
        int(np.max(np.sum(np.asarray(batch["cosmo_mem_mask"]), axis=1))),
    )
    latent_bucket = select_latent_bucket(max_valid_latents, buckets)
    step_cap = max_steps_for_latent_bucket(
        latent_bucket,
        temporal_factor=temporal_factor,
        tokens_per_chunk=tokens_per_chunk,
        global_cap=max_step_tokens,
    )
    for mask_key in ("cosmo_action_mask", "cosmo_proprio_mask"):
        max_valid_steps = int(np.max(np.sum(np.asarray(batch[mask_key]), axis=1)))
        if max_valid_steps > step_cap:
            raise ValueError(
                f"{mask_key} needs {max_valid_steps} steps, but latent bucket "
                f"{latent_bucket} permits {step_cap}"
            )

    cropped = dict(batch)
    for key in ("cosmo_latent_emb", "cosmo_mem_mask"):
        cropped[key] = batch[key][:, :latent_bucket]
    for key in (
        "cosmo_action_emb",
        "cosmo_action_mask",
        "cosmo_proprio_emb",
        "cosmo_proprio_mask",
    ):
        cropped[key] = batch[key][:, :step_cap]
    return cropped


class _Stage2MemoryBatchCropper:
    def __init__(
        self,
        *,
        buckets,
        temporal_factor: int,
        tokens_per_chunk: int,
        max_step_tokens: int,
    ):
        self._buckets = tuple(int(bucket) for bucket in buckets)
        self._temporal_factor = int(temporal_factor)
        self._tokens_per_chunk = int(tokens_per_chunk)
        self._max_step_tokens = int(max_step_tokens)
        self._logged_shapes = set()

    def __call__(self, batch):
        cropped = crop_stage2_memory_batch(
            batch,
            buckets=self._buckets,
            temporal_factor=self._temporal_factor,
            tokens_per_chunk=self._tokens_per_chunk,
            max_step_tokens=self._max_step_tokens,
        )
        shape = (
            int(cropped["cosmo_latent_emb"].shape[1]),
            int(cropped["cosmo_action_emb"].shape[1]),
        )
        if shape not in self._logged_shapes:
            logging.info(
                "Stage-2 memory bucket activated: latent=%d action/proprio=%d",
                *shape,
            )
            self._logged_shapes.add(shape)
        return cropped


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._total_samples = len(data_loader._data_loader.dataset)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield HistAugObservation.from_dict(batch), batch["actions"]


def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: str | DictConfig | None,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    late_episode_loss_weight: float = 1.0,
    late_episode_loss_fraction: float = 0.5,
    key_event_manifest_path: str | None = None,
    key_event_loss_weight: float = 1.0,
    key_event_window_radius: int = 5,
    length_bucketed_batches: bool = False,
) -> DataLoader[tuple[HistAugObservation, _model.Actions]]:
    history_config = get_history_config(history_config)

    dataset = RoboMMEDataset(
        dataset_path=dataset_path,
        data_config=data_config,
        history_config=history_config,
        action_horizon=action_horizon,
        late_episode_loss_weight_value=late_episode_loss_weight,
        late_episode_loss_fraction=late_episode_loss_fraction,
        key_event_manifest_path=key_event_manifest_path,
        key_event_loss_weight=key_event_loss_weight,
        key_event_window_radius=key_event_window_radius,
    )

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")

    sampler = None
    batch_transform = None
    if length_bucketed_batches:
        cb = history_config.cosmo_bottleneck_memory
        if str(getattr(cb, "latent_source", "")) != "offline":
            raise ValueError("Stage-2 length buckets require offline memory latents")
        pool_h = int(getattr(cb, "pool_h", 0) or getattr(cb, "pool_hw", 0) or 1)
        pool_w = int(getattr(cb, "pool_w", 0) or getattr(cb, "pool_hw", 0) or 1)
        temporal_factor = int(getattr(cb, "temporal_factor", 1))
        tokens_per_chunk = int(history_config.num_views) * pool_h * pool_w
        max_step_tokens = int(getattr(cb, "max_step_tokens", 0) or 800)
        latent_cap = int(getattr(cb, "latent_cap", 0) or 2048)
        buckets = tuple(
            bucket for bucket in DEFAULT_LATENT_BUCKETS if bucket <= latent_cap
        )
        if not buckets or buckets[-1] != latent_cap:
            buckets = (*buckets, latent_cap)
        samples, frame_counts = load_stage2_variant_frame_counts(
            dataset_path,
            history_config,
            expected_samples=len(dataset),
        )
        sampler = LengthBucketedSampler(
            samples,
            frame_counts,
            batch_size=local_batch_size,
            buckets=buckets,
            temporal_factor=temporal_factor,
            tokens_per_chunk=tokens_per_chunk,
            seed=seed,
        )
        batch_transform = _Stage2MemoryBatchCropper(
            buckets=buckets,
            temporal_factor=temporal_factor,
            tokens_per_chunk=tokens_per_chunk,
            max_step_tokens=max_step_tokens,
        )
        logging.info(
            "Stage-2 length buckets enabled: latent=%s temporal_factor=%d "
            "tokens_per_chunk=%d max_steps=%d",
            buckets,
            temporal_factor,
            tokens_per_chunk,
            max_step_tokens,
        )

    dataset = transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        batch_transform=batch_transform,
        seed=seed,
        framework="jax",
    )

    return DataLoaderImpl(data_config, data_loader)
