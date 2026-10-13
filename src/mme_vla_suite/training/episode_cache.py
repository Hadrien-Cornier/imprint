"""Bounded per-process cache for read-only stage-2 episode stores."""

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from mme_vla_suite.shared.visual_streams import visual_feature_filename

def _freeze_cached_array(value: np.ndarray) -> np.ndarray:
    """Return an array that callers cannot make writeable again."""
    if isinstance(value, np.memmap):
        value.setflags(write=False)
        return value
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _freeze_cached_value(value):
    """Recursively make cached metadata safe to share with callers."""
    if isinstance(value, np.ndarray):
        return _freeze_cached_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_cached_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_cached_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_cached_value(item) for item in value)
    return value


class EpisodeStoreEntry:
    """Read-only resources for one on-disk episode store."""

    def __init__(self, episode_dir: Path, visual_feature_stream: str = "siglip") -> None:
        self.episode_dir = episode_dir
        with (episode_dir / "chunk_index.json").open() as file:
            self.chunk_index = _freeze_cached_value(json.load(file))
        with (episode_dir / "variant_index.json").open() as file:
            self.variant_index = _freeze_cached_value(json.load(file))
        self.action_proprio = _freeze_cached_value(
            np.load(episode_dir / "action_proprio.npy", allow_pickle=True).item()
        )
        self.recent_visual = _freeze_cached_value(
            np.load(
                episode_dir / visual_feature_filename(visual_feature_stream), mmap_mode="r"
            )
        )
        # Preserve the attribute used by existing SigLIP callers.
        self.recent_siglip = self.recent_visual
        self._latents: dict[int, np.memmap] = {}

    def get_latents(self, variant: int) -> np.memmap:
        """Return the lazily opened read-only latent memmap for ``variant``."""
        variant = int(variant)
        latents = self._latents.get(variant)
        if latents is None:
            latents = _freeze_cached_value(
                np.load(
                    self.episode_dir / f"latents_v{variant}.npy", mmap_mode="r"
                )
            )
            self._latents[variant] = latents
        return latents


class EpisodeStoreCache:
    """Process-local LRU cache of read-only episode store entries."""

    def __init__(self, capacity: int = 8, visual_feature_stream: str = "siglip") -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.visual_feature_stream = visual_feature_stream
        self._entries: OrderedDict[Path, EpisodeStoreEntry] = OrderedDict()

    def get(self, episode_dir: str | Path) -> EpisodeStoreEntry:
        """Return an entry for ``episode_dir``, refreshing its LRU position."""
        path = Path(episode_dir)
        entry = self._entries.get(path)
        if entry is not None:
            self._entries.move_to_end(path)
            return entry

        entry = EpisodeStoreEntry(path, self.visual_feature_stream)
        if len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)
        self._entries[path] = entry
        return entry
