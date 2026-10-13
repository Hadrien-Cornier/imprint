"""Recent visual feature selection for offline Stage-1 stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RecentFeatureSpec:
    filename: str
    expected_dim: int | None


RECENT_FEATURE_SPECS = {
    "siglip": RecentFeatureSpec("recent_siglip.npy", expected_dim=None),
    "dinov3": RecentFeatureSpec("dinov3.npy", expected_dim=768),
}


def _feature_spec(source: str) -> RecentFeatureSpec:
    try:
        return RECENT_FEATURE_SPECS[source]
    except KeyError as exc:
        supported = ", ".join(sorted(RECENT_FEATURE_SPECS))
        raise ValueError(
            f"Unsupported recent feature source {source!r}; expected one of: {supported}."
        ) from exc


def recent_feature_filename(source: str) -> str:
    return _feature_spec(source).filename


def load_recent_feature_array(
    episode_dir: str | Path,
    source: str,
    *,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    spec = _feature_spec(source)
    path = Path(episode_dir) / spec.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Recent feature source {source!r} requires {path}; the file is missing."
        )

    features = np.load(path, mmap_mode=mmap_mode)
    if features.ndim < 1:
        raise ValueError(f"Recent feature file {path} must have a feature dimension.")
    if spec.expected_dim is not None and features.shape[-1] != spec.expected_dim:
        raise ValueError(
            f"Recent feature file {path} for source {source!r} has dim "
            f"{features.shape[-1]}; expected {spec.expected_dim}."
        )
    return features


def validate_recent_feature_store(
    store: str | Path,
    source: str,
    *,
    expected_dim: int | None = None,
    expected_tokens: int | None = None,
) -> None:
    features_root = Path(store) / "features"
    episode_dirs = sorted(path for path in features_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {features_root}.")
    for episode_dir in episode_dirs:
        features = load_recent_feature_array(episode_dir, source)
        if expected_dim is not None and features.shape[-1] != expected_dim:
            raise ValueError(
                f"Recent feature file {episode_dir / recent_feature_filename(source)} "
                f"has dim {features.shape[-1]}, but the configured input dim "
                f"{expected_dim} does not match."
            )
        if expected_tokens is not None:
            feature_tokens = int(np.prod(features.shape[1:-1]))
            if feature_tokens != expected_tokens:
                raise ValueError(
                    f"Recent feature file {episode_dir / recent_feature_filename(source)} "
                    f"has {feature_tokens} reconstruction tokens per frame, but the "
                    f"configured reconstruction tokens {expected_tokens} do not match."
                )
