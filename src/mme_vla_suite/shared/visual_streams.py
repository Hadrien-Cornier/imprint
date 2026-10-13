"""Resolve validated visual feature streams for RMBench memory training."""

import json
from pathlib import Path

import numpy as np


def visual_feature_filename(stream: str) -> str:
    if stream == "siglip":
        return "recent_siglip.npy"
    if stream == "dinov3":
        return "dinov3.npy"
    raise ValueError(f"unsupported visual feature stream {stream!r}")


def load_dino_manifest(store_dir: str | Path) -> dict[str, int]:
    """Validate the DINOv3 replacement stream geometry."""
    path = Path(store_dir) / "meta" / "dino_store_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        manifest = json.load(handle)

    extractor = manifest.get("extractor", {})
    replacements = manifest.get("replacement_streams", {})
    recent = replacements.get("recent", {})
    alignment = replacements.get("alignment", {})
    model_id = str(extractor.get("model_id", ""))
    if "dinov3" not in model_id.lower():
        raise ValueError(f"{path}: expected DINOv3 extractor model_id, got {model_id!r}")

    expected = {"feature_dim": 768, "tokens_per_view": 64}
    for key, value in expected.items():
        if extractor.get(key) != value:
            raise ValueError(
                f"{path}: expected extractor.{key}={value!r}, "
                f"got {extractor.get(key)!r}"
            )
    if recent.get("file") != "dinov3.npy" or alignment.get("file") != "dinov3.npy":
        raise ValueError(f"{path}: DINOv3 replacements must use dinov3.npy")
    if recent.get("shape") != ["T", "V", 64, 768]:
        raise ValueError(f"{path}: invalid DINOv3 recent shape {recent.get('shape')!r}")
    if alignment.get("shape") != ["T", "V*P", 768]:
        raise ValueError(
            f"{path}: invalid DINOv3 alignment shape {alignment.get('shape')!r}"
        )
    return {**expected, "num_views": 3}


def load_visual_stream(
    episode_dir,
    *,
    stream,
    recent_idx,
    align_idx,
    features=None,
    recent_dtype=np.float32,
    alignment_dtype=np.float32,
):
    """Load aligned recent and reconstruction tensors from one DINO stream."""
    if stream != "dinov3":
        raise ValueError(f"unsupported visual stream {stream!r}")

    path = Path(episode_dir) / visual_feature_filename(stream)
    if not path.is_file():
        raise FileNotFoundError(path)
    features = np.load(path, mmap_mode="r") if features is None else features
    if features.ndim != 4:
        raise ValueError(f"{path} must be rank-4 (T, V, P, D), got {features.shape}")

    recent_idx = np.asarray(recent_idx, dtype=np.int64)
    align_idx = np.asarray(align_idx, dtype=np.int64)
    timesteps, views, patches, dim = features.shape
    for name, indices in (("recent_idx", recent_idx), ("align_idx", align_idx)):
        if indices.size and (indices.min() < 0 or indices.max() >= timesteps):
            raise IndexError(f"{name} outside [0, {timesteps}) for {path}")

    recent = np.ascontiguousarray(features[recent_idx]).reshape(-1, dim)
    alignment = np.ascontiguousarray(features[align_idx]).reshape(
        len(align_idx), views * patches, dim
    )
    return (
        recent.astype(recent_dtype, copy=False),
        alignment.astype(alignment_dtype, copy=False),
    )
