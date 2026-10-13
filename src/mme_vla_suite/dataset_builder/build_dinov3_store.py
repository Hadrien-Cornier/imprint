#!/usr/bin/env python3
"""Build an immutable DINOv3 visual-stream replacement store.

The source Wan store is never modified. All non-visual streams are copied or
hardlinked into ``--out-store``. Per episode, this builder replaces:

  recent_siglip.npy  -> dinov3.npy, exposed as (T, V, P, D)
  align_prefix.npy   -> dinov3.npy, exposed as (T, V*P, D)

One canonical tensor avoids storing the identical DINO targets twice. The
generated manifest records both logical stream views and the exact extractor
configuration. Training/serving loaders must read this manifest rather than
assuming the legacy filenames or 2048-wide feature vectors.

The build is incomplete until the resulting derived store is validated and
published to the versioned Hugging Face dataset release.
"""

import argparse
import datetime as dt
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = 1
FEATURE_FILE = "dinov3.npy"
LEGACY_VISUAL_FILES = frozenset({"recent_siglip.npy", "align_prefix.npy"})


def validate_pool_geometry(image_size, patch_size, pool_grid):
    """Return the native patch grid when it exactly supports local pooling."""
    image_size = int(image_size)
    patch_size = int(patch_size)
    ph, pw = (int(pool_grid[0]), int(pool_grid[1]))
    if image_size < 1 or patch_size < 1 or ph < 1 or pw < 1:
        raise ValueError("image_size, patch_size, and pool_grid dimensions must be positive")
    if image_size % patch_size:
        raise ValueError(
            f"image size {image_size} must be divisible by patch size {patch_size}"
        )
    grid = image_size // patch_size
    if grid % ph or grid % pw:
        raise ValueError(
            f"native patch grid {grid}x{grid} must divide requested pool grid {ph}x{pw}"
        )
    return (grid, grid)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def pool_dinov3_patches(hidden, num_register_tokens, patch_grid, pool_grid):
    """Drop DINO prefix tokens and local-average patch features.

    Args:
      hidden: ``(batch, 1 + registers + gh*gw, dim)`` NumPy array.
      num_register_tokens: Number of register tokens declared by the model.
      patch_grid: Native patch-grid height and width.
      pool_grid: Requested output-grid height and width.
    """
    hidden = np.asarray(hidden)
    if hidden.ndim != 3:
        raise ValueError(f"expected (B,N,D) hidden states, got {hidden.shape}")
    gh, gw = (int(patch_grid[0]), int(patch_grid[1]))
    ph, pw = (int(pool_grid[0]), int(pool_grid[1]))
    prefix = 1 + int(num_register_tokens)
    expected = prefix + gh * gw
    if hidden.shape[1] != expected:
        raise ValueError(
            f"DINO sequence length {hidden.shape[1]} != expected {expected} "
            f"(1 CLS + {num_register_tokens} registers + {gh}x{gw} patches)"
        )
    if gh % ph or gw % pw:
        raise ValueError(
            f"native patch grid {gh}x{gw} must divide requested pool grid {ph}x{pw}"
        )
    bh, bw = gh // ph, gw // pw
    patches = hidden[:, prefix:].reshape(hidden.shape[0], gh, gw, hidden.shape[-1])
    pooled = patches.reshape(hidden.shape[0], ph, bh, pw, bw, hidden.shape[-1]).mean(axis=(2, 4))
    return pooled.reshape(hidden.shape[0], ph * pw, hidden.shape[-1])


class DinoV3Extractor:
    """Frozen Hugging Face DINOv3 patch extractor with exact local-average pooling."""

    def __init__(
        self,
        model_id,
        revision=None,
        device="cuda",
        batch_size=24,
        pool_grid=(8, 8),
        image_size=256,
        torch_dtype="bfloat16",
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3 extraction requires torch and transformers in the build environment."
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"unknown torch dtype {torch_dtype!r}; choose {sorted(dtype_map)}")
        self._torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.pool_grid = (int(pool_grid[0]), int(pool_grid[1]))
        self.image_size = int(image_size)
        if self.batch_size < 1 or min(self.pool_grid) < 1 or self.image_size < 1:
            raise ValueError("batch_size, image_size, and pool_grid dimensions must be positive")

        load_kwargs = {"revision": revision} if revision else {}
        self.processor = AutoImageProcessor.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype_map[torch_dtype], **load_kwargs
        ).to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.revision = revision
        self.torch_dtype = torch_dtype

        config = self.model.config
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.patch_size = int(getattr(config, "patch_size", 0))
        self.feature_dim = int(getattr(config, "hidden_size", 0))
        if self.patch_size < 1 or self.feature_dim < 1:
            raise ValueError(
                "DINO model config must expose positive patch_size and hidden_size; "
                f"got patch_size={self.patch_size}, hidden_size={self.feature_dim}"
            )
        validate_pool_geometry(self.image_size, self.patch_size, self.pool_grid)

    @property
    def model_info(self):
        processor = self.processor
        return _jsonable(
            {
                "model_id": self.model_id,
                "revision": self.revision,
                "feature_dim": self.feature_dim,
                "tokens_per_view": self.pool_grid[0] * self.pool_grid[1],
                "pool_grid": list(self.pool_grid),
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "source_output": "last_hidden_state",
                "dropped_prefix_tokens": 1 + self.num_register_tokens,
                "pooling": "local_average",
                "target_normalization": "per_token_layer_norm_at_loss",
                "processor_class": type(processor).__name__,
                "processor": {
                    "size": getattr(processor, "size", None),
                    "crop_size": getattr(processor, "crop_size", None),
                    "image_mean": getattr(processor, "image_mean", None),
                    "image_std": getattr(processor, "image_std", None),
                    "do_resize": getattr(processor, "do_resize", None),
                    "do_center_crop": getattr(processor, "do_center_crop", None),
                    "do_rescale": getattr(processor, "do_rescale", None),
                    "do_normalize": getattr(processor, "do_normalize", None),
                },
                "torch_dtype": self.torch_dtype,
            }
        )

    def encode(self, frames):
        """Encode uint8 ``(T,V,H,W,3)`` RGB frames as pooled ``(T,V,P,D)`` features."""
        frames = np.asarray(frames)
        if frames.ndim != 5 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"expected uint8 (T,V,H,W,3) RGB frames, got {frames.shape} {frames.dtype}")
        t, v = frames.shape[:2]
        flat = frames.reshape((-1, *frames.shape[2:]))
        outputs = []
        torch = self._torch
        for start in range(0, len(flat), self.batch_size):
            batch = [image for image in flat[start:start + self.batch_size]]
            resize = {"height": self.image_size, "width": self.image_size}
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                size=resize,
                crop_size=resize,
            )
            pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
            ih, iw = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
            if ih % self.patch_size or iw % self.patch_size:
                raise ValueError(
                    f"processor output {ih}x{iw} is incompatible with patch size {self.patch_size}"
                )
            patch_grid = (ih // self.patch_size, iw // self.patch_size)
            with torch.inference_mode():
                hidden = self.model(pixel_values=pixel_values).last_hidden_state
            pooled = pool_dinov3_patches(
                hidden.float().cpu().numpy(),
                num_register_tokens=self.num_register_tokens,
                patch_grid=patch_grid,
                pool_grid=self.pool_grid,
            )
            outputs.append(pooled.astype(np.float16, copy=False))
        return np.concatenate(outputs, axis=0).reshape(
            t, v, self.pool_grid[0] * self.pool_grid[1], self.feature_dim
        )


def _copy_or_link(source, destination, copy_mode):
    source = Path(os.path.realpath(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _copy_source_without_legacy_visual_streams(source_store, output_store, copy_mode):
    """Materialize every source file except the two replaced visual streams."""
    source_store = Path(source_store)
    output_store = Path(output_store)
    for root, _, filenames in os.walk(source_store):
        root_path = Path(root)
        relative_root = root_path.relative_to(source_store)
        for name in filenames:
            relative = relative_root / name
            if (
                len(relative.parts) == 3
                and relative.parts[0] == "features"
                and relative.parts[1].startswith("episode_")
                and name in LEGACY_VISUAL_FILES
            ):
                continue
            _copy_or_link(root_path / name, output_store / relative, copy_mode)


def _episode_dirs(store):
    features = Path(store) / "features"
    if not features.is_dir():
        raise FileNotFoundError(f"source store has no features directory: {features}")
    dirs = [p for p in features.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    if not dirs:
        raise FileNotFoundError(f"source store has no episode_* feature directories: {features}")
    return sorted(dirs, key=lambda p: int(p.name.split("_", 1)[1]))


def _atomic_save(path, array):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _build_manifest(source_store, source_episodes, episode_lengths, feature_shape, extractor, copy_mode):
    views, patches, dim = feature_shape
    return {
        "format_version": FORMAT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_store": str(source_store),
        "episodes": len(source_episodes),
        "episode_lengths": episode_lengths,
        "extractor": _jsonable(extractor.model_info),
        "replacement_streams": {
            "recent": {
                "file": FEATURE_FILE,
                "shape": ["T", "V", patches, dim],
                "layout": "time_view_patch_feature",
            },
            "alignment": {
                "file": FEATURE_FILE,
                "shape": ["T", "V*P", dim],
                "reshape": ["T", "V*P", "D"],
                "layout": "view_major_flattened",
            },
        },
        "replaced_source_files": sorted(LEGACY_VISUAL_FILES),
        "copy_mode": copy_mode,
    }


def _write_manifest(output_store, manifest):
    metadata_dir = Path(output_store) / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    temporary = metadata_dir / ".dino_store_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_dir / "dino_store_manifest.json")


def build_derived_store(source_store, output_store, extractor, copy_mode="hardlink"):
    """Create a self-contained derived store with DINO replacing both visual streams."""
    source_store = Path(source_store).expanduser().resolve()
    output_store = Path(output_store).expanduser().resolve()
    if source_store == output_store:
        raise ValueError("source_store and output_store must be different directories")
    if not source_store.is_dir():
        raise FileNotFoundError(f"source store does not exist: {source_store}")
    if copy_mode not in ("hardlink", "copy"):
        raise ValueError("copy_mode must be 'hardlink' or 'copy'")
    if output_store.exists() and any(output_store.iterdir()):
        raise FileExistsError(f"output store must be empty: {output_store}")
    output_store.mkdir(parents=True, exist_ok=True)

    source_episodes = _episode_dirs(source_store)
    _copy_source_without_legacy_visual_streams(source_store, output_store, copy_mode)

    feature_shape = None
    episode_lengths = {}
    for source_episode in source_episodes:
        frames_path = source_episode / "frames.npy"
        if not frames_path.exists():
            raise FileNotFoundError(f"{source_episode} is missing frames.npy")
        frames = np.load(frames_path, mmap_mode="r")
        features = np.asarray(extractor.encode(frames))
        if features.ndim != 4:
            raise ValueError(f"{source_episode}: extractor returned {features.shape}, expected (T,V,P,D)")
        if features.shape[:2] != frames.shape[:2]:
            raise ValueError(
                f"{source_episode}: DINO shape {features.shape[:2]} does not match frame shape {frames.shape[:2]}"
            )
        if features.dtype not in (np.float16, np.float32):
            raise ValueError(f"{source_episode}: expected float16/float32 DINO features, got {features.dtype}")

        shape = tuple(int(x) for x in features.shape[1:])
        if feature_shape is None:
            feature_shape = shape
        elif shape != feature_shape:
            raise ValueError(f"{source_episode}: inconsistent DINO feature shape {shape}, expected {feature_shape}")

        destination_episode = output_store / "features" / source_episode.name
        destination_episode.mkdir(parents=True, exist_ok=True)
        _atomic_save(destination_episode / FEATURE_FILE, features)
        episode_lengths[source_episode.name] = int(features.shape[0])

    manifest = _build_manifest(
        source_store,
        source_episodes,
        episode_lengths,
        feature_shape,
        extractor,
        copy_mode,
    )
    _write_manifest(output_store, manifest)
    return manifest


def finalize_existing_derived_store(source_store, output_store, extractor, copy_mode="hardlink"):
    """Validate existing DINO tensors and atomically write their manifest.

    This recovery mode is intentionally non-destructive: it never encodes,
    copies, or rewrites feature arrays. It is only for an interrupted build
    where every ``dinov3.npy`` was already written but the manifest was not.
    """
    source_store = Path(source_store).expanduser().resolve()
    output_store = Path(output_store).expanduser().resolve()
    if source_store == output_store:
        raise ValueError("source_store and output_store must be different directories")
    if not source_store.is_dir():
        raise FileNotFoundError(f"source store does not exist: {source_store}")
    if not output_store.is_dir():
        raise FileNotFoundError(f"output store does not exist: {output_store}")

    source_episodes = _episode_dirs(source_store)
    output_episodes = {episode.name: episode for episode in _episode_dirs(output_store)}
    expected_names = {episode.name for episode in source_episodes}
    if set(output_episodes) != expected_names:
        missing = sorted(expected_names - set(output_episodes))
        extra = sorted(set(output_episodes) - expected_names)
        raise ValueError(
            f"output episodes do not match source; missing={missing[:5]} extra={extra[:5]}"
        )

    feature_shape = None
    episode_lengths = {}
    for source_episode in source_episodes:
        frames_path = source_episode / "frames.npy"
        dino_path = output_episodes[source_episode.name] / FEATURE_FILE
        if not frames_path.exists():
            raise FileNotFoundError(f"{source_episode} is missing frames.npy")
        if not dino_path.exists():
            raise FileNotFoundError(f"{dino_path} is missing")

        frames = np.load(frames_path, mmap_mode="r")
        features = np.load(dino_path, mmap_mode="r")
        if features.ndim != 4 or features.shape[:2] != frames.shape[:2]:
            raise ValueError(
                f"{dino_path}: DINO shape {features.shape} does not match "
                f"frame time/views {frames.shape[:2]}"
            )
        if features.dtype not in (np.float16, np.float32):
            raise ValueError(f"{dino_path}: expected float16/float32 DINO features, got {features.dtype}")

        shape = tuple(int(x) for x in features.shape[1:])
        if feature_shape is None:
            feature_shape = shape
        elif shape != feature_shape:
            raise ValueError(f"{dino_path}: inconsistent DINO feature shape {shape}, expected {feature_shape}")
        episode_lengths[source_episode.name] = int(features.shape[0])

    manifest = _build_manifest(
        source_store,
        source_episodes,
        episode_lengths,
        feature_shape,
        extractor,
        copy_mode,
    )
    _write_manifest(output_store, manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", required=True, help="Existing immutable Wan store.")
    parser.add_argument("--out-store", required=True, help="New empty derived-store directory.")
    parser.add_argument(
        "--model-id",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face DINOv3 model id; requires approved access.",
    )
    parser.add_argument("--revision", default=None, help="Optional pinned Hugging Face revision.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pool-size", type=int, default=8, help="Square output patch grid per view.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square DINO processor output; must support exact local patch pooling.",
    )
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hardlink immutable source files when possible; copied fallback remains self-contained.",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate an interrupted, fully encoded output store and write only its manifest.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extractor = DinoV3Extractor(
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        pool_grid=(args.pool_size, args.pool_size),
        image_size=args.image_size,
        torch_dtype=args.torch_dtype,
    )
    build = finalize_existing_derived_store if args.finalize_existing else build_derived_store
    manifest = build(
        source_store=args.source_store,
        output_store=args.out_store,
        extractor=extractor,
        copy_mode=args.copy_mode,
    )
    print(
        f"built {manifest['episodes']} episodes -> {args.out_store}; "
        f"DINO shape={manifest['replacement_streams']['recent']['shape']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
