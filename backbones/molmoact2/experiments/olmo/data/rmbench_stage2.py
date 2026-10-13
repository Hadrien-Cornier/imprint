"""RMBench compact Stage-2 source and Pi0.5 offline memory assembly."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np


RMBENCH_STAGE2_REPO_ID = "rmbench_stage2"
RMBENCH_STAGE2_DATA_ROOT_ENV = "RMBENCH_STAGE2_DATA_ROOT"
RMBENCH_STAGE1_FEATURE_ROOT_ENV = "RMBENCH_STAGE1_FEATURE_ROOT"


@dataclass(frozen=True)
class RMBenchMemoryLayout:
    latent_cap: int = 4096
    max_recent_tokens: int = 768
    max_step_tokens: int = 1244
    recent_dim: int = 768
    stream_dim: int = 16
    language_dim: int = 2048
    latent_dim: int = 48
    latent_pool_h: int | None = 2
    latent_pool_w: int | None = 2
    wan_h: int = 256
    wan_w: int = 256


def _fixed_pad(values: np.ndarray, length: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((length, width), dtype=np.float32)
    mask = np.zeros((length,), dtype=np.bool_)
    selected = values[-length:] if values.shape[0] > length else values
    if selected.shape[0]:
        padded[: selected.shape[0]] = selected
        mask[: selected.shape[0]] = True
    return padded, mask


def _pool_latents(latents: np.ndarray, layout: RMBenchMemoryLayout) -> np.ndarray:
    if (
        layout.latent_pool_h is None
        or layout.latent_pool_w is None
        or latents.shape[0] == 0
    ):
        return latents

    grid_h = layout.wan_h // 16
    grid_w = layout.wan_w // 16
    pool_h = layout.latent_pool_h
    pool_w = layout.latent_pool_w
    pooled_spatial = pool_h * pool_w
    if latents.shape[1] % pooled_spatial == 0 and latents.shape[1] < grid_h * grid_w:
        return latents
    if latents.shape[1] % (grid_h * grid_w):
        raise ValueError(
            "Offline latent tokens do not match the configured DINOv3 spatial grid: "
            f"shape={latents.shape}, grid={grid_h}x{grid_w}."
        )
    if grid_h % pool_h or grid_w % pool_w:
        raise ValueError(
            f"Configured latent pooling {pool_h}x{pool_w} must divide {grid_h}x{grid_w}."
        )

    n_chunks, _, latent_dim = latents.shape
    n_views = latents.shape[1] // (grid_h * grid_w)
    block_h = grid_h // pool_h
    block_w = grid_w // pool_w
    grid = latents.reshape(n_chunks, n_views, grid_h, grid_w, latent_dim)
    grid = grid.reshape(
        n_chunks,
        n_views,
        pool_h,
        block_h,
        pool_w,
        block_w,
        latent_dim,
    ).mean(axis=(3, 5))
    return grid.reshape(n_chunks, n_views * pooled_spatial, latent_dim)


class CompactRMBenchMemoryAssembler:
    """Reproduce Pi0.5's offline-variant memory stream from the feature store."""

    def __init__(
        self,
        *,
        feature_root: str | Path,
        layout: RMBenchMemoryLayout = RMBenchMemoryLayout(),
        n_variants: int = 8,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.layout = layout
        self.n_variants = int(n_variants)

    def gather(
        self,
        *,
        episode_index: int,
        step_index: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        episode_dir = self.feature_root / f"episode_{int(episode_index)}"
        with (episode_dir / "variant_index.json").open(encoding="utf-8") as f:
            variant_index = json.load(f)
        with (episode_dir / "chunk_index.json").open(encoding="utf-8") as f:
            chunk_index = json.load(f)

        candidates = [
            variant
            for variant in range(self.n_variants)
            if variant_index.get(f"v{variant}", {}).get("frame_idx", [])
            and variant_index[f"v{variant}"]["frame_idx"][0] <= int(step_index)
        ]
        variant = int(rng.choice(candidates)) if candidates else 0
        all_frame_indices = np.asarray(
            variant_index[f"v{variant}"]["frame_idx"],
            dtype=np.int64,
        )
        frame_indices = all_frame_indices[all_frame_indices <= int(step_index)]
        chunk_frames = int(chunk_index["chunk_frames"])
        n_complete_chunks = len(frame_indices) // chunk_frames

        latent_path = episode_dir / f"latents_v{variant}.npy"
        latents = np.load(latent_path, mmap_mode="r")[:n_complete_chunks]
        latents = _pool_latents(np.asarray(latents), self.layout)
        if latents.shape[0]:
            latent_stream = np.ascontiguousarray(latents).reshape(-1, latents.shape[-1]).astype(
                np.float32,
                copy=False,
            )
        else:
            latent_stream = np.zeros((0, self.layout.latent_dim), dtype=np.float32)

        recent_start = min(len(frame_indices), n_complete_chunks * chunk_frames)
        recent_indices = frame_indices[recent_start:]
        recent_features = np.load(episode_dir / "dinov3.npy", mmap_mode="r")
        if len(recent_indices):
            recent_stream = np.ascontiguousarray(recent_features[recent_indices]).reshape(
                -1,
                self.layout.recent_dim,
            ).astype(np.float32, copy=False)
        else:
            recent_stream = np.zeros((0, self.layout.recent_dim), dtype=np.float32)

        local_actions_path = episode_dir / f"action_causal_v{variant}.npy"
        if local_actions_path.is_file():
            local_actions = np.load(local_actions_path, mmap_mode="r")
            if len(local_actions) != len(all_frame_indices):
                raise ValueError(
                    f"{local_actions_path} has {len(local_actions)} rows, expected "
                    f"{len(all_frame_indices)} from variant_index.json."
                )
            action_stream = np.ascontiguousarray(local_actions[: len(frame_indices)]).astype(
                np.float32,
                copy=False,
            )
        else:
            action_proprio = np.load(
                episode_dir / "action_proprio.npy",
                allow_pickle=True,
            ).item()
            action_source = action_proprio.get("action_causal", action_proprio["action"])
            action_stream = np.ascontiguousarray(action_source[frame_indices]).astype(
                np.float32,
                copy=False,
            )

        action_proprio = np.load(
            episode_dir / "action_proprio.npy",
            allow_pickle=True,
        ).item()
        proprio_stream = np.ascontiguousarray(
            action_proprio["proprio"][frame_indices],
        ).astype(np.float32, copy=False)
        language_embedding = np.load(episode_dir / "lang_emb.npy").astype(
            np.float32,
            copy=False,
        ).reshape(-1)
        if language_embedding.shape != (self.layout.language_dim,):
            raise ValueError(
                f"Stored Pi0.5 language embedding for episode {episode_index} has shape "
                f"{language_embedding.shape}, expected ({self.layout.language_dim},)."
            )

        latent_padded, latent_mask = _fixed_pad(
            latent_stream,
            self.layout.latent_cap,
            self.layout.latent_dim,
        )
        recent_padded, recent_mask = _fixed_pad(
            recent_stream,
            self.layout.max_recent_tokens,
            self.layout.recent_dim,
        )
        action_padded, action_mask = _fixed_pad(
            action_stream,
            self.layout.max_step_tokens,
            self.layout.stream_dim,
        )
        proprio_padded, proprio_mask = _fixed_pad(
            proprio_stream,
            self.layout.max_step_tokens,
            self.layout.stream_dim,
        )
        return {
            "cosmo_latent_emb": latent_padded,
            "cosmo_mem_mask": latent_mask,
            "cosmo_recent_emb": recent_padded,
            "cosmo_recent_mask": recent_mask,
            "cosmo_action_emb": action_padded,
            "cosmo_action_mask": action_mask,
            "cosmo_proprio_emb": proprio_padded,
            "cosmo_proprio_mask": proprio_mask,
            "cosmo_lang_emb": language_embedding,
        }


class CompactRMBenchFrameDataset:
    """LeRobot-shaped, worker-safe view over the compact RMBench Stage-2 store."""

    repo_id = RMBENCH_STAGE2_REPO_ID
    revision = "compact-stage2"

    def __init__(
        self,
        *,
        data_root: str | Path,
        feature_root: str | Path,
        repo_id: str = RMBENCH_STAGE2_REPO_ID,
    ) -> None:
        self.data_root = Path(data_root)
        self.feature_root = Path(feature_root)
        self.repo_id = str(repo_id)
        self.root = self.data_root
        with (self.data_root / "meta" / "samples.json").open(encoding="utf-8") as f:
            self.samples = json.load(f)
        self.features = {
            "observation.images.image": {},
            "observation.images.wrist_image": {},
            "observation.images.right_image": {},
            "observation.state": {},
            "action": {},
            "task": {},
            "cosmo_latent_emb": {},
            "cosmo_mem_mask": {},
            "cosmo_recent_emb": {},
            "cosmo_recent_mask": {},
            "cosmo_action_emb": {},
            "cosmo_action_mask": {},
            "cosmo_proprio_emb": {},
            "cosmo_proprio_mask": {},
            "cosmo_lang_emb": {},
        }
        self.meta = SimpleNamespace(
            camera_keys=[
                "observation.images.image",
                "observation.images.wrist_image",
                "observation.images.right_image",
            ],
            episodes=[],
            features=self.features,
        )
        self._memory = CompactRMBenchMemoryAssembler(feature_root=self.feature_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[int(index)]
        sample_index = int(sample["sample_index"])
        with (self.data_root / "data" / f"{sample_index}.pkl").open("rb") as f:
            row = pickle.load(f)
        frame = {
            "index": sample_index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["step_index"]),
            "task": row["prompt"],
            "observation.images.image": row["image"],
            "observation.images.wrist_image": row["wrist_image"],
            "observation.images.right_image": row["right_image"],
            "observation.state": row["state"],
            "action": row["actions"],
        }
        frame.update(
            self._memory.gather(
                episode_index=frame["episode_index"],
                step_index=frame["frame_index"],
                rng=np.random.default_rng(),
            )
        )
        return frame


def load_compact_rmbench_stats(data_root: str | Path) -> Dict[str, Any]:
    """Translate compact-store statistics to the LeRobot robot-processor schema."""
    with (Path(data_root) / "meta" / "stats.json").open(encoding="utf-8") as f:
        stats = json.load(f)
    count = [float(stats["total_samples"])]
    return {
        "action": {**stats["actions"], "count": count},
        "observation.state": {**stats["state"], "count": count},
    }


def compact_rmbench_dataset_from_environment() -> CompactRMBenchFrameDataset:
    data_root = os.environ.get(RMBENCH_STAGE2_DATA_ROOT_ENV, "").strip()
    feature_root = os.environ.get(RMBENCH_STAGE1_FEATURE_ROOT_ENV, "").strip()
    if not data_root or not feature_root:
        raise ValueError(
            f"{RMBENCH_STAGE2_DATA_ROOT_ENV} and {RMBENCH_STAGE1_FEATURE_ROOT_ENV} "
            "must be set for the rmbench_stage2 mixture."
        )
    return CompactRMBenchFrameDataset(data_root=data_root, feature_root=feature_root)
