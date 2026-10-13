from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")
DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINO_IMAGE_SIZE = 256
DINO_FEATURE_DIM = 768
DINO_TOKENS_PER_VIEW = 64
PI05_LANGUAGE_DIM = 2048
PI05_LANGUAGE_EMBEDDING_RTOL = 1e-5
PI05_LANGUAGE_EMBEDDING_ATOL = 1e-6
WAN_NATIVE_PATCH_STRIDE = 16


@dataclass(frozen=True)
class RMBenchMemoryLayout:
    temporal_factor: int = 4
    downsample: int = 2
    latent_cap: int = 4096
    max_recent_tokens: int = 768
    max_step_tokens: int = 1244
    latent_dim: int = 48
    recent_dim: int = 768
    stream_dim: int = 16
    num_views: int = 3
    pool_h: int = 2
    pool_w: int = 2
    wan_h: int = 256
    wan_w: int = 256

    def __post_init__(self) -> None:
        positive_values = {
            "temporal_factor": self.temporal_factor,
            "downsample": self.downsample,
            "latent_cap": self.latent_cap,
            "max_recent_tokens": self.max_recent_tokens,
            "max_step_tokens": self.max_step_tokens,
            "latent_dim": self.latent_dim,
            "recent_dim": self.recent_dim,
            "stream_dim": self.stream_dim,
            "num_views": self.num_views,
            "pool_h": self.pool_h,
            "pool_w": self.pool_w,
            "wan_h": self.wan_h,
            "wan_w": self.wan_w,
        }
        for name, value in positive_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but received {value!r}.")
        if self.num_views != len(CAMERA_ORDER):
            raise ValueError(
                f"RMBench memory requires {len(CAMERA_ORDER)} views, but received {self.num_views}."
            )
        if self.latent_dim != 48:
            raise ValueError(
                f"RMBench memory latent_dim must be 48, but received {self.latent_dim}."
            )
        if self.recent_dim != DINO_FEATURE_DIM:
            raise ValueError(
                "RMBench memory recent_dim must be "
                f"{DINO_FEATURE_DIM}, but received {self.recent_dim}."
            )
        if self.stream_dim != 16:
            raise ValueError(
                f"RMBench memory stream_dim must be 16, but received {self.stream_dim}."
            )
        if (self.pool_h, self.pool_w) != (2, 2):
            raise ValueError(
                "RMBench memory currently requires 2x2 latent pooling, but received "
                f"{self.pool_h}x{self.pool_w}."
            )
        if self.wan_h != 256 or self.wan_w != 256:
            raise ValueError(
                "RMBench memory currently requires 256x256 Wan inputs, but received "
                f"{self.wan_h}x{self.wan_w}."
            )
        if self.wan_h % WAN_NATIVE_PATCH_STRIDE or self.wan_w % WAN_NATIVE_PATCH_STRIDE:
            raise ValueError(
                "RMBench memory Wan dimensions must be divisible by the native patch stride "
                f"{WAN_NATIVE_PATCH_STRIDE}, but received {self.wan_h}x{self.wan_w}."
            )
        grid_h, grid_w = self.wan_grid_shape
        if grid_h % self.pool_h or grid_w % self.pool_w:
            raise ValueError(
                "RMBench memory latent pooling must evenly divide the native Wan grid "
                f"{grid_h}x{grid_w}, "
                f"but received {self.pool_h}x{self.pool_w}."
            )

    @property
    def wan_grid_shape(self) -> tuple[int, int]:
        return (
            self.wan_h // WAN_NATIVE_PATCH_STRIDE,
            self.wan_w // WAN_NATIVE_PATCH_STRIDE,
        )


def build_wan_frontend(device: str, variant: str = "Wan2.2-TI2V-5B"):
    try:
        from mme_vla_suite.models.representation.wan_frontend import WanFrontend
    except ImportError as exc:
        raise RuntimeError(
            "The pinned Pi0.5 Wan source must be on PYTHONPATH to build the "
            "RMBench online memory frontend."
        ) from exc
    return WanFrontend(variant=variant, device=device)


def pool_dinov3_hidden_states(
    hidden: np.ndarray,
    *,
    num_register_tokens: int,
    patch_grid: tuple[int, int],
) -> np.ndarray:
    """Drop DINOv3 prefix tokens and average native 16x16 patches into an 8x8 grid."""
    hidden = np.asarray(hidden)
    if hidden.ndim != 3:
        raise ValueError(
            "DINOv3 hidden states must have shape [B, N, D], "
            f"but received {hidden.shape}."
        )
    if tuple(patch_grid) != (16, 16):
        raise ValueError(
            "DINOv3 recent-feature pooling requires patch_grid=(16, 16), "
            f"but received {patch_grid!r}."
        )
    if hidden.shape[-1] != DINO_FEATURE_DIM:
        raise ValueError(
            f"DINOv3 hidden states must have feature dim {DINO_FEATURE_DIM}, "
            f"but received {hidden.shape[-1]}."
        )
    if isinstance(num_register_tokens, bool) or not isinstance(
        num_register_tokens, (int, np.integer)
    ):
        raise ValueError(
            "DINOv3 num_register_tokens must be a non-negative integer, "
            f"but received {num_register_tokens!r}."
        )
    num_register_tokens = int(num_register_tokens)
    if num_register_tokens < 0:
        raise ValueError(
            "DINOv3 num_register_tokens must be a non-negative integer, "
            f"but received {num_register_tokens}."
        )

    prefix_tokens = 1 + num_register_tokens
    expected_tokens = prefix_tokens + 16 * 16
    if hidden.shape[1] != expected_tokens:
        raise ValueError(
            "DINOv3 hidden states have an unexpected token count: expected "
            f"{expected_tokens} for 1 CLS token, {num_register_tokens} register tokens, "
            f"and a 16x16 patch grid, but received {hidden.shape[1]}."
        )

    patches = hidden[:, prefix_tokens:, :].reshape(
        hidden.shape[0],
        8,
        2,
        8,
        2,
        DINO_FEATURE_DIM,
    )
    return patches.mean(axis=(2, 4)).reshape(hidden.shape[0], DINO_TOKENS_PER_VIEW, DINO_FEATURE_DIM)


@dataclass(frozen=True)
class RMBenchLanguageEmbeddingResolver:
    """Maps exact RMBench prompt text to its saved Pi0.5 language embedding."""

    by_prompt: dict[str, np.ndarray]

    @classmethod
    def build(
        cls,
        stage2_root: Path,
        feature_root: Path,
    ) -> RMBenchLanguageEmbeddingResolver:
        episodes_path = Path(stage2_root) / "meta" / "episodes.jsonl"
        grouped: dict[str, list[np.ndarray]] = {}

        with episodes_path.open("r", encoding="utf-8") as episodes_file:
            for line_number, line in enumerate(episodes_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    prompt = record["prompt"]
                    episode_index = int(record["global_episode_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid RMBench episode record at {episodes_path}:{line_number}; "
                        "expected prompt and global_episode_index."
                    ) from exc
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"Invalid RMBench episode record at {episodes_path}:{line_number}; "
                        f"prompt must be a string, but received {type(prompt).__name__}."
                    )

                embedding_path = (
                    Path(feature_root) / f"episode_{episode_index}" / "lang_emb.npy"
                )
                embedding = np.asarray(
                    np.load(embedding_path, allow_pickle=False),
                    dtype=np.float32,
                )
                if embedding.shape != (PI05_LANGUAGE_DIM,):
                    raise ValueError(
                        "Stored Pi0.5 language embedding must have shape "
                        f"({PI05_LANGUAGE_DIM},), but {embedding_path} has shape {embedding.shape}."
                    )
                grouped.setdefault(prompt, []).append(embedding.copy())

        resolved: dict[str, np.ndarray] = {}
        for prompt, embeddings in grouped.items():
            reference = embeddings[0]
            if not all(
                np.allclose(
                    reference,
                    embedding,
                    rtol=PI05_LANGUAGE_EMBEDDING_RTOL,
                    atol=PI05_LANGUAGE_EMBEDDING_ATOL,
                    equal_nan=False,
                )
                for embedding in embeddings[1:]
            ):
                raise ValueError(
                    "Stored Pi0.5 language embeddings disagree for exact prompt "
                    f"{prompt!r}."
                )
            resolved[prompt] = reference.copy()
        return cls(by_prompt=resolved)

    def resolve(self, prompt: str) -> np.ndarray:
        try:
            embedding = self.by_prompt[prompt]
        except KeyError as exc:
            raise KeyError(
                f"no stored Pi0.5 language embedding for exact prompt {prompt!r}"
            ) from exc
        return embedding.copy()


class DinoV3RecentEncoder:
    """Frozen local DINOv3 encoder for RMBench recent-history camera frames."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        batch_size: int = 24,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"DINOv3 batch_size must be positive, but received {batch_size}.")

        import torch
        from transformers import AutoImageProcessor, AutoModel

        local_path = Path(model_path).expanduser().resolve()
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"DINOv3 model_path must be an existing local directory, but received {local_path}."
            )

        processor = AutoImageProcessor.from_pretrained(
            str(local_path),
            revision=DINO_MODEL_REVISION,
            local_files_only=True,
        )
        processor.size = {"height": DINO_IMAGE_SIZE, "width": DINO_IMAGE_SIZE}
        processor.crop_size = {"height": DINO_IMAGE_SIZE, "width": DINO_IMAGE_SIZE}
        processor.do_resize = True
        processor.do_center_crop = True
        model = AutoModel.from_pretrained(
            str(local_path),
            revision=DINO_MODEL_REVISION,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        model.eval()

        config = model.config
        patch_size = getattr(config, "patch_size", None)
        hidden_size = getattr(config, "hidden_size", None)
        if patch_size != 16:
            raise ValueError(
                f"DINOv3 model must use patch_size=16, but local model has {patch_size!r}."
            )
        if hidden_size != DINO_FEATURE_DIM:
            raise ValueError(
                f"DINOv3 model must use hidden_size={DINO_FEATURE_DIM}, "
                f"but local model has {hidden_size!r}."
            )
        num_register_tokens = getattr(config, "num_register_tokens", None)
        if isinstance(num_register_tokens, bool) or not isinstance(num_register_tokens, int):
            raise ValueError(
                "DINOv3 model config must define an integer num_register_tokens, "
                f"but local model has {num_register_tokens!r}."
            )

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device
        self._batch_size = int(batch_size)
        self._local_path = local_path
        self._patch_size = int(patch_size)
        self._num_register_tokens = int(num_register_tokens)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            raise ValueError(
                f"DINOv3 recent-history frames must use uint8 dtype, but received {frames.dtype}."
            )
        if frames.ndim != 5:
            raise ValueError(
                "DINOv3 recent-history frames must have shape [T, 3, H, W, 3], "
                f"but received {frames.shape}."
            )
        if frames.shape[1] != len(CAMERA_ORDER) or frames.shape[-1] != 3:
            raise ValueError(
                "DINOv3 recent-history frames must have exactly three RGB cameras in "
                f"{CAMERA_ORDER} order, but received {frames.shape}."
            )
        if frames.shape[0] < 1 or frames.shape[2] < 1 or frames.shape[3] < 1:
            raise ValueError(
                "DINOv3 recent-history frames require non-empty time, height, and width "
                f"dimensions, but received {frames.shape}."
            )

        time_steps = frames.shape[0]
        flat_frames = frames.reshape(-1, frames.shape[2], frames.shape[3], 3)
        batches: list[np.ndarray] = []
        for start in range(0, flat_frames.shape[0], self._batch_size):
            image_batch = list(flat_frames[start : start + self._batch_size])
            processed = self._processor(images=image_batch, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(
                self._device,
                dtype=self._torch.bfloat16,
            )
            with self._torch.inference_mode():
                hidden_states = self._model(pixel_values=pixel_values).last_hidden_state
            pooled = pool_dinov3_hidden_states(
                hidden_states.float().cpu().numpy(),
                num_register_tokens=self._num_register_tokens,
                patch_grid=(16, 16),
            )
            expected_batch_shape = (len(image_batch), DINO_TOKENS_PER_VIEW, DINO_FEATURE_DIM)
            if pooled.shape != expected_batch_shape:
                raise ValueError(
                    "DINOv3 pooled output has unexpected shape: expected "
                    f"{expected_batch_shape}, but received {pooled.shape}."
                )
            batches.append(pooled.astype(np.float32, copy=False))

        output = np.concatenate(batches, axis=0).reshape(
            time_steps,
            len(CAMERA_ORDER),
            DINO_TOKENS_PER_VIEW,
            DINO_FEATURE_DIM,
        )
        expected_shape = (
            time_steps,
            len(CAMERA_ORDER),
            DINO_TOKENS_PER_VIEW,
            DINO_FEATURE_DIM,
        )
        if output.shape != expected_shape:
            raise ValueError(
                f"DINOv3 recent encoder produced shape {output.shape}, expected {expected_shape}."
            )
        return output

    def provenance(self) -> dict[str, str | int]:
        return {
            "model_id": DINO_MODEL_ID,
            "revision": DINO_MODEL_REVISION,
            "image_size": DINO_IMAGE_SIZE,
            "patch_size": self._patch_size,
            "num_register_tokens": self._num_register_tokens,
            "local_path": str(self._local_path),
        }


class RMBenchOnlineMemoryConditioner:
    """Builds live RMBench memory inputs without reading offline visual features."""

    def __init__(
        self,
        layout: RMBenchMemoryLayout,
        dino_encoder,
        wan_frontend,
        offline_feature_root: str | Path | None = None,
    ) -> None:
        if not isinstance(layout, RMBenchMemoryLayout):
            raise TypeError(
                "layout must be an RMBenchMemoryLayout, but received "
                f"{type(layout).__name__}."
            )
        if offline_feature_root is not None and str(offline_feature_root).strip():
            raise ValueError(
                "offline visual features are forbidden for RMBench online memory rollouts."
            )
        if not callable(dino_encoder):
            raise TypeError("dino_encoder must be callable.")
        if not callable(getattr(wan_frontend, "make_streamer", None)):
            raise TypeError("wan_frontend must define make_streamer(num_views=3).")

        self.layout = layout
        self._dino_encoder = dino_encoder
        self._wan_frontend = wan_frontend
        self._language_embedding: np.ndarray | None = None
        self._streamer = None
        self._stream_all: list[np.ndarray] = []
        self._dino_history: list[np.ndarray] = []
        self._raw_action_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []
        self._proprio_history: list[np.ndarray] = []
        self._absolute_frame_count = 0

    @staticmethod
    def _float32_vector(value, *, name: str, width: int) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be float32-compatible with shape ({width},).") from exc
        if vector.shape != (width,):
            raise ValueError(
                f"{name} must have shape ({width},), but received {vector.shape}."
            )
        return vector.copy()

    @staticmethod
    def _is_memmap_backed(value: np.ndarray) -> bool:
        current = value
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if isinstance(current, np.memmap):
                return True
            seen.add(id(current))
            current = getattr(current, "base", None)
        return False

    def reset(self, language_embedding) -> None:
        self._language_embedding = self._float32_vector(
            language_embedding,
            name="language_embedding",
            width=PI05_LANGUAGE_DIM,
        )
        self._streamer = self._wan_frontend.make_streamer(num_views=self.layout.num_views)
        if not callable(getattr(self._streamer, "push", None)):
            raise TypeError("Wan streamer must define push(frames).")
        self._stream_all = []
        self._dino_history = []
        self._raw_action_history = []
        self._action_history = []
        self._proprio_history = []
        self._absolute_frame_count = 0

    def _pool_latent_chunks(self, chunks) -> list[np.ndarray]:
        if isinstance(chunks, np.ndarray):
            chunks = [chunks]
        try:
            iterator = iter(chunks)
        except TypeError as exc:
            raise ValueError("Wan streamer must return an iterable of latent chunks.") from exc

        pooled_chunks: list[np.ndarray] = []
        output_tokens = self.layout.num_views * self.layout.pool_h * self.layout.pool_w
        grid_h, grid_w = self.layout.wan_grid_shape
        native_tokens_per_view = grid_h * grid_w
        for chunk in iterator:
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim == 2 and chunk.shape == (output_tokens, self.layout.latent_dim):
                pooled_chunks.append(chunk.copy())
                continue

            native_shape = (
                self.layout.num_views,
                grid_h,
                grid_w,
                self.layout.latent_dim,
            )
            if chunk.ndim == 2 and chunk.shape == (
                self.layout.num_views * native_tokens_per_view,
                self.layout.latent_dim,
            ):
                native = chunk.reshape(native_shape)
            elif chunk.ndim == 3 and chunk.shape == (
                self.layout.num_views,
                native_tokens_per_view,
                self.layout.latent_dim,
            ):
                native = chunk.reshape(native_shape)
            elif chunk.shape == native_shape:
                native = chunk
            else:
                raise ValueError(
                    "Wan latent chunk must be prepooled with shape "
                    f"({output_tokens}, {self.layout.latent_dim}) or represent three "
                    f"native {grid_h}x{grid_w} grids, but received {chunk.shape}."
                )

            pooled = native.reshape(
                self.layout.num_views,
                self.layout.pool_h,
                grid_h // self.layout.pool_h,
                self.layout.pool_w,
                grid_w // self.layout.pool_w,
                self.layout.latent_dim,
            ).mean(axis=(2, 4))
            pooled = pooled.reshape(output_tokens, self.layout.latent_dim)
            if pooled.shape != (output_tokens, self.layout.latent_dim):
                raise ValueError(
                    f"Pooled Wan latent chunk has shape {pooled.shape}, expected "
                    f"({output_tokens}, {self.layout.latent_dim})."
                )
            pooled_chunks.append(pooled.astype(np.float32, copy=False))
        return pooled_chunks

    def append(self, frames, state, executed_action) -> None:
        if self._streamer is None or self._language_embedding is None:
            raise RuntimeError("RMBench online memory must be reset before append().")
        if (
            not isinstance(frames, np.ndarray)
            or self._is_memmap_backed(frames)
            or frames.dtype != np.uint8
        ):
            raise ValueError("frames must be a live uint8 ndarray with shape [3, H, W, 3].")
        if (
            frames.ndim != 4
            or frames.shape[0] != self.layout.num_views
            or frames.shape[-1] != 3
            or frames.shape[1] < 1
            or frames.shape[2] < 1
        ):
            raise ValueError(
                "frames must have shape [3, H, W, 3] in head_camera, left_camera, "
                f"right_camera order, but received {frames.shape}."
            )

        proprio = self._float32_vector(state, name="state", width=self.layout.stream_dim)
        action = self._float32_vector(
            executed_action,
            name="executed_action",
            width=self.layout.stream_dim,
        )
        if self._absolute_frame_count % self.layout.downsample == 0:
            if self._absolute_frame_count == 0:
                causal_action = np.zeros_like(action)
            elif self.layout.downsample == 1:
                causal_action = action
            else:
                previous_sampled_frame = (
                    self._absolute_frame_count - self.layout.downsample + 1
                )
                if previous_sampled_frame >= len(self._raw_action_history):
                    raise RuntimeError(
                        "RMBench action history is missing the action required for "
                        "variant-causal memory conditioning."
                    )
                causal_action = self._raw_action_history[previous_sampled_frame]
            dino_features = np.asarray(self._dino_encoder(frames[None, ...]), dtype=np.float32)
            expected_dino_shape = (
                1,
                self.layout.num_views,
                DINO_TOKENS_PER_VIEW,
                self.layout.recent_dim,
            )
            if dino_features.shape != expected_dino_shape:
                raise ValueError(
                    f"DINO encoder produced shape {dino_features.shape}, expected "
                    f"{expected_dino_shape}."
                )
            self._dino_history.append(dino_features[0].copy())
            self._stream_all.extend(
                self._pool_latent_chunks(self._streamer.push(frames[None, ...]))
            )
            self._proprio_history.append(proprio)
            self._action_history.append(causal_action)
        self._raw_action_history.append(action)
        self._absolute_frame_count += 1

    @staticmethod
    def _tail_pad(values: np.ndarray, length: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float32)
        if length < 1 or width < 1:
            raise ValueError("tail padding length and width must be positive.")
        if values.size == 0:
            values = np.empty((0, width), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(
                f"values must have shape [N, {width}], but received {values.shape}."
            )
        selected = values[-length:]
        output = np.zeros((length, width), dtype=np.float32)
        mask = np.zeros(length, dtype=bool)
        output[: len(selected)] = selected
        mask[: len(selected)] = True
        return output, mask

    def as_model_inputs(self, device: str):
        if self._streamer is None or self._language_embedding is None:
            raise RuntimeError("RMBench online memory must be reset before as_model_inputs().")

        import torch

        target_device = torch.device(device)
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested unavailable CUDA device {target_device}.")
            if target_device.index is not None and target_device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested unavailable CUDA device {target_device}.")

        n_complete_chunks = len(self._dino_history) // self.layout.temporal_factor
        if len(self._stream_all) < n_complete_chunks:
            raise RuntimeError(
                "Wan streamer emitted fewer latent chunks than required by the "
                "Stage-2 chunk boundary: "
                f"received {len(self._stream_all)}, expected at least {n_complete_chunks} "
                f"for {len(self._dino_history)} history frames."
            )
        finalized_chunks = self._stream_all[:n_complete_chunks]
        latent_values = (
            np.concatenate(finalized_chunks, axis=0)
            if finalized_chunks
            else np.empty((0, self.layout.latent_dim), dtype=np.float32)
        )
        cosmo, cosmo_mask = self._tail_pad(
            latent_values,
            self.layout.latent_cap,
            self.layout.latent_dim,
        )
        recent_history = self._dino_history[
            n_complete_chunks * self.layout.temporal_factor :
        ]
        recent_values = (
            np.asarray(recent_history, dtype=np.float32).reshape(-1, self.layout.recent_dim)
            if recent_history
            else np.empty((0, self.layout.recent_dim), dtype=np.float32)
        )
        recent, recent_mask = self._tail_pad(
            recent_values,
            self.layout.max_recent_tokens,
            self.layout.recent_dim,
        )
        actions, action_mask = self._tail_pad(
            np.asarray(self._action_history, dtype=np.float32),
            self.layout.max_step_tokens,
            self.layout.stream_dim,
        )
        proprio, proprio_mask = self._tail_pad(
            np.asarray(self._proprio_history, dtype=np.float32),
            self.layout.max_step_tokens,
            self.layout.stream_dim,
        )

        def tensor(values: np.ndarray):
            return torch.from_numpy(values[None]).to(device=target_device, dtype=torch.float32)

        def mask_tensor(values: np.ndarray):
            return torch.from_numpy(values[None]).to(device=target_device, dtype=torch.bool)

        return {
            "memory_cosmo_latents": tensor(cosmo),
            "memory_cosmo_mask": mask_tensor(cosmo_mask),
            "memory_recent_embeddings": tensor(recent),
            "memory_recent_mask": mask_tensor(recent_mask),
            "memory_action_embeddings": tensor(actions),
            "memory_action_mask": mask_tensor(action_mask),
            "memory_proprio_embeddings": tensor(proprio),
            "memory_proprio_mask": mask_tensor(proprio_mask),
            "memory_language_embedding": tensor(self._language_embedding),
        }
