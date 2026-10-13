import json
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lerobot"
    / "src"
    / "lerobot"
    / "policies"
    / "molmoact2"
    / "rmbench_memory.py"
)
_SPEC = importlib.util.spec_from_file_location("rmbench_memory_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PI05_LANGUAGE_DIM = _MODULE.PI05_LANGUAGE_DIM
PI05_LANGUAGE_EMBEDDING_ATOL = _MODULE.PI05_LANGUAGE_EMBEDDING_ATOL
PI05_LANGUAGE_EMBEDDING_RTOL = _MODULE.PI05_LANGUAGE_EMBEDDING_RTOL
RMBenchLanguageEmbeddingResolver = _MODULE.RMBenchLanguageEmbeddingResolver
DinoV3RecentEncoder = _MODULE.DinoV3RecentEncoder
RMBenchMemoryLayout = _MODULE.RMBenchMemoryLayout
RMBenchOnlineMemoryConditioner = _MODULE.RMBenchOnlineMemoryConditioner
pool_dinov3_hidden_states = _MODULE.pool_dinov3_hidden_states


def test_pool_dinov3_hidden_states_discards_prefix_and_pools_2x2_patches() -> None:
    hidden = np.arange(1 * (1 + 2 + 256) * 768, dtype=np.float32).reshape(1, 259, 768)

    pooled = pool_dinov3_hidden_states(
        hidden,
        num_register_tokens=2,
        patch_grid=(16, 16),
    )

    assert pooled.shape == (1, 64, 768)
    source_patches = hidden[0, 3:].reshape(16, 16, 768)
    np.testing.assert_array_equal(
        pooled[0, 0],
        source_patches[:2, :2].mean(axis=(0, 1)),
    )


def _write_language_features(
    tmp_path,
    *,
    prompts: list[object],
    vectors: list[np.ndarray],
) -> tuple[Path, Path]:
    stage2_root = tmp_path / "stage2"
    feature_root = tmp_path / "features"
    meta_dir = stage2_root / "meta"
    meta_dir.mkdir(parents=True)

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for episode_id, prompt in enumerate(prompts):
            f.write(
                json.dumps(
                    {
                        "global_episode_index": episode_id,
                        "prompt": prompt,
                    }
                )
                + "\n"
            )
            episode_dir = feature_root / f"episode_{episode_id}"
            episode_dir.mkdir(parents=True)
            np.save(episode_dir / "lang_emb.npy", vectors[episode_id])

    return stage2_root, feature_root


def test_language_resolver_uses_exact_prompt_and_rejects_missing_prompt(tmp_path) -> None:
    vector = np.arange(PI05_LANGUAGE_DIM, dtype=np.float32)
    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, vector.copy()],
    )

    resolver = RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)

    np.testing.assert_array_equal(resolver.resolve("open the drawer"), vector)
    with pytest.raises(KeyError, match="no stored Pi0.5 language embedding"):
        resolver.resolve("open drawer")


def test_language_resolver_rejects_conflicting_prompt_embeddings(tmp_path) -> None:
    vector = np.zeros(PI05_LANGUAGE_DIM, dtype=np.float32)
    conflicting_vector = vector.copy()
    conflicting_vector[0] = 2.0
    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, conflicting_vector],
    )

    with pytest.raises(ValueError, match="language embeddings disagree"):
        RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)


def test_language_resolver_accepts_duplicate_embeddings_within_float32_tolerance(tmp_path) -> None:
    vector = np.zeros(PI05_LANGUAGE_DIM, dtype=np.float32)
    nearly_equal = vector.copy()
    nearly_equal[0] = PI05_LANGUAGE_EMBEDDING_ATOL / 2
    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, nearly_equal],
    )

    resolver = RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)

    np.testing.assert_array_equal(resolver.resolve("open the drawer"), vector)


def test_language_resolver_rejects_duplicate_embeddings_beyond_float32_tolerance(tmp_path) -> None:
    vector = np.zeros(PI05_LANGUAGE_DIM, dtype=np.float32)
    different = vector.copy()
    different[0] = PI05_LANGUAGE_EMBEDDING_ATOL * 2
    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, different],
    )

    with pytest.raises(ValueError, match="language embeddings disagree"):
        RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)


def test_language_resolver_accepts_nonzero_duplicate_difference_within_rtol(tmp_path) -> None:
    vector = np.ones(PI05_LANGUAGE_DIM, dtype=np.float32)
    nearly_equal = vector.copy()
    nearly_equal[0] += PI05_LANGUAGE_EMBEDDING_ATOL * 2
    difference = abs(float(nearly_equal[0] - vector[0]))
    assert difference > PI05_LANGUAGE_EMBEDDING_ATOL
    assert difference < PI05_LANGUAGE_EMBEDDING_RTOL * abs(float(vector[0]))

    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, nearly_equal],
    )

    resolver = RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)

    np.testing.assert_array_equal(resolver.resolve("open the drawer"), vector)


def test_language_resolver_rejects_nonzero_duplicate_difference_beyond_rtol(tmp_path) -> None:
    vector = np.ones(PI05_LANGUAGE_DIM, dtype=np.float32)
    different = vector.copy()
    different[0] += PI05_LANGUAGE_EMBEDDING_RTOL * 2
    difference = abs(float(different[0] - vector[0]))
    assert difference > PI05_LANGUAGE_EMBEDDING_RTOL * abs(float(vector[0]))

    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, different],
    )

    with pytest.raises(ValueError, match="language embeddings disagree"):
        RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)


def test_language_resolver_rejects_duplicate_nan_embeddings(tmp_path) -> None:
    vector = np.zeros(PI05_LANGUAGE_DIM, dtype=np.float32)
    vector[0] = np.nan
    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=["open the drawer", "open the drawer"],
        vectors=[vector, vector.copy()],
    )

    with pytest.raises(ValueError, match="language embeddings disagree"):
        RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)


def test_language_resolver_rejects_non_string_metadata_prompt(tmp_path) -> None:
    stage2_root, feature_root = _write_language_features(
        tmp_path,
        prompts=[123],
        vectors=[np.zeros(PI05_LANGUAGE_DIM, dtype=np.float32)],
    )

    with pytest.raises(ValueError, match="prompt must be a string"):
        RMBenchLanguageEmbeddingResolver.build(stage2_root, feature_root)


def test_dino_recent_encoder_casts_model_input_to_bfloat16_and_returns_float32() -> None:
    class FakeTensor:
        def __init__(self, array) -> None:
            self.array = np.asarray(array)
            self.to_calls = []

        def to(self, device, dtype=None):
            self.to_calls.append((device, dtype))
            return self

        def float(self):
            self.array = self.array.astype(np.float32, copy=False)
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakeTorch:
        bfloat16 = object()

        @staticmethod
        def inference_mode():
            return nullcontext()

    class FakeProcessor:
        def __init__(self) -> None:
            self.pixel_values = FakeTensor(np.zeros((6, 3, 256, 256), dtype=np.float32))

        def __call__(self, *, images, return_tensors):
            assert len(images) == 6
            assert return_tensors == "pt"
            return {"pixel_values": self.pixel_values}

    class FakeModel:
        def __init__(self) -> None:
            self.inputs = []

        def __call__(self, *, pixel_values):
            self.inputs.append(pixel_values)
            return SimpleNamespace(
                last_hidden_state=FakeTensor(np.zeros((6, 259, 768), dtype=np.float32))
            )

    encoder = object.__new__(DinoV3RecentEncoder)
    encoder._torch = FakeTorch()
    encoder._processor = FakeProcessor()
    encoder._model = FakeModel()
    encoder._device = "fake-cuda:0"
    encoder._batch_size = 24
    encoder._num_register_tokens = 2

    output = encoder(np.zeros((2, 3, 4, 5, 3), dtype=np.uint8))

    assert encoder._model.inputs == [encoder._processor.pixel_values]
    assert encoder._processor.pixel_values.to_calls == [
        ("fake-cuda:0", encoder._torch.bfloat16)
    ]
    assert output.shape == (2, 3, 64, 768)
    assert output.dtype == np.float32


class TerminalFakeStreamer:
    def __init__(self) -> None:
        self.frames = []

    def push(self, frames: np.ndarray) -> list[np.ndarray]:
        assert frames.ndim == 5
        assert frames.shape[0] == 1
        assert frames.shape[1] == 3
        self.frames.append(frames.copy())
        if len(self.frames) in (1, 5):
            return [np.full((12, 48), len(self.frames), dtype=np.float32)]
        return []


class FakeWanFrontend:
    def __init__(self) -> None:
        self.requested_num_views = None
        self.streamer = TerminalFakeStreamer()

    def make_streamer(self, *, num_views: int) -> TerminalFakeStreamer:
        assert num_views == 3
        self.requested_num_views = num_views
        return self.streamer


class CapacityTerminalStreamer(TerminalFakeStreamer):
    def push(self, frames: np.ndarray) -> list[np.ndarray]:
        assert frames.ndim == 5
        assert frames.shape[0] == 1
        assert frames.shape[1] == 3
        self.frames.append(frames.copy())
        if (len(self.frames) - 1) % 4 == 0:
            return [np.full((12, 48), len(self.frames), dtype=np.float32)]
        return []


class CapacityWanFrontend:
    def __init__(self) -> None:
        self.streamer = CapacityTerminalStreamer()

    def make_streamer(self, *, num_views: int) -> CapacityTerminalStreamer:
        assert num_views == 3
        return self.streamer


class FakeDino:
    def __init__(self) -> None:
        self.frames = []

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        self.frames.append(frames.copy())
        assert frames.shape[0] == 1
        return np.full(
            (frames.shape[0], 3, 64, 768),
            len(self.frames),
            dtype=np.float32,
        )


def _install_fake_torch(monkeypatch) -> None:
    class FakeDevice:
        def __init__(self, value) -> None:
            self.value = str(value)
            self.type = self.value.split(":", maxsplit=1)[0]
            self.index = int(self.value.split(":", maxsplit=1)[1]) if ":" in self.value else None

        def __str__(self) -> str:
            return self.value

    class FakeTensor:
        def __init__(self, value) -> None:
            self.value = np.asarray(value)
            self.dtype = self.value.dtype
            self.device = None

        @property
        def shape(self):
            return self.value.shape

        def to(self, *, device, dtype):
            self.device = device
            self.dtype = dtype
            if dtype == "torch.float32":
                self.value = self.value.astype(np.float32)
            elif dtype == "torch.bool":
                self.value = self.value.astype(bool)
            return self

        def sum(self):
            return self.value.sum()

        def numpy(self):
            return self.value

        def __getitem__(self, item):
            return self.value[item]

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        bool="torch.bool",
        device=FakeDevice,
        from_numpy=FakeTensor,
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_online_memory_conditioner_excludes_current_terminal_latent_from_payload(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    layout = RMBenchMemoryLayout(
        temporal_factor=4,
        downsample=1,
        latent_cap=24,
        max_recent_tokens=384,
        max_step_tokens=6,
    )
    dino = FakeDino()
    wan = FakeWanFrontend()
    conditioner = RMBenchOnlineMemoryConditioner(layout, dino, wan)
    language = np.ones(2048, dtype=np.float32)

    conditioner.reset(language)
    for step in range(5):
        frames = np.full((3, 8, 8, 3), step, dtype=np.uint8)
        conditioner.append(
            frames,
            np.full(16, step, dtype=np.float32),
            np.full(16, 100 + step, dtype=np.float32),
        )

    payload = conditioner.as_model_inputs("cpu")

    assert wan.requested_num_views == 3
    assert len(dino.frames) == 5
    assert len(wan.streamer.frames) == 5
    for step, frames in enumerate(wan.streamer.frames):
        np.testing.assert_array_equal(
            frames,
            np.full((1, 3, 8, 8, 3), step, dtype=np.uint8),
        )

    assert tuple(payload["memory_cosmo_latents"].shape) == (1, 24, 48)
    assert tuple(payload["memory_cosmo_mask"].shape) == (1, 24)
    assert payload["memory_cosmo_mask"].sum().item() == 12
    assert payload["memory_cosmo_latents"][0, 0, 0].item() == 1.0
    assert tuple(payload["memory_recent_embeddings"].shape) == (1, 384, 768)
    assert tuple(payload["memory_recent_mask"].shape) == (1, 384)
    assert payload["memory_recent_mask"].sum().item() == 3 * 64
    assert tuple(payload["memory_action_embeddings"].shape) == (1, 6, 16)
    assert tuple(payload["memory_action_mask"].shape) == (1, 6)
    assert payload["memory_action_mask"].sum().item() == 5
    assert tuple(payload["memory_proprio_embeddings"].shape) == (1, 6, 16)
    assert tuple(payload["memory_proprio_mask"].shape) == (1, 6)
    assert payload["memory_proprio_mask"].sum().item() == 5
    assert tuple(payload["memory_language_embedding"].shape) == (1, 2048)
    assert payload["memory_cosmo_latents"].dtype.__str__() == "torch.float32"
    assert payload["memory_cosmo_mask"].dtype.__str__() == "torch.bool"
    np.testing.assert_array_equal(
        payload["memory_language_embedding"].numpy(),
        language[None],
    )
    assert payload["memory_recent_embeddings"][0, 0, 0].item() == 5.0
    assert payload["memory_action_embeddings"][0, :5, 0].tolist() == [0, 101, 102, 103, 104]
    assert payload["memory_proprio_embeddings"][0, :5, 0].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("frame_count", "expected_chunk_values", "expected_recent_values"),
    [
        (4, [1.0], []),
        (5, [1.0], [5.0]),
        (8, [1.0, 5.0], []),
    ],
)
def test_online_memory_conditioner_matches_stage2_chunk_boundary(
    monkeypatch,
    frame_count,
    expected_chunk_values,
    expected_recent_values,
) -> None:
    _install_fake_torch(monkeypatch)
    conditioner = RMBenchOnlineMemoryConditioner(
        RMBenchMemoryLayout(
            temporal_factor=4,
            downsample=1,
            latent_cap=36,
            max_recent_tokens=384,
            max_step_tokens=8,
        ),
        FakeDino(),
        FakeWanFrontend(),
    )
    conditioner.reset(np.zeros(2048, dtype=np.float32))
    for step in range(frame_count):
        conditioner.append(
            np.full((3, 8, 8, 3), step, dtype=np.uint8),
            np.full(16, step, dtype=np.float32),
            np.full(16, 100 + step, dtype=np.float32),
        )

    payload = conditioner.as_model_inputs("cpu")

    assert payload["memory_cosmo_mask"].sum().item() == len(expected_chunk_values) * 12
    assert payload["memory_cosmo_latents"][0, : len(expected_chunk_values) * 12, 0].tolist() == [
        value for value in expected_chunk_values for _ in range(12)
    ]
    assert payload["memory_recent_mask"].sum().item() == len(expected_recent_values) * 3 * 64
    assert payload["memory_recent_embeddings"][
        0, : len(expected_recent_values) * 3 * 64, 0
    ].tolist() == [
        value for value in expected_recent_values for _ in range(3 * 64)
    ]


def test_online_memory_conditioner_uses_previous_sampled_action_for_variant_causality(
    monkeypatch,
) -> None:
    _install_fake_torch(monkeypatch)
    conditioner = RMBenchOnlineMemoryConditioner(
        RMBenchMemoryLayout(
            temporal_factor=4,
            downsample=2,
            latent_cap=36,
            max_recent_tokens=384,
            max_step_tokens=8,
        ),
        FakeDino(),
        FakeWanFrontend(),
    )
    conditioner.reset(np.zeros(2048, dtype=np.float32))

    # The Stage-2 v0 store keeps raw frames [0, 2, 4] and writes each memory
    # action from the preceding selected frame: [zero, action[0], action[2]].
    for raw_step in range(6):
        conditioner.append(
            np.full((3, 8, 8, 3), raw_step, dtype=np.uint8),
            np.full(16, raw_step, dtype=np.float32),
            np.full(16, 0.0 if raw_step == 0 else 100.0 + raw_step - 1, dtype=np.float32),
        )

    payload = conditioner.as_model_inputs("cpu")

    assert payload["memory_action_embeddings"][0, :3, 0].tolist() == [0.0, 100.0, 102.0]


def test_online_memory_conditioner_forbids_offline_visual_features() -> None:
    with pytest.raises(ValueError, match="offline visual features are forbidden"):
        RMBenchOnlineMemoryConditioner(
            RMBenchMemoryLayout(),
            FakeDino(),
            FakeWanFrontend(),
            offline_feature_root="/tmp/offline",
        )


def test_online_memory_conditioner_pools_16x16_wan_latents_without_corrupting_pooled_chunks() -> None:
    conditioner = RMBenchOnlineMemoryConditioner(
        RMBenchMemoryLayout(),
        FakeDino(),
        FakeWanFrontend(),
    )
    native = np.arange(3 * 256 * 48, dtype=np.float32).reshape(3 * 256, 48)
    already_pooled = np.full((12, 48), -1.0, dtype=np.float32)

    pooled_native, preserved = conditioner._pool_latent_chunks([native, already_pooled])

    assert pooled_native.shape == (12, 48)
    expected_first_cell = native.reshape(3, 16, 16, 48)[0, :8, :8].mean(axis=(0, 1))
    np.testing.assert_array_equal(pooled_native[0], expected_first_cell)
    np.testing.assert_array_equal(preserved, already_pooled)


def test_online_memory_conditioner_requires_reset_and_live_frames() -> None:
    conditioner = RMBenchOnlineMemoryConditioner(
        RMBenchMemoryLayout(),
        FakeDino(),
        FakeWanFrontend(),
    )

    with pytest.raises(RuntimeError, match="reset"):
        conditioner.append(
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros(16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="shape"):
        conditioner.reset(np.zeros(2048, dtype=np.float32))
        conditioner.append(
            np.zeros((3, 8, 8, 3), dtype=np.float32),
            np.zeros(16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        )


def test_online_memory_conditioner_records_state_and_action_only_at_selected_steps(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    conditioner = RMBenchOnlineMemoryConditioner(
        RMBenchMemoryLayout(downsample=2, max_step_tokens=6),
        FakeDino(),
        FakeWanFrontend(),
    )
    conditioner.reset(np.zeros(2048, dtype=np.float32))

    for step in range(5):
        conditioner.append(
            np.full((3, 8, 8, 3), step, dtype=np.uint8),
            np.full(16, step, dtype=np.float32),
            np.full(16, 100 + step, dtype=np.float32),
        )

    payload = conditioner.as_model_inputs("cpu")

    assert len(conditioner._dino_history) == 3
    assert payload["memory_recent_mask"].sum().item() == 3 * 3 * 64
    assert payload["memory_action_mask"].sum().item() == 3
    assert payload["memory_proprio_mask"].sum().item() == 3
    assert payload["memory_action_embeddings"][0, :3, 0].tolist() == [0, 101, 103]
    assert payload["memory_proprio_embeddings"][0, :3, 0].tolist() == [0, 2, 4]


def test_online_memory_conditioner_retains_newest_values_after_all_stream_caps_overflow(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    layout = RMBenchMemoryLayout(
        downsample=1,
        latent_cap=18,
        max_recent_tokens=128,
        max_step_tokens=6,
    )
    conditioner = RMBenchOnlineMemoryConditioner(
        layout,
        FakeDino(),
        CapacityWanFrontend(),
    )
    conditioner.reset(np.zeros(2048, dtype=np.float32))

    empty = conditioner.as_model_inputs("cpu")
    for values_key, mask_key in (
        ("memory_cosmo_latents", "memory_cosmo_mask"),
        ("memory_recent_embeddings", "memory_recent_mask"),
        ("memory_action_embeddings", "memory_action_mask"),
        ("memory_proprio_embeddings", "memory_proprio_mask"),
    ):
        assert not empty[mask_key].numpy().any()
        assert not empty[values_key].numpy().any()

    for step in range(13):
        conditioner.append(
            np.full((3, 8, 8, 3), step, dtype=np.uint8),
            np.full(16, step, dtype=np.float32),
            np.full(16, 100 + step, dtype=np.float32),
        )

    payload = conditioner.as_model_inputs("cpu")

    assert tuple(payload["memory_cosmo_latents"].shape) == (1, 18, 48)
    assert tuple(payload["memory_recent_embeddings"].shape) == (1, 128, 768)
    assert tuple(payload["memory_action_embeddings"].shape) == (1, 6, 16)
    assert tuple(payload["memory_proprio_embeddings"].shape) == (1, 6, 16)
    assert payload["memory_cosmo_mask"].sum().item() == 18
    assert payload["memory_recent_mask"].sum().item() == 128
    assert payload["memory_action_mask"].sum().item() == 6
    assert payload["memory_proprio_mask"].sum().item() == 6
    assert payload["memory_cosmo_latents"][0, :6, 0].tolist() == [5.0] * 6
    assert payload["memory_cosmo_latents"][0, 6:, 0].tolist() == [9.0] * 12
    assert payload["memory_recent_embeddings"][0, :, 0].tolist() == [13.0] * 128
    assert payload["memory_action_embeddings"][0, :, 0].tolist() == [107, 108, 109, 110, 111, 112]
    assert payload["memory_proprio_embeddings"][0, :, 0].tolist() == [7, 8, 9, 10, 11, 12]


def test_online_memory_conditioner_zero_pads_partial_payload_after_terminal_finalization(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    layout = RMBenchMemoryLayout(
        downsample=1,
        latent_cap=18,
        max_recent_tokens=512,
        max_step_tokens=8,
    )
    conditioner = RMBenchOnlineMemoryConditioner(
        layout,
        FakeDino(),
        FakeWanFrontend(),
    )
    conditioner.reset(np.zeros(2048, dtype=np.float32))

    for step in range(6):
        conditioner.append(
            np.full((3, 8, 8, 3), step, dtype=np.uint8),
            np.full(16, step, dtype=np.float32),
            np.full(16, 100 + step, dtype=np.float32),
        )

    payload = conditioner.as_model_inputs("cpu")
    stream_contracts = (
        ("memory_cosmo_latents", "memory_cosmo_mask", 12),
        ("memory_recent_embeddings", "memory_recent_mask", 2 * 3 * 64),
        ("memory_action_embeddings", "memory_action_mask", 6),
        ("memory_proprio_embeddings", "memory_proprio_mask", 6),
    )
    for values_key, mask_key, valid_count in stream_contracts:
        values = payload[values_key].numpy()[0]
        mask = payload[mask_key].numpy()[0]
        assert mask[:valid_count].all()
        assert not mask[valid_count:].any()
        assert not values[valid_count:].any()

    assert payload["memory_cosmo_latents"][0, :12, 0].tolist() == [1.0] * 12
    assert payload["memory_recent_embeddings"][0, : 3 * 64, 0].tolist() == [5.0] * (3 * 64)
    assert payload["memory_recent_embeddings"][0, 3 * 64 : 2 * 3 * 64, 0].tolist() == [6.0] * (3 * 64)
    assert payload["memory_action_embeddings"][0, :6, 0].tolist() == [0, 101, 102, 103, 104, 105]
    assert payload["memory_proprio_embeddings"][0, :6, 0].tolist() == [0, 1, 2, 3, 4, 5]


def test_online_memory_conditioner_rejects_mmap_visual_inputs_through_ndarray_base_chain(tmp_path) -> None:
    conditioner = RMBenchOnlineMemoryConditioner(
        RMBenchMemoryLayout(),
        FakeDino(),
        FakeWanFrontend(),
    )
    conditioner.reset(np.zeros(2048, dtype=np.float32))
    frames = np.memmap(
        tmp_path / "offline-frames.bin",
        mode="w+",
        dtype=np.uint8,
        shape=(3, 8, 8, 3),
    )

    for offline_view in (np.asarray(frames), np.asarray(frames).view()):
        with pytest.raises(ValueError, match="live uint8"):
            conditioner.append(
                offline_view,
                np.zeros(16, dtype=np.float32),
                np.zeros(16, dtype=np.float32),
            )
