from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


try:
    import cv2 as _cv2  # noqa: F401
except ModuleNotFoundError:

    def _resize_uniform_image(image, size, interpolation):
        width, height = size
        return np.full(
            (height, width, image.shape[-1]),
            image[0, 0],
            dtype=image.dtype,
        )

    sys.modules["cv2"] = SimpleNamespace(
        INTER_LINEAR=1,
        resize=_resize_uniform_image,
    )


def _load_adapter():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rmbench_molmoact2_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("rmbench_molmoact2_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load adapter module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


def test_adapter_exports_unused_official_client_get_model_hook():
    assert callable(adapter.get_model)
    assert adapter.get_model({"port": 8031}) is None


def make_observation(*, omit_camera: str | None = None) -> dict:
    cameras = {}
    for index, name in enumerate(reversed(adapter.CAMERA_ORDER)):
        if name != omit_camera:
            cameras[name] = {
                "rgb": np.full((4, 6, 3), index, dtype=np.uint8),
            }
    return {
        "observation": cameras,
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


class FakePolicy:
    def __init__(self, actions=None):
        self.actions = (
            np.zeros((1, 1, adapter.ACTION_WIDTH), dtype=np.float32)
            if actions is None
            else actions
        )
        self.reset_count = 0
        self.batches = []
        self.norm_tags = []
        self.n_action_steps = []
        self.generators = []
        self.select_action_calls = 0

    def reset(self):
        self.reset_count += 1

    def select_action(self, batch, *, norm_tag):
        self.select_action_calls += 1
        raise AssertionError("RMBench adapter must request predict_action_chunk")

    def predict_action_chunk(self, batch, *, norm_tag, n_action_steps, generator=None):
        self.batches.append(batch)
        self.norm_tags.append(norm_tag)
        self.n_action_steps.append(n_action_steps)
        self.generators.append(generator)
        return self.actions


class FakeConditioner:
    def __init__(self):
        self.reset_languages = []
        self.appended = []
        self.devices = []
        self.payload_calls = []
        self.offline_read_count = 0

    def reset(self, language_embedding):
        self.reset_languages.append(np.asarray(language_embedding).copy())

    def append(self, frames, state, executed_action):
        self.appended.append(
            (
                np.asarray(frames).copy(),
                np.asarray(state).copy(),
                np.asarray(executed_action).copy(),
            )
        )

    def as_model_inputs(self, *, device):
        self.devices.append(device)
        payload = {
            "memory_cosmo_latents": "cosmo",
            "memory_cosmo_mask": "cosmo_mask",
            "memory_recent_embeddings": "recent",
            "memory_recent_mask": "recent_mask",
            "memory_action_embeddings": "action",
            "memory_action_mask": "action_mask",
            "memory_proprio_embeddings": "proprio",
            "memory_proprio_mask": "proprio_mask",
            "memory_language_embedding": self.reset_languages[-1][None, :],
        }
        self.payload_calls.append(payload)
        return payload


class BatchedMemoryConditioner(FakeConditioner):
    def as_model_inputs(self, *, device):
        self.devices.append(device)
        return {
            "memory_cosmo_latents": np.ones((1, 2, 48), dtype=np.float32),
            "memory_cosmo_mask": np.ones((1, 2), dtype=bool),
            "memory_recent_embeddings": np.ones((1, 3, 768), dtype=np.float32),
            "memory_recent_mask": np.ones((1, 3), dtype=bool),
            "memory_action_embeddings": np.ones((1, 4, 16), dtype=np.float32),
            "memory_action_mask": np.ones((1, 4), dtype=bool),
            "memory_proprio_embeddings": np.ones((1, 4, 16), dtype=np.float32),
            "memory_proprio_mask": np.ones((1, 4), dtype=bool),
            "memory_language_embedding": np.ones((1, 2048), dtype=np.float32),
        }


def test_server_unbatches_online_memory_before_policy_collation():
    policy = FakePolicy(np.zeros((1, 1, adapter.ACTION_WIDTH), dtype=np.float32))
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=BatchedMemoryConditioner(),
        prediction_horizon=1,
    )

    server.eval({"prompt": "put the item away", "observation": make_observation()})

    batch = policy.batches[0]
    assert batch["memory_cosmo_latents"].shape == (2, 48)
    assert batch["memory_cosmo_mask"].shape == (2,)
    assert batch["memory_recent_embeddings"].shape == (3, 768)
    assert batch["memory_recent_mask"].shape == (3,)
    assert batch["memory_action_embeddings"].shape == (4, 16)
    assert batch["memory_action_mask"].shape == (4,)
    assert batch["memory_proprio_embeddings"].shape == (4, 16)
    assert batch["memory_proprio_mask"].shape == (4,)
    assert batch["memory_language_embedding"].shape == (2048,)


def test_server_rollout_uses_online_memory_and_full_action_chunks():
    first_chunk = np.arange(
        adapter.PREDICTION_HORIZON * adapter.ACTION_WIDTH,
        dtype=np.float32,
    ).reshape(1, adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH)
    second_chunk = first_chunk + 10_000
    policy = FakePolicy(actions=first_chunk)
    conditioner = FakeConditioner()
    stored_language = np.linspace(0.0, 1.0, 2048, dtype=np.float32)
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda prompt: stored_language,
        conditioner=conditioner,
    )
    observation0 = make_observation()
    observation1 = make_observation()

    server.reset_model()
    first_actions = server.eval(
        {"prompt": "put the item away", "observation": observation0}
    )
    server.add_buffer(
        {
            "observations": [observation1],
            "actions": first_actions[:1],
        }
    )
    policy.actions = second_chunk
    second_actions = server.eval(
        {"prompt": "put the item away", "observation": observation1}
    )

    assert first_actions.shape == (adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH)
    assert second_actions.shape == (adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH)
    np.testing.assert_array_equal(first_actions, first_chunk[0])
    np.testing.assert_array_equal(second_actions, second_chunk[0])
    assert policy.reset_count == 2
    assert policy.n_action_steps == [
        adapter.PREDICTION_HORIZON,
        adapter.PREDICTION_HORIZON,
    ]
    assert policy.norm_tags == ["rmbench_stage2", "rmbench_stage2"]
    assert policy.select_action_calls == 0
    assert len(policy.batches) == 2
    for batch in policy.batches:
        assert batch["memory_language_embedding"].shape == (2048,)
        np.testing.assert_array_equal(
            batch["memory_language_embedding"], stored_language
        )
    assert conditioner.devices == ["cpu", "cpu"]
    assert len(conditioner.payload_calls) == 2
    for payload in conditioner.payload_calls:
        assert payload["memory_language_embedding"].shape == (1, 2048)
    assert len(conditioner.reset_languages) == 1
    assert conditioner.reset_languages[0].shape == (2048,)
    np.testing.assert_array_equal(conditioner.reset_languages[0], stored_language)
    assert len(conditioner.appended) == 2
    np.testing.assert_array_equal(
        conditioner.appended[0][2], np.zeros(adapter.ACTION_WIDTH, dtype=np.float32)
    )
    np.testing.assert_array_equal(conditioner.appended[1][2], first_chunk[0, 0])
    assert conditioner.offline_read_count == 0


def test_server_no_memory_uses_native_molmoact2_inputs_only():
    actions = np.zeros((1, 2, adapter.ACTION_WIDTH), dtype=np.float32)
    policy = FakePolicy(actions)
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        memory_enabled=False,
        prediction_horizon=2,
    )
    observation = make_observation()

    result = server.eval(
        {"prompt": "Move The Block", "observation": observation}
    )
    response = server.add_buffer(
        {
            "observations": [observation],
            "actions": result[:1],
        }
    )

    assert result.shape == (2, adapter.ACTION_WIDTH)
    assert response == {"buffered": 1}
    assert policy.reset_count == 1
    assert policy.batches[0]["task"] == "move the block"
    assert not set(adapter._MEMORY_INPUT_KEYS).intersection(policy.batches[0])


def test_server_no_memory_rejects_memory_dependencies():
    with pytest.raises(ValueError, match="must be omitted"):
        adapter.RMBenchMolmoAct2ServerModel(
            FakePolicy(),
            memory_enabled=False,
            language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        )


def test_extract_rmbench_observation_uses_fixed_camera_order_and_native_state():
    frames, state = adapter.extract_rmbench_observation(make_observation())

    assert frames.shape == (3, 256, 256, 3)
    assert frames.dtype == np.uint8
    assert frames[:, 128, 128, 0].tolist() == [2, 1, 0]
    np.testing.assert_array_equal(
        state,
        np.array(
            [0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 0, 6, 13],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        adapter.model_to_native(state),
        np.arange(14, dtype=np.float32),
    )


def test_resize_and_convert_rgb_matches_stage2_bgr_storage_order():
    image = np.empty((4, 6, 3), dtype=np.uint8)
    image[...] = np.array([17, 113, 251], dtype=np.uint8)

    result = adapter.resize_and_convert_rgb(image)

    assert result.dtype == np.uint8
    assert result.shape == (adapter.IMAGE_SIZE, adapter.IMAGE_SIZE, 3)
    np.testing.assert_array_equal(
        result[adapter.IMAGE_SIZE // 2, adapter.IMAGE_SIZE // 2],
        np.array([251, 113, 17], dtype=np.uint8),
    )


def test_server_uses_lowercase_vla_prompt_but_original_language_lookup_prompt():
    prompt = "On The Table, Move The Red Block."
    lookup_prompts = []
    policy = FakePolicy(np.zeros((1, 1, adapter.ACTION_WIDTH), dtype=np.float32))
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda value: (
            lookup_prompts.append(value) or np.ones(2048, dtype=np.float32)
        ),
        conditioner=FakeConditioner(),
        prediction_horizon=1,
    )

    server.eval({"prompt": prompt, "observation": make_observation()})

    assert lookup_prompts == [prompt]
    assert policy.batches[0]["task"] == prompt.lower()


def test_extract_rmbench_observation_identifies_missing_camera():
    with pytest.raises(KeyError, match="right_camera"):
        adapter.extract_rmbench_observation(make_observation(omit_camera="right_camera"))


def test_server_forwards_native_memory_and_buffers_executed_actions():
    policy = FakePolicy()
    conditioner = FakeConditioner()
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=conditioner,
        prediction_horizon=1,
    )
    observation = make_observation()

    actions = server.eval({"prompt": "put the item away", "observation": observation})
    response = server.add_buffer(
        {
            "observations": [observation],
            "actions": np.ones((1, 16), dtype=np.float32),
        }
    )

    assert actions.shape == (1, 16)
    assert response == {"buffered": 1}
    assert policy.reset_count == 1
    assert policy.norm_tags == ["rmbench_stage2"]
    assert policy.n_action_steps == [1]
    assert policy.select_action_calls == 0
    assert len(policy.batches) == 1
    batch = policy.batches[0]
    assert {
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.images.right_image",
    }.issubset(batch)
    assert {
        "memory_cosmo_latents",
        "memory_cosmo_mask",
        "memory_recent_embeddings",
        "memory_recent_mask",
        "memory_action_embeddings",
        "memory_action_mask",
        "memory_proprio_embeddings",
        "memory_proprio_mask",
        "memory_language_embedding",
    }.issubset(batch)
    assert batch["task"] == "put the item away"
    assert conditioner.devices == ["cpu"]
    assert len(conditioner.appended) == 2
    np.testing.assert_array_equal(
        conditioner.appended[0][2], np.zeros(16, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        conditioner.appended[1][2], np.ones(16, dtype=np.float32)
    )
    assert len(conditioner.reset_languages) == 1
    np.testing.assert_array_equal(
        conditioner.reset_languages[0], np.ones(2048, dtype=np.float32)
    )


def test_server_requests_default_prediction_horizon_action_chunk():
    policy = FakePolicy(
        np.zeros(
            (1, adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH),
            dtype=np.float32,
        )
    )
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=FakeConditioner(),
    )

    actions = server.eval({"prompt": "default horizon task", "observation": make_observation()})

    assert actions.shape == (adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH)
    assert policy.norm_tags == ["rmbench_stage2"]
    assert policy.n_action_steps == [adapter.PREDICTION_HORIZON]
    assert policy.select_action_calls == 0


def test_server_reuses_seeded_flow_generator_across_action_chunks(monkeypatch):
    class FakeGenerator:
        def __init__(self, *, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

        def initial_seed(self):
            return self.seed

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator))
    policy = FakePolicy(
        np.zeros(
            (1, adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH),
            dtype=np.float32,
        )
    )
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=FakeConditioner(),
        flow_seed=17,
    )
    observation = make_observation()

    first_actions = server.eval({"prompt": "move", "observation": observation})
    server.add_buffer(
        {
            "observations": [observation],
            "actions": first_actions[:1],
        }
    )
    server.eval({"prompt": "move", "observation": observation})

    assert len(policy.generators) == 2
    assert policy.generators[0] is policy.generators[1]
    assert isinstance(policy.generators[0], FakeGenerator)
    assert policy.generators[0].device == "cpu"
    assert policy.generators[0].initial_seed() == 17


def test_server_seeds_flow_generator_on_loaded_model_device(monkeypatch):
    class FakeGenerator:
        def __init__(self, *, device):
            self.device = device

        def manual_seed(self, seed):
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator))
    policy = FakePolicy(
        np.zeros(
            (1, adapter.PREDICTION_HORIZON, adapter.ACTION_WIDTH),
            dtype=np.float32,
        )
    )
    policy.device = "cpu"
    policy._handles = SimpleNamespace(device="cuda")
    server = adapter.RMBenchMolmoAct2ServerModel(
        policy,
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=FakeConditioner(),
        flow_seed=17,
    )

    server.eval({"prompt": "move", "observation": make_observation()})

    assert policy.generators[0].device == "cuda"


def test_server_rejects_prompt_changes_until_reset():
    server = adapter.RMBenchMolmoAct2ServerModel(
        FakePolicy(),
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=FakeConditioner(),
        prediction_horizon=1,
    )
    observation = make_observation()
    server.eval({"prompt": "first prompt", "observation": observation})

    with pytest.raises(ValueError, match="prompt changed"):
        server.eval({"prompt": "second prompt", "observation": observation})


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros((1, 15), dtype=np.float32), "shape"),
        (np.full((1, 16), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_server_rejects_invalid_buffer_actions(actions, message):
    server = adapter.RMBenchMolmoAct2ServerModel(
        FakePolicy(),
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=FakeConditioner(),
        prediction_horizon=1,
    )
    observation = make_observation()
    server.eval({"prompt": "task", "observation": observation})

    with pytest.raises(ValueError, match=message):
        server.add_buffer({"observations": [observation], "actions": actions})


def test_server_rejects_buffering_before_an_active_rollout():
    server = adapter.RMBenchMolmoAct2ServerModel(
        FakePolicy(),
        language_resolver=lambda prompt: np.ones(2048, dtype=np.float32),
        conditioner=FakeConditioner(),
        prediction_horizon=1,
    )

    with pytest.raises(RuntimeError, match="before active rollout"):
        server.add_buffer(
            {
                "observations": [make_observation()],
                "actions": np.zeros((1, 16), dtype=np.float32),
            }
        )


class FakeTaskEnv:
    def __init__(self):
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 2
        self.taken_actions = []

    def get_instruction(self):
        return "move"

    def take_action(self, action):
        self.taken_actions.append(np.asarray(action).copy())
        self.take_action_cnt += 1

    def get_obs(self):
        return make_observation()


class FakeTCPModel:
    def __init__(self):
        self.calls = []

    def call(self, *, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "eval":
            return np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
        if func_name == "add_buffer":
            return {"buffered": len(obs["observations"])}
        raise AssertionError(f"unexpected callback {func_name}")


def test_eval_callback_executes_actions_and_buffers_once(monkeypatch):
    task_env = FakeTaskEnv()
    model = FakeTCPModel()
    monkeypatch.setenv("RMBENCH_EXECUTION_HORIZON", "1")

    adapter.eval(task_env, model, make_observation())

    assert [call[0] for call in model.calls] == ["eval", "add_buffer"]
    assert len(task_env.taken_actions) == 1
    np.testing.assert_array_equal(
        task_env.taken_actions[0],
        np.array([0, 1, 2, 3, 4, 5, 14, 7, 8, 9, 10, 11, 12, 15], dtype=np.float32),
    )
    buffered = model.calls[1][1]
    assert len(buffered["observations"]) == 1
    np.testing.assert_array_equal(
        buffered["actions"],
        np.stack(
            [
                np.arange(16, dtype=np.float32),
            ]
        ),
    )


def test_eval_callback_rejects_invalid_execution_horizon(monkeypatch):
    monkeypatch.setenv("RMBENCH_EXECUTION_HORIZON", "51")

    with pytest.raises(ValueError, match="RMBENCH_EXECUTION_HORIZON"):
        adapter.eval(FakeTaskEnv(), FakeTCPModel(), make_observation())
