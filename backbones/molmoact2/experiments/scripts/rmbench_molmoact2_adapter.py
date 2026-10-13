"""Raw-TCP RMBench adapter for MolmoAct2 online-memory rollouts."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")
ACTION_WIDTH = 16
NATIVE_ACTION_WIDTH = 14
PREDICTION_HORIZON = 50
IMAGE_SIZE = 256

_MEMORY_INPUT_KEYS = (
    "memory_cosmo_latents",
    "memory_cosmo_mask",
    "memory_recent_embeddings",
    "memory_recent_mask",
    "memory_action_embeddings",
    "memory_action_mask",
    "memory_proprio_embeddings",
    "memory_proprio_mask",
    "memory_language_embedding",
)


def get_model(_usr_args: Mapping) -> None:
    """Satisfy the official client's vestigial policy-factory lookup.

    The client creates its TCP ``ModelClient`` directly from ``port`` and never
    calls this hook.
    """
    return None


def _float32_vector(value: Any, *, name: str, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must have a numeric dtype, got {array.dtype}")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def native_to_model(native_state: np.ndarray) -> np.ndarray:
    """Map RMBench's native 14-D state layout to MolmoAct2's 16-D layout."""
    native_state = _float32_vector(
        native_state,
        name="native state",
        width=NATIVE_ACTION_WIDTH,
    )
    model_state = np.zeros((ACTION_WIDTH,), dtype=np.float32)
    model_state[:6] = native_state[:6]
    model_state[7:13] = native_state[7:13]
    model_state[14] = native_state[6]
    model_state[15] = native_state[13]
    return model_state


def model_to_native(model_action: np.ndarray) -> np.ndarray:
    """Map a MolmoAct2 16-D action into RMBench's native 14-D layout."""
    model_action = _float32_vector(
        model_action,
        name="model action",
        width=ACTION_WIDTH,
    )
    return np.concatenate(
        (
            model_action[:6],
            model_action[14:15],
            model_action[7:13],
            model_action[15:16],
        )
    ).astype(np.float32, copy=False)


def resize_with_pad(
    image: np.ndarray,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> np.ndarray:
    """Resize an RGB frame using RMBench's aspect-ratio-preserving pad rule."""
    import cv2

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC image, got {image.shape} {image.dtype}")
    if height < 1 or width < 1:
        raise ValueError("resize dimensions must be positive")

    source_height, source_width = image.shape[:2]
    if source_height < 1 or source_width < 1:
        raise ValueError(f"expected non-empty HWC image, got {image.shape}")
    ratio = max(source_width / width, source_height / height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    pad_top, remainder_height = divmod(height - resized_height, 2)
    pad_left, remainder_width = divmod(width - resized_width, 2)
    pad_bottom = pad_top + remainder_height
    pad_right = pad_left + remainder_width
    output[pad_top : height - pad_bottom, pad_left : width - pad_right] = resized
    return output


def resize_and_convert_rgb(image: np.ndarray) -> np.ndarray:
    """Match Stage-2's BGR storage convention for one rendered RGB frame."""
    return np.ascontiguousarray(resize_with_pad(image)[..., ::-1])


def extract_rmbench_observation(observation: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-order RGB camera frames and a 16-D MolmoAct2 state."""
    if not isinstance(observation, Mapping):
        raise TypeError(
            "RMBench observation must be a mapping, got "
            f"{type(observation).__name__}"
        )
    try:
        cameras = observation["observation"]
    except KeyError as exc:
        raise KeyError("RMBench observation is missing 'observation' cameras") from exc
    if not isinstance(cameras, Mapping):
        raise TypeError("RMBench observation['observation'] must be a camera mapping")

    frames = []
    for camera_name in CAMERA_ORDER:
        try:
            camera = cameras[camera_name]
        except KeyError as exc:
            raise KeyError(f"RMBench observation is missing camera {camera_name!r}") from exc
        if not isinstance(camera, Mapping):
            raise TypeError(f"RMBench camera {camera_name!r} must be a mapping")
        try:
            frame = camera["rgb"]
        except KeyError as exc:
            raise KeyError(f"RMBench camera {camera_name!r} is missing 'rgb'") from exc
        frames.append(resize_and_convert_rgb(frame))

    try:
        native_state = observation["joint_action"]["vector"]
    except KeyError as exc:
        raise KeyError(
            "RMBench observation is missing joint_action.vector"
        ) from exc
    return np.stack(frames, axis=0), native_to_model(native_state)


def _resolve_language_embedding(language_resolver, prompt: str) -> np.ndarray:
    if callable(language_resolver):
        embedding = language_resolver(prompt)
    else:
        resolve = getattr(language_resolver, "resolve", None)
        if not callable(resolve):
            raise TypeError(
                "language_resolver must be callable or define resolve(prompt)"
            )
        embedding = resolve(prompt)
    return _float32_vector(
        embedding,
        name="stored Pi0.5 language embedding",
        width=2048,
    ).copy()


def _policy_device(policy) -> Any:
    device = getattr(policy, "device", None)
    if device is not None:
        return device

    parameters = getattr(policy, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(parameters())
        except StopIteration:
            pass
        else:
            parameter_device = getattr(parameter, "device", None)
            if parameter_device is not None:
                return parameter_device
    return "cpu"


def _flow_generator_device(policy) -> Any:
    """Resolve the device used by the wrapped action model's random trajectory."""
    handles = getattr(policy, "_handles", None)
    handles_device = getattr(handles, "device", None)
    if handles_device is not None:
        return handles_device
    return _policy_device(policy)


def _as_action_array(result: Any) -> np.ndarray:
    if isinstance(result, Mapping):
        for key in ("actions", "action", "pred_actions"):
            if key in result:
                result = result[key]
                break
        else:
            raise ValueError(
                "action dictionary must contain one of 'actions', 'action', or "
                "'pred_actions'"
            )

    detach = getattr(result, "detach", None)
    if callable(detach):
        result = detach()
    cpu = getattr(result, "cpu", None)
    if callable(cpu):
        result = cpu()
    numpy = getattr(result, "numpy", None)
    if callable(numpy):
        result = numpy()

    actions = np.asarray(result)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1:] != (ACTION_WIDTH,):
        raise ValueError(
            "policy actions must have shape [T, "
            f"{ACTION_WIDTH}], got {actions.shape}"
        )
    if not np.issubdtype(actions.dtype, np.number):
        raise ValueError(f"policy actions must have a numeric dtype, got {actions.dtype}")
    actions = np.asarray(actions, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("policy returned non-finite actions")
    return actions


def _execution_horizon() -> int:
    raw_value = os.environ.get("RMBENCH_EXECUTION_HORIZON")
    if raw_value is None:
        return PREDICTION_HORIZON
    try:
        horizon = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "RMBENCH_EXECUTION_HORIZON must be an integer in "
            f"[1, {PREDICTION_HORIZON}], got {raw_value!r}"
        ) from exc
    if not 1 <= horizon <= PREDICTION_HORIZON:
        raise ValueError(
            "RMBENCH_EXECUTION_HORIZON must be in "
            f"[1, {PREDICTION_HORIZON}], got {horizon}"
        )
    return horizon


def _as_single_example_memory_inputs(memory_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the online conditioner's batch-of-one tensors into dataset examples."""
    result = dict(memory_inputs)
    for key in _MEMORY_INPUT_KEYS:
        value = result[key]
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        if len(shape) < 1 or shape[0] != 1:
            raise ValueError(
                f"{key} must be a batch-of-one tensor before policy collation, "
                f"got shape {tuple(shape)}"
            )
        result[key] = value[0]
    return result


class RMBenchMolmoAct2ServerModel:
    """Stateful MolmoAct2 policy exposed through RMBench's raw TCP transport."""

    def __init__(
        self,
        policy,
        *,
        language_resolver=None,
        conditioner=None,
        conditioner_factory: Callable[[], Any] | None = None,
        memory_enabled: bool = True,
        prediction_horizon: int = PREDICTION_HORIZON,
        norm_tag: str = "rmbench_stage2",
        flow_seed: int | None = None,
    ) -> None:
        if not isinstance(memory_enabled, bool):
            raise TypeError("memory_enabled must be a boolean")
        if conditioner is not None and conditioner_factory is not None:
            raise ValueError("provide either conditioner or conditioner_factory, not both")
        if memory_enabled and (conditioner is None and conditioner_factory is None):
            raise ValueError("conditioner or conditioner_factory is required")
        if memory_enabled and language_resolver is None:
            raise ValueError("language_resolver is required when memory is enabled")
        if not memory_enabled and (
            language_resolver is not None
            or conditioner is not None
            or conditioner_factory is not None
        ):
            raise ValueError(
                "language_resolver and conditioner must be omitted when memory is disabled"
            )
        if conditioner_factory is not None and not callable(conditioner_factory):
            raise TypeError("conditioner_factory must be callable")
        if (
            isinstance(prediction_horizon, bool)
            or not isinstance(prediction_horizon, int)
            or prediction_horizon < 1
        ):
            raise ValueError("prediction_horizon must be a positive integer")
        if not isinstance(norm_tag, str) or not norm_tag:
            raise ValueError("norm_tag must be a non-empty string")
        if flow_seed is not None and (
            isinstance(flow_seed, bool)
            or not isinstance(flow_seed, int)
            or flow_seed < 0
        ):
            raise ValueError("flow_seed must be a non-negative integer or None")

        self.policy = policy
        self.memory_enabled = memory_enabled
        self.language_resolver = language_resolver
        self.conditioner = conditioner
        self._conditioner_factory = conditioner_factory
        self.prediction_horizon = prediction_horizon
        self.norm_tag = norm_tag
        self.flow_seed = flow_seed
        self._episode_active = False
        self._prompt: str | None = None
        self._flow_generator = None

    def _get_conditioner(self):
        if not self.memory_enabled:
            raise RuntimeError("memory conditioner requested while memory is disabled")
        if self.conditioner is None:
            assert self._conditioner_factory is not None
            self.conditioner = self._conditioner_factory()
        return self.conditioner

    def reset_model(self) -> None:
        self.policy.reset()
        self._episode_active = False
        self._prompt = None
        self._flow_generator = None

    def _reset_flow_generator(self) -> None:
        self._flow_generator = None
        if self.flow_seed is None:
            return
        import torch

        self._flow_generator = torch.Generator(
            device=_flow_generator_device(self.policy)
        )
        self._flow_generator.manual_seed(self.flow_seed)

    def _start_episode(self, prompt: str, observation: Mapping) -> None:
        self.policy.reset()
        self._reset_flow_generator()
        if self.memory_enabled:
            frames, state = extract_rmbench_observation(observation)
            language_embedding = _resolve_language_embedding(
                self.language_resolver, prompt
            )
            conditioner = self._get_conditioner()
            reset = getattr(conditioner, "reset", None)
            append = getattr(conditioner, "append", None)
            if not callable(reset) or not callable(append):
                raise TypeError(
                    "conditioner must define reset(language_embedding) and append(...)"
                )
            reset(language_embedding)
            append(
                frames,
                state,
                np.zeros((ACTION_WIDTH,), dtype=np.float32),
            )
        self._episode_active = True
        self._prompt = prompt

    def _predict(self, observation: Mapping) -> np.ndarray:
        frames, state = extract_rmbench_observation(observation)
        batch = {
            "observation.images.image": frames[0],
            "observation.images.wrist_image": frames[1],
            "observation.images.right_image": frames[2],
            "observation.state": state,
            "task": self._prompt.lower(),
        }
        if self.memory_enabled:
            conditioner = self._get_conditioner()
            as_model_inputs = getattr(conditioner, "as_model_inputs", None)
            if not callable(as_model_inputs):
                raise TypeError("conditioner must define as_model_inputs(device=...)")
            memory_inputs = as_model_inputs(device=_policy_device(self.policy))
            if not isinstance(memory_inputs, Mapping):
                raise TypeError("conditioner.as_model_inputs() must return a mapping")
            missing_memory_inputs = [
                key for key in _MEMORY_INPUT_KEYS if key not in memory_inputs
            ]
            if missing_memory_inputs:
                raise ValueError(
                    "conditioner did not provide required native memory inputs: "
                    f"{missing_memory_inputs}"
                )
            batch.update(_as_single_example_memory_inputs(memory_inputs))
        actions = _as_action_array(
            self.policy.predict_action_chunk(
                batch,
                norm_tag=self.norm_tag,
                n_action_steps=self.prediction_horizon,
                generator=self._flow_generator,
            )
        )
        expected_shape = (self.prediction_horizon, ACTION_WIDTH)
        if actions.shape != expected_shape:
            raise ValueError(
                f"expected predicted actions {expected_shape}, got {actions.shape}"
            )
        return actions

    def eval(self, payload: Mapping) -> np.ndarray:
        if not isinstance(payload, Mapping):
            raise TypeError("eval payload must be a mapping")
        try:
            prompt = payload["prompt"]
            observation = payload["observation"]
        except KeyError as exc:
            raise KeyError("eval payload requires prompt and observation") from exc
        if not isinstance(prompt, str):
            raise TypeError(f"eval prompt must be a string, got {type(prompt).__name__}")
        if not self._episode_active:
            self._start_episode(prompt, observation)
        elif prompt != self._prompt:
            raise ValueError("episode prompt changed without reset_model")
        return self._predict(observation)

    def add_buffer(self, payload: Mapping) -> dict[str, int]:
        if not self._episode_active:
            raise RuntimeError("add_buffer called before active rollout")
        if not isinstance(payload, Mapping):
            raise TypeError("add_buffer payload must be a mapping")
        try:
            observations = payload["observations"]
            actions = payload["actions"]
        except KeyError as exc:
            raise KeyError("add_buffer payload requires observations and actions") from exc
        if not isinstance(observations, Sequence) or isinstance(
            observations, (str, bytes)
        ):
            raise TypeError("add_buffer observations must be a sequence")

        action_array = np.asarray(actions)
        expected_shape = (len(observations), ACTION_WIDTH)
        if action_array.shape != expected_shape:
            raise ValueError(
                f"expected actions with shape {expected_shape}, got {action_array.shape}"
            )
        if not np.issubdtype(action_array.dtype, np.number):
            raise ValueError(
                f"executed actions must have a numeric dtype, got {action_array.dtype}"
            )
        action_array = np.asarray(action_array, dtype=np.float32)
        if not np.isfinite(action_array).all():
            raise ValueError("executed actions contain non-finite values")

        if self.memory_enabled:
            conditioner = self._get_conditioner()
            for observation, action in zip(observations, action_array, strict=True):
                frames, state = extract_rmbench_observation(observation)
                conditioner.append(frames, state, action)
        return {"buffered": len(observations)}


def eval(task_env, model, observation) -> None:
    """Run one official RMBench raw-TCP action chunk and buffer live observations."""
    actions = _as_action_array(
        model.call(
            func_name="eval",
            obs={
                "prompt": task_env.get_instruction(),
                "observation": observation,
            },
        )
    )
    if actions.shape != (PREDICTION_HORIZON, ACTION_WIDTH):
        raise ValueError(
            "expected predicted actions "
            f"({PREDICTION_HORIZON}, {ACTION_WIDTH}), got {actions.shape}"
        )

    buffered_observations = []
    buffered_actions = []
    for model_action in actions[:_execution_horizon()]:
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            break
        task_env.take_action(model_to_native(model_action))
        buffered_observations.append(task_env.get_obs())
        buffered_actions.append(model_action)

    if buffered_observations:
        response = model.call(
            func_name="add_buffer",
            obs={
                "observations": buffered_observations,
                "actions": np.stack(buffered_actions),
            },
        )
        if not isinstance(response, Mapping) or response.get("buffered") != len(
            buffered_observations
        ):
            raise RuntimeError(
                f"policy server did not buffer the action chunk: {response}"
            )


def reset_model(model) -> None:
    """Compatibility callback used by RMBench's raw TCP evaluator."""
    model.call(func_name="reset_model")
