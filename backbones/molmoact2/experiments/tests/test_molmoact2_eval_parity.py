from __future__ import annotations

from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("draccus")
from lerobot.policies.molmoact2.modeling_molmoact2 import _disable_inference_token_bias
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.hf_backend import MolmoAct2HFBackend


def test_hf_policy_preserves_native_observation_image_order():
    first = np.full((2, 2, 3), 1, dtype=np.uint8)
    second = np.full((2, 2, 3), 2, dtype=np.uint8)
    obs = {
        "observation.images.image2": first,
        "observation.images.image": second,
    }

    images = MolmoAct2HFBackend._extract_images(obs)

    assert [int(image[0, 0, 0]) for image in images] == [1, 2]


def test_disable_inference_token_bias_removes_legacy_nonpersistent_bias():
    token_bias = object()
    model = SimpleNamespace(transformer=SimpleNamespace(token_bias=token_bias))

    assert _disable_inference_token_bias(model)
    assert model.transformer.token_bias is None
    assert not _disable_inference_token_bias(model)


class _RecordingRobotProcessor:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.metadata_by_tag = {tag: object()}
        self.calls = []

    def resolve_tag(self, tag: str) -> str:
        return tag

    def get_action_dim(self, tag: str) -> int:
        assert tag == self.tag
        return 16

    def unnormalize_action(self, actions, *, repo_id: str):
        self.calls.append((actions.detach().clone(), repo_id))
        return actions + 3.0


def test_local_predict_action_chunk_unnormalizes_the_complete_tagged_chunk():
    norm_tag = "rmbench_stage2"
    raw_chunk = torch.arange(50 * 16, dtype=torch.float32).reshape(1, 50, 16)
    robot_processor = _RecordingRobotProcessor(norm_tag)
    policy = object.__new__(MolmoAct2Policy)
    object.__setattr__(
        policy,
        "_handles",
        SimpleNamespace(
            n_obs_steps=1,
            norm_tag=norm_tag,
            robot_processor=robot_processor,
            max_action_dim=16,
            device=torch.device("cpu"),
        ),
    )
    object.__setattr__(policy, "_hf_backend", None)
    object.__setattr__(policy, "_obs_history", defaultdict(deque))
    object.__setattr__(policy, "_last_depth_video_codes_by_batch", {})
    object.__setattr__(policy, "_last_model_inference_s", 0.0)
    object.__setattr__(policy, "_last_model_inference_calls", 0)

    def generate_chunk(observations, **kwargs):
        assert len(observations) == 1
        assert kwargs["norm_tag"] == norm_tag
        assert kwargs["n_action_steps"] == 50
        return SimpleNamespace(
            style="robot_action",
            actions=raw_chunk.clone(),
            depth_bins=None,
        )

    object.__setattr__(
        policy,
        "generate_inference_result_from_observations",
        generate_chunk,
    )

    actions = policy.predict_action_chunk(
        {"observation.state": torch.zeros(16)},
        norm_tag=norm_tag,
        n_action_steps=50,
    )

    assert len(robot_processor.calls) == 1
    normalized_actions, repo_id = robot_processor.calls[0]
    assert repo_id == norm_tag
    assert normalized_actions.shape == (1, 50, 16)
    assert torch.equal(normalized_actions, raw_chunk)
    assert torch.equal(actions, raw_chunk + 3.0)
