from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy


def test_resolve_style_for_inference_rejects_depth_style_when_disabled():
    handles = SimpleNamespace(
        default_style="robot_depth_action",
        enable_depth_reasoning=False,
        depth_start_token_id=1,
        depth_end_token_id=2,
        depth_token_id_to_bin={3: 0},
    )

    with pytest.raises(ValueError, match="enable_depth_reasoning"):
        MolmoAct2Policy._resolve_style_for_inference(handles)


class _NativeModel:
    def __init__(self) -> None:
        self.action_inputs = None
        self.generation_batches = []

    def generate(self, batch, **kwargs):
        self.generation_batches.append(batch)
        return SimpleNamespace(
            token_ids=torch.tensor([[[10, 11, 12]]]),
            attn_key_values=object(),
        )

    def generate_actions(self, **kwargs):
        self.action_inputs = kwargs
        return torch.zeros(1, 2, 3)

    def _get_encoder_attention_mask(self, input_ids, attention_mask):
        return None


@pytest.mark.parametrize(
    ("style", "uses_depth"),
    [
        ("robot_action", False),
        ("robot_depth_action", True),
    ],
)
def test_continuous_action_generation_forwards_native_memory_inputs(style, uses_depth):
    native_model = _NativeModel()
    handles = SimpleNamespace(
        model=native_model,
        default_style=style,
        enable_depth_reasoning=uses_depth,
        depth_start_token_id=10,
        depth_end_token_id=12,
        depth_token_id_to_bin={11: 0},
        inference_action_mode="continuous",
        num_steps=2,
        max_action_dim=3,
        device=torch.device("cpu"),
        robot_processor=None,
    )
    collated = {
        "input_ids": torch.tensor([[1, 2]]),
        "states": torch.zeros(1, 3),
        "memory_cosmo_latents": torch.ones(1, 2, 4),
        "memory_recent_embeddings": torch.ones(1, 2, 4),
        "memory_action_embeddings": torch.ones(1, 2, 3),
        "memory_proprio_embeddings": torch.ones(1, 2, 3),
        "memory_cosmo_mask": torch.ones(1, 2, dtype=torch.bool),
        "memory_recent_mask": torch.ones(1, 2, dtype=torch.bool),
        "memory_action_mask": torch.ones(1, 2, dtype=torch.bool),
        "memory_proprio_mask": torch.ones(1, 2, dtype=torch.bool),
        "memory_language_embedding": torch.ones(1, 4),
        "memory_tokens": torch.ones(1, 2, 4),
        "memory_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    expected_memory = {
        key: value
        for key, value in MolmoAct2Policy._build_action_model_inputs(collated).items()
        if key.startswith("memory_")
    }

    policy = object.__new__(MolmoAct2Policy)
    policy._generate_inference_result(collated, handles)

    assert native_model.action_inputs is not None
    assert set(native_model.action_inputs).intersection(expected_memory) == set(expected_memory)
    for key, value in expected_memory.items():
        assert native_model.action_inputs[key] is value
    if uses_depth:
        assert len(native_model.generation_batches) == 1
        assert not set(native_model.generation_batches[0]).intersection(expected_memory)
