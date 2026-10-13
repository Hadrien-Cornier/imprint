import pytest
import torch

from olmo.nn.action_expert import ActionExpert, ActionExpertConfig, MemoryFiLMRMSNorm
from olmo.models.molmoact2.molmoact2 import MolmoAct2Config


def test_memory_seam_config_is_omegaconf_compatible() -> None:
    omegaconf = pytest.importorskip("omegaconf")

    config = omegaconf.OmegaConf.create({"model": MolmoAct2Config()})

    assert config.model.action_expert.memory_seam_mode == "native"


def test_pi05_memory_seam_rejects_incompatible_attention_geometry() -> None:
    with pytest.raises(ValueError, match="num_heads must be divisible"):
        ActionExpertConfig(
            hidden_size=8,
            num_heads=2,
            memory_enabled=True,
            memory_seam_mode="pi05",
            memory_num_heads=4,
            memory_num_kv_heads=3,
        ).build(llm_dim=8, llm_kv_dim=4)


def test_pi05_memory_seam_rejects_hidden_size_incompatible_with_memory_heads() -> None:
    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        ActionExpertConfig(
            hidden_size=10,
            num_heads=2,
            memory_enabled=True,
            memory_seam_mode="pi05",
            memory_num_heads=4,
            memory_num_kv_heads=1,
        ).build(llm_dim=8, llm_kv_dim=4)


def test_memory_seam_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="memory_seam_mode"):
        ActionExpertConfig(
            hidden_size=8,
            num_heads=2,
            memory_enabled=True,
            memory_seam_mode="typo",  # type: ignore[arg-type]
        ).build(llm_dim=8, llm_kv_dim=4)


def test_pi05_memory_seam_uses_grouped_attention_and_nonidentity_film() -> None:
    expert = _build_action_expert(memory_enabled=True, memory_seam_mode="pi05")
    memory_attn = expert.blocks[0].memory_attn
    memory_film = expert.blocks[0].memory_film

    assert memory_attn is not None
    assert memory_attn.num_heads == 4
    assert memory_attn.num_kv_heads == 1
    assert memory_attn.q_norm is None
    assert memory_attn.k_norm is None
    assert memory_attn.rope is not None
    assert memory_attn.input_norm is not None
    assert memory_film is not None

    x = torch.randn(2, 3, 8)
    memory_read = torch.randn(2, 3, 8)
    assert not torch.equal(memory_film(x, memory_read), memory_film.norm(x))


def test_pi05_memory_film_is_composed_through_ff_norm_before_mlp() -> None:
    class FixedMemoryAttention(torch.nn.Module):
        def forward(self, x, *, kv, attn_mask=None):
            return torch.zeros_like(x)

    class OffsetFilm(torch.nn.Module):
        def forward(self, x, memory_read):
            return x + 3

    class RecordingNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None

        def forward(self, x):
            self.input = x
            return x + 5

    class RecordingMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None

        def forward(self, x):
            self.input = x
            return torch.zeros_like(x)

    expert = _build_action_expert(memory_enabled=True, memory_seam_mode="pi05")
    block = expert.blocks[0]
    block.memory_attn = FixedMemoryAttention()
    block.memory_film = OffsetFilm()
    block.ff_norm = RecordingNorm()
    block.mlp = RecordingMLP()
    with torch.no_grad():
        block.modulation.linear.bias[8 * expert.hidden_size : 9 * expert.hidden_size].fill_(1)

    x = torch.randn(2, 3, expert.hidden_size)
    block(
        x,
        torch.zeros(2, expert.hidden_size),
        cross_context=torch.randn(2, 4, expert.hidden_size),
        memory_tokens=torch.randn(2, 5, expert.hidden_size),
    )

    torch.testing.assert_close(block.ff_norm.input, x + 3)
    torch.testing.assert_close(block.mlp.input, x + 8)


def test_pi05_checkpoint_synthesis_uses_configured_film_std_and_lecun_attention() -> None:
    pi05 = _build_action_expert(memory_enabled=True, memory_seam_mode="pi05")
    pi05.config.memory_film_init_std = 0.0
    baseline = _build_action_expert(memory_enabled=False)
    state = {f"action_expert.{name}": value.clone() for name, value in baseline.state_dict().items()}

    pi05.prepare_state_dict_for_loading(state, prefix="action_expert.")

    film_weight = state["action_expert.blocks.0.memory_film.to_scale_shift.weight"]
    out_weight = state["action_expert.blocks.0.memory_attn.out_proj.weight"]
    assert torch.equal(film_weight, torch.zeros_like(film_weight))
    assert 0.2 < out_weight.std().item() < 0.5


def test_pi05_strictly_loads_native_memory_state_after_preparation() -> None:
    native = _build_action_expert(memory_enabled=True)
    pi05 = _build_action_expert(memory_enabled=True, memory_seam_mode="pi05")
    pi05.config.memory_film_init_std = 0.0
    state = {f"action_expert.{name}": value.clone() for name, value in native.state_dict().items()}
    native_q = state["action_expert.blocks.0.memory_attn.q_proj.weight"].clone()
    native_out = state["action_expert.blocks.0.memory_attn.out_proj.weight"].clone()

    pi05.prepare_state_dict_for_loading(state, prefix="action_expert.")
    pi05.load_state_dict(
        {name.removeprefix("action_expert."): value for name, value in state.items()},
        strict=True,
    )
    assert torch.equal(
        state["action_expert.blocks.0.memory_film.to_scale_shift.weight"],
        torch.zeros_like(state["action_expert.blocks.0.memory_film.to_scale_shift.weight"]),
    )
    assert not torch.equal(state["action_expert.blocks.0.memory_attn.q_proj.weight"], native_q)
    assert not torch.equal(state["action_expert.blocks.0.memory_attn.out_proj.weight"], native_out)


def test_pi05_memory_attention_supports_masked_forward_and_backward() -> None:
    expert = _build_action_expert(memory_enabled=True, memory_seam_mode="pi05")
    with torch.no_grad():
        expert.blocks[0].modulation.linear.bias[
            8 * expert.hidden_size : 9 * expert.hidden_size
        ].fill_(1)
        expert.final_layer.linear.weight.normal_(mean=0.0, std=0.02)
    action_inputs = _action_inputs()
    memory_tokens = torch.randn(2, 5, expert.hidden_size)
    memory_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]],
        dtype=torch.bool,
    )

    output = expert(
        **action_inputs,
        memory_tokens=memory_tokens,
        memory_attention_mask=memory_mask,
    )
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    memory_attn = expert.blocks[0].memory_attn
    assert memory_attn is not None
    for parameter in (
        memory_attn.q_proj.weight,
        memory_attn.kv_proj.weight,
        memory_attn.out_proj.weight,
        memory_attn.input_norm.scale,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def _build_action_expert(
    *,
    memory_enabled: bool,
    memory_seam_mode: str = "native",
) -> ActionExpert:
    config = ActionExpertConfig(
        max_horizon=2,
        max_action_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=1.0,
        ffn_multiple_of=8,
        timestep_embed_dim=4,
        compile=None,
        memory_enabled=memory_enabled,
        memory_seam_mode=memory_seam_mode,  # type: ignore[arg-type]
    )
    return config.build(llm_dim=8, llm_kv_dim=4)


def _action_inputs() -> dict[str, object]:
    torch.manual_seed(3)
    return {
        "actions": torch.randn(2, 2, 3),
        "timesteps": torch.tensor([0.1, 0.7]),
        "encoder_kv_states": [(torch.randn(2, 3, 4), torch.randn(2, 3, 4))],
        "encoder_attention_mask": torch.ones(2, 3, dtype=torch.bool),
    }


def _make_memory_seam_non_identity(expert: ActionExpert) -> None:
    block = expert.blocks[0]
    assert block.memory_attn is not None
    assert block.memory_film is not None
    with torch.no_grad():
        hidden_size = expert.hidden_size
        block.modulation.linear.bias[8 * hidden_size : 9 * hidden_size].fill_(1.0)
        block.memory_attn.q_proj.weight.zero_()
        block.memory_attn.q_proj.bias.zero_()
        block.memory_attn.kv_proj.weight.zero_()
        block.memory_attn.kv_proj.bias.zero_()
        block.memory_attn.kv_proj.weight[hidden_size:].copy_(torch.eye(hidden_size))
        block.memory_attn.out_proj.weight.copy_(torch.eye(hidden_size))
        block.memory_attn.out_proj.bias.zero_()
        block.memory_film.to_scale_shift.weight.fill_(0.1)
        block.memory_film.to_scale_shift.bias.zero_()
        expert.final_layer.linear.weight.fill_(0.1)


def test_identity_initialized_memory_seam_matches_baseline_action_expert() -> None:
    baseline = _build_action_expert(memory_enabled=False).eval()
    memory = _build_action_expert(memory_enabled=True).eval()
    memory.load_state_dict(baseline.state_dict(), strict=False)
    action_inputs = _action_inputs()

    expected = baseline(**action_inputs)
    actual = memory(
        **action_inputs,
        memory_tokens=torch.randn(2, 5, 8),
        memory_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    torch.testing.assert_close(actual, expected)


def test_memory_tokens_change_action_expert_output_after_memory_seam_is_non_identity() -> None:
    expert = _build_action_expert(memory_enabled=True).eval()
    _make_memory_seam_non_identity(expert)
    action_inputs = _action_inputs()

    first = expert(
        **action_inputs,
        memory_tokens=torch.zeros(2, 5, 8),
        memory_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    second = expert(
        **action_inputs,
        memory_tokens=torch.ones(2, 5, 8),
        memory_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    assert not torch.equal(first, second)


def test_memory_tokens_are_converted_to_action_expert_dtype() -> None:
    expert = _build_action_expert(memory_enabled=True).eval()
    action_inputs = _action_inputs()
    memory_tokens = torch.randn(2, 5, 8)
    memory_mask = torch.ones(2, 5, dtype=torch.bool)

    expected = expert(
        **action_inputs,
        memory_tokens=memory_tokens,
        memory_attention_mask=memory_mask,
    )
    actual = expert(
        **action_inputs,
        memory_tokens=memory_tokens.to(torch.float64),
        memory_attention_mask=memory_mask,
    )

    torch.testing.assert_close(actual, expected)


def test_memory_adapter_projects_memory_inside_action_expert() -> None:
    expert = _build_action_expert(memory_enabled=True)
    expert.memory_adapter = torch.nn.Linear(10, expert.hidden_size)
    _make_memory_seam_non_identity(expert)
    action_inputs = _action_inputs()
    memory_tokens = torch.randn(2, 5, 10, requires_grad=True)

    output = expert(
        **action_inputs,
        memory_tokens=memory_tokens,
        memory_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    output.square().mean().backward()

    assert memory_tokens.grad is not None
    assert expert.memory_adapter.weight.grad is not None
    assert torch.count_nonzero(expert.memory_adapter.weight.grad) > 0


def test_identity_memory_film_preserves_forward_values_and_memory_gradients() -> None:
    film = MemoryFiLMRMSNorm(hidden_size=8)
    x = torch.randn(2, 3, 8)
    memory_read = torch.randn(2, 3, 8, requires_grad=True)

    assert not film.reference_weight.requires_grad
    assert not film.reference_bias.requires_grad

    actual = film(x, memory_read)

    torch.testing.assert_close(actual, film.norm(x))
    (actual - torch.ones_like(actual)).square().mean().backward()

    assert memory_read.grad is not None
    assert torch.count_nonzero(memory_read.grad) > 0
    assert film.to_scale_shift.weight.grad is not None
    assert torch.count_nonzero(film.to_scale_shift.weight.grad) > 0


def test_legacy_checkpoint_initializes_trainable_identity_memory_seam_after_to_empty() -> None:
    baseline = _build_action_expert(memory_enabled=False).eval()
    memory = _build_action_expert(memory_enabled=True).eval()
    memory.to_empty(device="cpu")
    with torch.no_grad():
        for name, value in memory.state_dict().items():
            if ".memory_" in name:
                value.fill_(1)

    checkpoint_state = {
        f"action_expert.{name}": value.detach().clone()
        for name, value in baseline.state_dict().items()
    }
    memory.prepare_state_dict_for_loading(checkpoint_state, prefix="action_expert.")

    memory_state = {
        key.removeprefix("action_expert."): value
        for key, value in checkpoint_state.items()
    }
    restored = _build_action_expert(memory_enabled=True).eval()
    restored.load_state_dict(memory_state, strict=True)

    assert torch.count_nonzero(
        checkpoint_state["action_expert.blocks.0.memory_attn.q_proj.weight"]
    ) > 0
    assert torch.count_nonzero(
        checkpoint_state["action_expert.blocks.0.memory_attn.out_proj.weight"]
    ) > 0
    torch.testing.assert_close(
        checkpoint_state["action_expert.blocks.0.memory_film.to_scale_shift.weight"],
        checkpoint_state["action_expert.blocks.0.memory_film.reference_weight"],
    )
    torch.testing.assert_close(
        checkpoint_state["action_expert.blocks.0.memory_film.to_scale_shift.bias"],
        checkpoint_state["action_expert.blocks.0.memory_film.reference_bias"],
    )

    action_inputs = _action_inputs()
    expected = baseline(**action_inputs)
    actual = restored(
        **action_inputs,
        memory_tokens=torch.randn(2, 5, 8),
        memory_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    torch.testing.assert_close(actual, expected)


def test_legacy_checkpoint_initializes_missing_memory_adapter() -> None:
    baseline = _build_action_expert(memory_enabled=False).eval()
    memory = _build_action_expert(memory_enabled=True).eval()
    memory.memory_adapter = torch.nn.Linear(10, memory.hidden_size)
    memory.to_empty(device="cpu")
    checkpoint_state = {
        f"action_expert.{name}": value.detach().clone()
        for name, value in baseline.state_dict().items()
    }

    memory.prepare_state_dict_for_loading(checkpoint_state, prefix="action_expert.")

    weight = checkpoint_state["action_expert.memory_adapter.weight"]
    bias = checkpoint_state["action_expert.memory_adapter.bias"]
    assert torch.isfinite(weight).all()
    assert torch.count_nonzero(weight) > 0
    assert torch.equal(bias, torch.zeros_like(bias))


def test_legacy_memory_seam_receives_action_loss_gradients() -> None:
    baseline = _build_action_expert(memory_enabled=False)
    memory = _build_action_expert(memory_enabled=True)
    memory.memory_adapter = torch.nn.Linear(10, memory.hidden_size)
    with torch.no_grad():
        baseline.final_layer.linear.weight.normal_(mean=0.0, std=0.02)
        baseline.blocks[0].modulation.linear.bias[
            8 * baseline.hidden_size : 9 * baseline.hidden_size
        ].fill_(1.0)

    checkpoint_state = {
        f"action_expert.{name}": value.detach().clone()
        for name, value in baseline.state_dict().items()
    }
    memory.prepare_state_dict_for_loading(checkpoint_state, prefix="action_expert.")
    restored = _build_action_expert(memory_enabled=True)
    restored.memory_adapter = torch.nn.Linear(10, restored.hidden_size)
    restored.load_state_dict(
        {
            key.removeprefix("action_expert."): value
            for key, value in checkpoint_state.items()
        },
        strict=True,
    )

    output = restored(
        **_action_inputs(),
        memory_tokens=torch.randn(2, 5, 10),
        memory_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    (output - torch.ones_like(output)).square().mean().backward()

    block = restored.blocks[0]
    assert restored.memory_adapter is not None
    assert torch.count_nonzero(restored.memory_adapter.weight.grad) > 0
    assert block.memory_attn is not None
    assert torch.count_nonzero(block.memory_attn.q_proj.weight.grad) > 0
    assert torch.count_nonzero(block.memory_attn.kv_proj.weight.grad) > 0
    assert torch.count_nonzero(block.memory_attn.out_proj.weight.grad) > 0
    assert block.memory_film is not None
    assert torch.count_nonzero(block.memory_film.to_scale_shift.weight.grad) > 0
