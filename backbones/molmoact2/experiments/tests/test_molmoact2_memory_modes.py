from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from olmo.models.model import OLMoOutput
from olmo.models.molmoact2.molmoact2 import MolmoAct2, MolmoAct2Config
from olmo.nn.memory_bottleneck import (
    MemoryBottleneckConfig,
    MemoryBottleneckEncoder,
)
from olmo.train.checkpoint_loading import (
    load_unsharded_checkpoint_allowing_missing_action_expert,
)


def test_molmoact2_config_exposes_memory_modes() -> None:
    config = MolmoAct2Config()

    assert config.memory.enabled is False
    assert config.memory.mode == "frozen"


def _memory_config(mode: str) -> MemoryBottleneckConfig:
    return MemoryBottleneckConfig(
        enabled=True,
        mode=mode,
        num_latents=2,
        memory_dim=8,
        cosmos_latent_dim=4,
        recent_dim=4,
        action_dim=3,
        proprio_dim=3,
        language_dim=4,
        tokenizer_hidden_dim=4,
        num_cross_blocks=1,
        num_heads=2,
    )


def _memory_kwargs(config: MemoryBottleneckConfig, batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "memory_cosmo_latents": torch.randn(batch_size, 2, config.cosmos_latent_dim),
        "memory_recent_embeddings": torch.randn(batch_size, 2, config.recent_dim),
        "memory_action_embeddings": torch.randn(batch_size, 2, config.action_dim),
        "memory_proprio_embeddings": torch.randn(batch_size, 2, config.proprio_dim),
        "memory_cosmo_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "memory_recent_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "memory_action_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "memory_proprio_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "memory_language_embedding": torch.randn(batch_size, config.language_dim),
        "memory_tokens": None,
        "memory_attention_mask": None,
    }


def _bare_memory_model(config: MemoryBottleneckConfig) -> MolmoAct2:
    model = object.__new__(MolmoAct2)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        memory=config,
        state_format="discrete",
        action_horizon=2,
        max_action_dim=3,
    )
    model.memory_encoder = (
        MemoryBottleneckEncoder(config)
        if config.mode in {"frozen", "finetune"}
        else None
    )
    return model


@pytest.mark.parametrize("mode", ["frozen", "precomputed"])
def test_forward_rejects_memory_batch_size_before_backbone_execution(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    config = _memory_config(mode)
    model = _bare_memory_model(config)
    memory_kwargs = _memory_kwargs(config, batch_size=2)
    if mode == "precomputed":
        memory_kwargs = {
            "memory_tokens": torch.randn(2, config.num_latents, config.memory_dim),
            "memory_attention_mask": torch.ones(2, config.num_latents, dtype=torch.bool),
            **{
                key: None
                for key in (
                    "memory_cosmo_latents",
                    "memory_recent_embeddings",
                    "memory_action_embeddings",
                    "memory_proprio_embeddings",
                    "memory_cosmo_mask",
                    "memory_recent_mask",
                    "memory_action_mask",
                    "memory_proprio_mask",
                    "memory_language_embedding",
                )
            },
        }

    def unexpected_backbone(**_: object) -> None:
        raise AssertionError("backbone ran before memory batch validation")

    monkeypatch.setattr(model, "_run_backbone", unexpected_backbone)
    monkeypatch.setattr(model, "_require_action_expert", lambda: None)

    with pytest.raises(ValueError, match="memory batch size"):
        model(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            actions=torch.ones(1, model.config.action_horizon, model.config.max_action_dim),
            **memory_kwargs,
        )


def test_forward_allows_vlm_only_batch_without_memory_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _bare_memory_model(_memory_config("frozen"))
    expected = OLMoOutput(
        logits=torch.zeros(1, 3, 4),
        attn_key_values=None,
        hidden_states=None,
    )

    def fake_backbone(**_: object):
        return expected, None, None

    monkeypatch.setattr(model, "_run_backbone", fake_backbone)

    actual = model(input_ids=torch.ones(1, 3, dtype=torch.long))

    assert actual.logits is expected.logits
    assert actual.metrics == {}
    assert actual.internal == {}


def test_memory_mode_controls_encoder_gradients() -> None:
    for mode in ("frozen", "finetune"):
        config = _memory_config(mode)
        model = _bare_memory_model(config)
        memory_tokens, memory_mask = model._resolve_memory_tokens(
            expected_batch_size=2,
            **_memory_kwargs(config, batch_size=2),
        )

        assert memory_tokens is not None
        assert memory_mask is not None
        if mode == "frozen":
            assert not memory_tokens.requires_grad
            assert all(
                parameter.grad is None
                for parameter in model.memory_encoder.parameters()  # type: ignore[union-attr]
            )
        else:
            memory_tokens.square().mean().backward()
            gradients = [
                parameter.grad
                for parameter in model.memory_encoder.parameters()  # type: ignore[union-attr]
            ]
            assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_memory_adapter_is_owned_by_action_expert_optimizer() -> None:
    config = _memory_config("finetune")
    model = _bare_memory_model(config)
    model.action_expert = nn.Module()
    model.action_expert_depth_gate = None
    model.action_expert.memory_adapter = nn.Linear(config.memory_dim, 6)
    assert {
        id(parameter) for parameter in model.action_expert.memory_adapter.parameters()
    } <= {id(parameter) for parameter in model.get_action_expert_parameters()}


def test_precomputed_memory_matches_encoder_output() -> None:
    encoder_config = _memory_config("finetune")
    encoder_model = _bare_memory_model(encoder_config)
    raw_memory = _memory_kwargs(encoder_config, batch_size=2)
    encoded_tokens, encoded_mask = encoder_model._resolve_memory_tokens(
        expected_batch_size=2,
        **raw_memory,
    )

    cached_config = _memory_config("precomputed")
    cached_model = _bare_memory_model(cached_config)
    cached_tokens, cached_mask = cached_model._resolve_memory_tokens(
        expected_batch_size=2,
        memory_tokens=encoded_tokens,
        memory_attention_mask=encoded_mask,
        **{
            key: None
            for key in (
                "memory_cosmo_latents",
                "memory_recent_embeddings",
                "memory_action_embeddings",
                "memory_proprio_embeddings",
                "memory_cosmo_mask",
                "memory_recent_mask",
                "memory_action_mask",
                "memory_proprio_mask",
                "memory_language_embedding",
            )
        },
    )

    torch.testing.assert_close(cached_tokens, encoded_tokens)
    assert torch.equal(cached_mask, encoded_mask)


def test_stage1_encoder_loads_after_parameter_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = _memory_config("frozen")
    config.stage1_checkpoint = str(tmp_path / "stage1.npz")
    model = _bare_memory_model(config)
    assert model.memory_encoder is not None
    expected_state = {
        name: torch.full_like(value, 2.0)
        for name, value in model.memory_encoder.state_dict().items()
    }
    captured: dict[str, object] = {}

    def fake_loader(path, encoder, *, strict):
        captured["path"] = path
        captured["encoder"] = encoder
        captured["strict"] = strict
        return expected_state

    monkeypatch.setattr(
        "olmo.models.molmoact2.molmoact2.load_stage1_bottleneck_state_dict",
        fake_loader,
    )
    monkeypatch.setattr(
        "olmo.models.molmoact2.molmoact2.Path.is_file",
        lambda _: True,
    )

    model.initialize_memory_encoder_from_stage1()

    assert captured["path"] == Path(config.stage1_checkpoint)
    assert captured["encoder"] is model.memory_encoder
    assert captured["strict"] is True
    assert model._memory_stage1_checkpoint_loaded
    for name, expected in expected_state.items():
        torch.testing.assert_close(model.memory_encoder.state_dict()[name], expected)


@pytest.mark.parametrize("rank", [0, 1])
def test_stage1_encoder_uses_rank_zero_state_dict_broadcast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    rank: int,
) -> None:
    config = _memory_config("frozen")
    config.stage1_checkpoint = str(tmp_path / "stage1.npz")
    model = _bare_memory_model(config)
    assert model.memory_encoder is not None
    expected_state = {
        name: value.detach().clone()
        for name, value in model.memory_encoder.state_dict().items()
    }
    loader_calls: list[Path] = []
    captured: dict[str, object] = {}

    def fake_loader(path, encoder, *, strict):
        loader_calls.append(path)
        assert encoder is model.memory_encoder
        assert strict is True
        return expected_state

    def fake_set_model_state_dict(*, model, model_state_dict, options):
        captured["model"] = model
        captured["state_dict"] = model_state_dict
        captured["options"] = options

    monkeypatch.setattr(
        "olmo.models.molmoact2.molmoact2.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "olmo.models.molmoact2.molmoact2.dist.get_rank",
        lambda: rank,
    )
    monkeypatch.setattr(
        "olmo.models.molmoact2.molmoact2.load_stage1_bottleneck_state_dict",
        fake_loader,
    )
    monkeypatch.setattr(
        "olmo.models.molmoact2.molmoact2.dist_cp_sd.set_model_state_dict",
        fake_set_model_state_dict,
    )
    if rank == 0:
        monkeypatch.setattr(
            "olmo.models.molmoact2.molmoact2.Path.is_file",
            lambda _: True,
        )

    model.initialize_memory_encoder_from_stage1()

    assert captured["model"] is model
    if rank == 0:
        assert loader_calls == [Path(config.stage1_checkpoint)]
        assert captured["state_dict"] == {
            f"memory_encoder.{name}": value
            for name, value in expected_state.items()
        }
    else:
        assert not loader_calls
        assert captured["state_dict"] == {}
    assert model._memory_stage1_checkpoint_loaded


class _CheckpointLoadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_encoder = nn.Linear(2, 2)
        self.action_expert = None
        self.action_expert_depth_gate = None


def _disable_checkpoint_layout_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "olmo.train.checkpoint_loading.checkpoint_llm_token_layout",
        lambda _: None,
    )
    monkeypatch.setattr(
        "olmo.train.checkpoint_loading.resolve_source_layout_from_state_dict",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "olmo.train.checkpoint_loading.get_checkpoint_wte_rows",
        lambda _: None,
    )
    monkeypatch.setattr(
        "olmo.train.checkpoint_loading.resize_or_init_base_embedding_for_checkpoint",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "olmo.train.checkpoint_loading.resize_or_init_action_expert_dim_for_checkpoint",
        lambda *_, **__: None,
    )


def test_model_only_checkpoint_allows_missing_memory_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _disable_checkpoint_layout_helpers(monkeypatch)
    torch.save({}, tmp_path / "model.pt")

    has_action_expert, has_memory_encoder = load_unsharded_checkpoint_allowing_missing_action_expert(
        str(tmp_path),
        _CheckpointLoadModel(),
    )

    assert not has_action_expert
    assert not has_memory_encoder


def test_memory_conditioned_checkpoint_requires_pretrained_action_expert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _disable_checkpoint_layout_helpers(monkeypatch)
    torch.save({}, tmp_path / "model.pt")
    model = _bare_memory_model(_memory_config("finetune"))
    model.action_expert = nn.Linear(2, 2)
    model.action_expert_depth_gate = None

    with pytest.raises(RuntimeError, match="requires pretrained action expert weights"):
        load_unsharded_checkpoint_allowing_missing_action_expert(
            str(tmp_path),
            model,
            require_action_expert=True,
        )


def test_model_only_checkpoint_initializes_missing_memory_to_action_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _disable_checkpoint_layout_helpers(monkeypatch)
    torch.save({}, tmp_path / "model.pt")
    model = _bare_memory_model(_memory_config("frozen"))
    model.action_expert = nn.Module()
    model.action_expert_depth_gate = None
    model.action_expert.memory_adapter = nn.Linear(2, 3)

    load_unsharded_checkpoint_allowing_missing_action_expert(
        str(tmp_path),
        model,
    )

    assert torch.isfinite(model.action_expert.memory_adapter.weight).all()
    assert torch.count_nonzero(model.action_expert.memory_adapter.weight) > 0


def test_model_only_checkpoint_reports_memory_encoder_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _disable_checkpoint_layout_helpers(monkeypatch)
    checkpoint_model = _CheckpointLoadModel()
    expected_state = {
        name: torch.full_like(value, 2.0)
        for name, value in checkpoint_model.memory_encoder.state_dict().items()
    }
    torch.save(
        {f"memory_encoder.{name}": value for name, value in expected_state.items()},
        tmp_path / "model.pt",
    )
    model = _CheckpointLoadModel()

    has_action_expert, has_memory_encoder = load_unsharded_checkpoint_allowing_missing_action_expert(
        str(tmp_path),
        model,
    )

    assert not has_action_expert
    assert has_memory_encoder
    for name, expected in expected_state.items():
        torch.testing.assert_close(model.memory_encoder.state_dict()[name], expected)
