import pytest
import torch

from olmo.memory_features import MemoryFeatureKeys
from olmo.nn.memory_bottleneck import MemoryBottleneckConfig, MemoryInputs


def test_default_memory_config_matches_pi05_dinov3_stage1_checkpoint() -> None:
    config = MemoryBottleneckConfig()

    assert config.recent_dim == 768
    assert config.action_dim == 16
    assert config.proprio_dim == 16


def test_pi05_dinov3_schema_maps_stage1_feature_names() -> None:
    keys = MemoryFeatureKeys.from_schema("pi05_dinov3")

    assert keys.cosmo_latents == "cosmo_latent_emb"
    assert keys.recent_embeddings == "cosmo_recent_emb"
    assert keys.action_embeddings == "cosmo_action_emb"
    assert keys.proprio_embeddings == "cosmo_proprio_emb"
    assert keys.cosmo_mask == "cosmo_mem_mask"
    assert keys.recent_mask == "cosmo_recent_mask"
    assert keys.action_mask == "cosmo_action_mask"
    assert keys.proprio_mask == "cosmo_proprio_mask"
    assert keys.language_embedding == "cosmo_lang_emb"


def _valid_memory_tensors(batch_size: int = 2) -> dict[str, torch.Tensor]:
    config = MemoryBottleneckConfig()
    return {
        "cosmo_latents": torch.zeros(batch_size, 3, 48),
        "recent_embeddings": torch.zeros(batch_size, 4, config.recent_dim),
        "action_embeddings": torch.zeros(batch_size, 4, config.action_dim),
        "proprio_embeddings": torch.zeros(batch_size, 4, config.proprio_dim),
        "cosmo_mask": torch.tensor([[True, False, False], [True, False, False]]),
        "recent_mask": torch.zeros(batch_size, 4, dtype=torch.bool),
        "action_mask": torch.zeros(batch_size, 4, dtype=torch.bool),
        "proprio_mask": torch.zeros(batch_size, 4, dtype=torch.bool),
        "language_embedding": torch.zeros(batch_size, 2048),
    }


def test_memory_inputs_require_one_valid_token_per_batch_row() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    tensors["cosmo_mask"][1] = False
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="at least one valid memory token"):
        inputs.validate()


def test_memory_inputs_reject_non_boolean_masks() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    tensors["recent_mask"] = tensors["recent_mask"].to(torch.float32)
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="must be boolean"):
        inputs.validate()


def test_memory_inputs_reject_invalid_stream_shapes() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    tensors["recent_embeddings"] = torch.zeros(2, 2048)
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="recent_embeddings must have rank 3"):
        inputs.validate()


def test_memory_inputs_reject_mismatched_batches_and_widths() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    tensors["action_embeddings"] = torch.zeros(1, 4, config.action_dim)
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="batch size"):
        inputs.validate()

    tensors = _valid_memory_tensors()
    tensors["proprio_embeddings"] = torch.zeros(2, 4, 13)
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="proprio_embeddings must have width 16"):
        inputs.validate()


def test_memory_inputs_require_floating_feature_tensors() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    tensors["action_embeddings"] = torch.zeros(2, 4, 14, dtype=torch.long)
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="action_embeddings must be floating point"):
        inputs.validate()


def test_memory_inputs_require_one_device() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    tensors["cosmo_latents"] = torch.zeros(2, 3, 48, device="meta")
    inputs = MemoryInputs.from_tensors(config=config, **tensors)

    with pytest.raises(ValueError, match="same device"):
        inputs.validate()


def test_memory_inputs_accept_native_model_keyword_names() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    tensors = _valid_memory_tensors()
    inputs = MemoryInputs.from_model_kwargs(
        config=config,
        memory_cosmo_latents=tensors["cosmo_latents"],
        memory_recent_embeddings=tensors["recent_embeddings"],
        memory_action_embeddings=tensors["action_embeddings"],
        memory_proprio_embeddings=tensors["proprio_embeddings"],
        memory_cosmo_mask=tensors["cosmo_mask"],
        memory_recent_mask=tensors["recent_mask"],
        memory_action_mask=tensors["action_mask"],
        memory_proprio_mask=tensors["proprio_mask"],
        memory_language_embedding=tensors["language_embedding"],
    )

    assert inputs.cosmo_latents is tensors["cosmo_latents"]
    inputs.validate()


def test_memory_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="memory mode"):
        MemoryBottleneckConfig(enabled=True, mode="invalid").validate()


def test_memory_config_rejects_non_positive_attention_heads() -> None:
    with pytest.raises(ValueError, match="num_heads must be positive"):
        MemoryBottleneckConfig(enabled=True, num_heads=0).validate()
