import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.integration.memory_pi0 import MemoryPi0Config
from mme_vla_suite.models.integration.memory_pi0_gemma import MemoryAttention
from mme_vla_suite.training.pi0_weight_loader import NativePi0Stage1BottleneckWeightLoader
from mme_vla_suite.training.pi0_weight_loader import validate_native_pi0_parameter_trees
from openpi.models import model as model_lib


def _array(shape=(2, 2)):
    return np.zeros(shape, dtype=np.float32)


def _native_pi0_tree():
    return {
        "state_proj": {"kernel": _array()},
        "action_time_mlp_in": {"kernel": _array()},
        "action_time_mlp_out": {"kernel": _array()},
        "action_in_proj": {"kernel": _array()},
        "PaliGemma": {"llm": {"core": _array()}},
    }


def test_memory_pi0_config_is_native_pi0_only():
    config = MemoryPi0Config()
    assert config.model_type is model_lib.ModelType.PI0
    assert config.pi05 is False
    assert config.discrete_state_input is False

    with pytest.raises(ValueError, match="pi05=False"):
        MemoryPi0Config(pi05=True)

    with pytest.raises(ValueError, match="continuous state"):
        MemoryPi0Config(discrete_state_input=True)


def test_native_pi0_loader_accepts_memory_only_new_parameters():
    reference = _native_pi0_tree()
    reference["mem_encoder"] = {"block": _array()}
    reference["PaliGemma"]["llm"]["mem_attn"] = {"kernel": _array()}
    validate_native_pi0_parameter_trees(reference, _native_pi0_tree())


def test_native_pi0_loader_rejects_pi05_checkpoint():
    reference = _native_pi0_tree()
    pi05 = _native_pi0_tree()
    del pi05["state_proj"]
    del pi05["action_time_mlp_in"]
    del pi05["action_time_mlp_out"]
    pi05["time_mlp_in"] = {"kernel": _array()}
    pi05["time_mlp_out"] = {"kernel": _array()}

    with pytest.raises(ValueError, match="missing roots"):
        validate_native_pi0_parameter_trees(reference, pi05)


def test_native_pi0_loader_rejects_missing_base_parameter():
    reference = _native_pi0_tree()
    loaded = _native_pi0_tree()
    del loaded["action_in_proj"]

    with pytest.raises(KeyError, match="non-memory"):
        validate_native_pi0_parameter_trees(reference, loaded)


def test_memory_attention_maps_tokens_into_action_expert_width():
    action_tokens = jnp.ones((2, 5, 64), dtype=jnp.float32)
    memory_tokens = jnp.ones((2, 7, 64), dtype=jnp.float32)
    memory_mask = jnp.ones((2, 7), dtype=jnp.bool_)
    attention = MemoryAttention()
    variables = attention.init(
        jax.random.key(0),
        action_tokens,
        memory_tokens,
        memory_mask,
    )
    output = attention.apply(
        variables,
        action_tokens,
        memory_tokens,
        memory_mask,
    )
    assert output.shape == action_tokens.shape
    assert jnp.all(jnp.isfinite(output))


def test_registered_stage2_profile_uses_pi0_base():
    from mme_vla_suite.training.config import get_config  # noqa: PLC0415

    config = get_config("mme_wan_stage2_pi0_frozen_precomputed")
    assert isinstance(config.model, MemoryPi0Config)
    assert config.model.model_type is model_lib.ModelType.PI0
    assert config.model.action_horizon == 50
    assert config.model.max_token_len == 64
    assert isinstance(
        config.weight_loader,
        NativePi0Stage1BottleneckWeightLoader,
    )
    assert "pi0_base/params" in config.weight_loader.params_path
    assert "pi05_base" not in config.weight_loader.params_path


def test_registered_memft_profile_trains_memory_encoder():
    from mme_vla_suite.training.config import get_config  # noqa: PLC0415

    config = get_config("mme_wan_stage2_pi0_memft")
    assert isinstance(config.model, MemoryPi0Config)
    assert config.model.model_type is model_lib.ModelType.PI0
    assert config.model.history_config.endswith("dinov3-memft.yaml")
    assert "precomputed" not in config.model.history_config
    assert config.freeze_filter.pattern.pattern == ".*img.*"
    assert isinstance(
        config.weight_loader,
        NativePi0Stage1BottleneckWeightLoader,
    )


def test_native_pi0_transform_drops_unused_symbolic_strings():
    from mme_vla_suite.training.config import DropUnusedSymbolicMemory  # noqa: PLC0415

    sample = {
        "state": _array(),
        "simple_subgoal": "unused simple text",
        "grounded_subgoal": "unused grounded text",
    }
    transformed = DropUnusedSymbolicMemory()(sample)
    assert set(transformed) == {"state"}
    assert transformed["state"] is sample["state"]
