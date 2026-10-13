from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf


def _config(encoder_type="gated_delta_hybrid", scan_mode="recurrent"):
    return OmegaConf.create(
        {
            "budget": 4,
            "num_views": 1,
            "memory_token_dim": 16,
            "cosmo_bottleneck_memory": {
                "memory_encoder_type": encoder_type,
                "num_latents": 4,
                "num_cross_blocks": 1,
                "num_heads": 2,
                "cosmos_latent_dim": 3,
                "siglip_dim": 8,
                "action_in_dim": 4,
                "proprio_in_dim": 5,
                "lang_emb_dim": 6,
                "action_tokenizer_hidden": 8,
                "proprio_tokenizer_hidden": 8,
                "temporal_factor": 2,
                "pool_hw": 1,
                "delta_num_layers": 2,
                "delta_num_heads": 2,
                "delta_head_dim": 8,
                "delta_conv_kernel": 3,
                "delta_ffn_inner": 32,
                "delta_scan_checkpoint_size": 5,
                "delta_full_scan_mode": scan_mode,
                "delta_wy_matmul_dtype": "float32",
                "delta_recent_fusion_blocks": 1,
                "delta_max_chunks": 4,
            },
            "cosmo_decoder": {
                "num_heads": 2,
                "num_blocks": 1,
                "align_dim": 8,
                "align_num_patches": 3,
                "ffn_inner": 32,
                "action_dim": 4,
                "proprio_dim": 5,
                "max_query_steps": 16,
            },
        }
    )


def _inputs():
    rng = np.random.default_rng(7)

    def as_jax(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.float32)

    return {
        "cosmo": as_jax(1, 4, 3),
        "recent": as_jax(1, 3, 8),
        "action": as_jax(1, 6, 4),
        "proprio": as_jax(1, 6, 5),
        "language": as_jax(1, 6),
        "cosmo_mask": jnp.asarray([[True, True, False, False]]),
        "recent_mask": jnp.asarray([[True, True, True]]),
        "action_mask": jnp.asarray([[True, True, True, True, True, True]]),
        "proprio_mask": jnp.asarray([[True, True, True, True, True, True]]),
    }


def test_factory_keeps_legacy_encoder_as_default():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    config = _config()
    del config.cosmo_bottleneck_memory.memory_encoder_type
    module = create_cosmo_memory_encoder(config, nnx.Rngs(0))
    assert module.__class__.__name__ == "CosmoBottleneckMemory"


def test_memory_compute_dtype_is_explicit_and_keeps_fp32_parameters():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    default_module = create_cosmo_memory_encoder(_config(), nnx.Rngs(0))
    assert default_module.dtype == jnp.float32

    mixed_config = _config(scan_mode="chunkwise_wy")
    mixed_config.cosmo_bottleneck_memory.memory_compute_dtype = "bfloat16"
    mixed_module = create_cosmo_memory_encoder(mixed_config, nnx.Rngs(0))
    assert mixed_module.dtype == jnp.bfloat16
    assert mixed_module.latents.value.dtype == jnp.float32
    assert mixed_module.delta_blocks[0].q_proj.kernel.value.dtype == jnp.float32


def test_policy_incremental_path_requires_explicit_encoder_type():
    from mme_vla_suite.policies.policy import _uses_gated_delta_memory

    config = _config()
    config.representation_type = "cosmo_bottleneck"
    assert _uses_gated_delta_memory(config)

    del config.cosmo_bottleneck_memory.memory_encoder_type
    assert not _uses_gated_delta_memory(config)
    config.representation_type = "perceptual"
    config.cosmo_bottleneck_memory.memory_encoder_type = "gated_delta_hybrid"
    assert not _uses_gated_delta_memory(config)


def test_gated_delta_full_scan_is_finite_and_has_legacy_shape():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    module = create_cosmo_memory_encoder(_config(), nnx.Rngs(0))
    inputs = _inputs()
    tokens, mask, stats = module(
        inputs["cosmo"],
        inputs["recent"],
        inputs["action"],
        inputs["proprio"],
        inputs["cosmo_mask"],
        lang_emb=inputs["language"],
        recent_mask=inputs["recent_mask"],
        action_mask=inputs["action_mask"],
        proprio_mask=inputs["proprio_mask"],
    )
    assert tokens.shape == (1, 4, 16)
    assert mask.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(tokens)))
    assert float(stats["delta/finalized_chunks"][0]) == 2.0
    assert float(stats["delta/recent_steps"][0]) == 2.0


def test_bfloat16_compute_keeps_master_parameters_and_gradients_float32():
    from mme_vla_suite.training.pretrain_loss import (
        composite_alignment_loss,
        make_stage1_modules,
    )

    config = _config(scan_mode="chunkwise_wy")
    config.cosmo_bottleneck_memory.delta_wy_matmul_dtype = "bfloat16"
    modules = make_stage1_modules(config, seed=0, dtype=jnp.bfloat16)
    bottleneck = modules["bottleneck"]
    decoder = modules["decoder"]
    assert bottleneck.latents.value.dtype == jnp.float32
    assert bottleneck.delta_blocks[0].q_proj.kernel.value.dtype == jnp.float32
    assert decoder.summary_emb.value.dtype == jnp.float32
    assert decoder.blocks[0].ffn_in.kernel.value.dtype == jnp.float32

    inputs = _inputs()
    batch = {
        "cosmo_latent_emb": inputs["cosmo"],
        "cosmo_recent_emb": inputs["recent"],
        "cosmo_action_emb": inputs["action"],
        "cosmo_proprio_emb": inputs["proprio"],
        "cosmo_mem_mask": inputs["cosmo_mask"],
        "cosmo_lang_emb": inputs["language"],
        "cosmo_recent_mask": inputs["recent_mask"],
        "cosmo_action_mask": inputs["action_mask"],
        "cosmo_proprio_mask": inputs["proprio_mask"],
        "query_steps": jnp.asarray([[0, 1]], dtype=jnp.int32),
        "query_mask": jnp.ones((1, 2), dtype=jnp.bool_),
        "align_target": jnp.zeros((1, 2, 3, 8), dtype=jnp.float16),
        "action_target": jnp.zeros((1, 2, 4), dtype=jnp.float32),
        "proprio_target": jnp.zeros((1, 2, 5), dtype=jnp.float32),
        "reward_target": jnp.zeros((1,), dtype=jnp.float32),
    }
    graphdef, state = nnx.split((bottleneck, decoder))

    def loss_fn(current_state):
        current_bottleneck, current_decoder = nnx.merge(
            graphdef,
            current_state,
        )
        total, _ = composite_alignment_loss(
            {
                "bottleneck": current_bottleneck,
                "decoder": current_decoder,
            },
            batch,
            {
                "align": 1.0,
                "action": 0.5,
                "proprio": 0.5,
                "reward": 0.5,
            },
        )
        return total

    total, gradients = jax.value_and_grad(loss_fn)(state)
    assert bool(jnp.isfinite(total))
    gradient_leaves = jax.tree.leaves(gradients)
    assert gradient_leaves
    assert all(leaf.dtype == jnp.float32 for leaf in gradient_leaves)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)


def test_rematerialized_scan_matches_plain_scan():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    checkpointed_config = _config()
    plain_config = _config()
    plain_config.cosmo_bottleneck_memory.delta_scan_checkpoint_size = 100
    checkpointed = create_cosmo_memory_encoder(checkpointed_config, nnx.Rngs(0))
    plain = create_cosmo_memory_encoder(plain_config, nnx.Rngs(0))
    inputs = _inputs()

    def run(module):
        tokens, _, _ = module(
            inputs["cosmo"],
            inputs["recent"],
            inputs["action"],
            inputs["proprio"],
            inputs["cosmo_mask"],
            lang_emb=inputs["language"],
            recent_mask=inputs["recent_mask"],
            action_mask=inputs["action_mask"],
            proprio_mask=inputs["proprio_mask"],
        )
        return tokens

    np.testing.assert_allclose(
        np.asarray(run(checkpointed)),
        np.asarray(run(plain)),
        rtol=2e-5,
        atol=2e-5,
    )


def test_rematerialized_scan_has_finite_input_gradient():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    module = create_cosmo_memory_encoder(_config(), nnx.Rngs(0))
    inputs = _inputs()

    def loss(cosmo):
        tokens, _, _ = module(
            cosmo,
            inputs["recent"],
            inputs["action"],
            inputs["proprio"],
            inputs["cosmo_mask"],
            lang_emb=inputs["language"],
            recent_mask=inputs["recent_mask"],
            action_mask=inputs["action_mask"],
            proprio_mask=inputs["proprio_mask"],
        )
        return jnp.mean(jnp.square(tokens))

    gradient = jax.grad(loss)(inputs["cosmo"])
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_chunkwise_wy_matches_recurrent_outputs_and_gradients():
    from mme_vla_suite.models.representation.gated_delta_mem import (
        _chunkwise_wy_delta_scan,
        _recurrent_delta_scan,
    )

    keys = jax.random.split(jax.random.key(11), 7)
    batch, length, heads, head_dim = 2, 10, 2, 4
    matrix = jax.random.normal(keys[0], (batch, heads, head_dim, head_dim))
    q = jax.random.normal(keys[1], (batch, length, heads, head_dim))
    k = jax.random.normal(keys[2], (batch, length, heads, head_dim))
    q = q * jax.lax.rsqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(keys[3], (batch, length, heads, head_dim))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)))
    decay = jnp.exp(-jax.nn.softplus(jax.random.normal(keys[5], (batch, length, heads)) - 4.0))
    mask = jnp.asarray(
        [
            [True, True, False, True, True, True, False, True, True, True],
            [True, False, True, True, True, False, True, True, False, True],
        ]
    )

    recurrent_state, recurrent_output = _recurrent_delta_scan(
        matrix, q, k, v, beta, decay, mask, checkpoint_size=5
    )
    wy_state, wy_output = _chunkwise_wy_delta_scan(
        matrix, q, k, v, beta, decay, mask, chunk_size=5
    )
    np.testing.assert_allclose(
        np.asarray(wy_state),
        np.asarray(recurrent_state),
        rtol=3e-4,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        np.asarray(wy_output),
        np.asarray(recurrent_output),
        rtol=3e-4,
        atol=3e-4,
    )

    def recurrent_loss(matrix, q, k, v, beta, decay):
        state, output = _recurrent_delta_scan(
            matrix, q, k, v, beta, decay, mask, checkpoint_size=5
        )
        return jnp.mean(jnp.square(state)) + jnp.mean(jnp.square(output))

    def wy_loss(matrix, q, k, v, beta, decay):
        state, output = _chunkwise_wy_delta_scan(
            matrix, q, k, v, beta, decay, mask, chunk_size=5
        )
        return jnp.mean(jnp.square(state)) + jnp.mean(jnp.square(output))

    recurrent_gradients = jax.grad(recurrent_loss, argnums=tuple(range(6)))(
        matrix, q, k, v, beta, decay
    )
    wy_gradients = jax.grad(wy_loss, argnums=tuple(range(6)))(
        matrix, q, k, v, beta, decay
    )
    for wy_gradient, recurrent_gradient in zip(
        wy_gradients, recurrent_gradients, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(wy_gradient),
            np.asarray(recurrent_gradient),
            rtol=2e-3,
            atol=2e-3,
        )


def test_chunkwise_full_scan_matches_recurrent_module():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    recurrent = create_cosmo_memory_encoder(
        _config(scan_mode="recurrent"),
        nnx.Rngs(0),
    )
    chunkwise = create_cosmo_memory_encoder(
        _config(scan_mode="chunkwise_wy"),
        nnx.Rngs(0),
    )
    inputs = _inputs()

    def run(module):
        tokens, _, _ = module(
            inputs["cosmo"],
            inputs["recent"],
            inputs["action"],
            inputs["proprio"],
            inputs["cosmo_mask"],
            lang_emb=inputs["language"],
            recent_mask=inputs["recent_mask"],
            action_mask=inputs["action_mask"],
            proprio_mask=inputs["proprio_mask"],
        )
        return tokens

    np.testing.assert_allclose(
        np.asarray(run(chunkwise)),
        np.asarray(run(recurrent)),
        rtol=5e-4,
        atol=5e-4,
    )


def test_chunkwise_wy_bfloat16_matmuls_stay_close_to_float32():
    from mme_vla_suite.models.representation.gated_delta_mem import (
        _chunkwise_wy_delta_scan,
    )

    keys = jax.random.split(jax.random.key(19), 6)
    shape = (2, 10, 2, 4)
    matrix = 0.01 * jax.random.normal(keys[0], (2, 2, 4, 4))
    q = jax.random.normal(keys[1], shape)
    k = jax.random.normal(keys[2], shape)
    q = q * jax.lax.rsqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(keys[3], shape)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (2, 10, 2)) - 1.0)
    decay = jnp.exp(-jax.nn.softplus(jax.random.normal(keys[5], (2, 10, 2)) - 4.0))
    mask = jnp.ones((2, 10), dtype=jnp.bool_)

    float_state, float_output = _chunkwise_wy_delta_scan(
        matrix,
        q,
        k,
        v,
        beta,
        decay,
        mask,
        chunk_size=5,
        matmul_dtype=jnp.float32,
    )
    mixed_state, mixed_output = _chunkwise_wy_delta_scan(
        matrix,
        q,
        k,
        v,
        beta,
        decay,
        mask,
        chunk_size=5,
        matmul_dtype=jnp.bfloat16,
    )
    np.testing.assert_allclose(
        np.asarray(mixed_state),
        np.asarray(float_state),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(mixed_output),
        np.asarray(float_output),
        rtol=2e-2,
        atol=2e-2,
    )


def test_full_scan_matches_incremental_updates_and_recent_read():
    from mme_vla_suite.models.representation.memory_factory import create_cosmo_memory_encoder

    module = create_cosmo_memory_encoder(_config(), nnx.Rngs(0))
    inputs = _inputs()
    full_tokens, _, _ = module(
        inputs["cosmo"],
        inputs["recent"],
        inputs["action"],
        inputs["proprio"],
        inputs["cosmo_mask"],
        lang_emb=inputs["language"],
        recent_mask=inputs["recent_mask"],
        action_mask=inputs["action_mask"],
        proprio_mask=inputs["proprio_mask"],
    )

    state = module.init_incremental_state(batch_size=1)
    for chunk in range(2):
        start = chunk * 2
        state = module.update_incremental(
            state,
            inputs["cosmo"][:, chunk : chunk + 1],
            inputs["action"][:, start : start + 2],
            inputs["proprio"][:, start : start + 2],
        )
    incremental_tokens, _, stats = module.read_incremental(
        state,
        inputs["recent"],
        inputs["action"][:, 4:],
        inputs["proprio"][:, 4:],
        lang_emb=inputs["language"],
        recent_mask=inputs["recent_mask"],
    )

    np.testing.assert_allclose(
        np.asarray(incremental_tokens),
        np.asarray(full_tokens),
        rtol=2e-5,
        atol=2e-5,
    )
    assert float(stats["delta/finalized_chunks"][0]) == 2.0
