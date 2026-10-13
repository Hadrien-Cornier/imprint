import dataclasses

import torch

from olmo.nn.memory_bottleneck import (
    MemoryBottleneckConfig,
    MemoryBottleneckEncoder,
    MemoryInputs,
)


def _memory_inputs(config: MemoryBottleneckConfig) -> MemoryInputs:
    torch.manual_seed(0)
    return MemoryInputs.from_tensors(
        cosmo_latents=torch.randn(2, 3, config.cosmos_latent_dim),
        recent_embeddings=torch.randn(2, 4, config.recent_dim),
        action_embeddings=torch.randn(2, 4, config.action_dim),
        proprio_embeddings=torch.randn(2, 4, config.proprio_dim),
        cosmo_mask=torch.tensor([[True, False, False], [True, True, False]]),
        recent_mask=torch.tensor([[True, False, False, False], [True, True, False, False]]),
        action_mask=torch.tensor([[True, False, False, False], [True, True, False, False]]),
        proprio_mask=torch.tensor([[True, False, False, False], [True, True, False, False]]),
        language_embedding=torch.randn(2, config.language_dim),
        config=config,
    )


def test_encoder_returns_masked_256_by_1024_memory_tokens() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    encoder = MemoryBottleneckEncoder(config)

    tokens, mask = encoder(_memory_inputs(config))

    assert tokens.shape == (2, 256, 1024)
    assert torch.equal(mask, torch.ones(2, 256, dtype=torch.bool))


def test_padded_stream_values_do_not_change_memory_tokens() -> None:
    config = MemoryBottleneckConfig(enabled=True)
    encoder = MemoryBottleneckEncoder(config).eval()
    inputs = _memory_inputs(config)

    baseline, _ = encoder(inputs)
    perturbed = dataclasses.replace(
        inputs,
        recent_embeddings=inputs.recent_embeddings.masked_fill(
            ~inputs.recent_mask.unsqueeze(-1), 1e6
        ),
    )
    actual, _ = encoder(perturbed)

    torch.testing.assert_close(actual, baseline)
