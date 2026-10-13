import numpy as np
import pytest
import torch

from olmo.nn.memory_bottleneck import MemoryBottleneckConfig, MemoryBottleneckEncoder
from olmo.nn.memory_checkpoint import load_stage1_bottleneck_state_dict


@pytest.fixture
def memory_encoder() -> MemoryBottleneckEncoder:
    return MemoryBottleneckEncoder(
        MemoryBottleneckConfig(
            enabled=True,
            num_latents=4,
            memory_dim=8,
            tokenizer_hidden_dim=4,
            num_heads=2,
            num_cross_blocks=1,
        )
    )


def test_converter_transposes_flax_dense_kernels_and_ignores_decoder(
    tmp_path, memory_encoder: MemoryBottleneckEncoder
) -> None:
    checkpoint = tmp_path / "stage1.npz"
    np.savez(
        checkpoint,
        **{
            "bottleneck/proj_cosmo/kernel": np.arange(48 * 8, dtype=np.float32).reshape(48, 8),
            "bottleneck/proj_cosmo/bias": np.zeros(8, dtype=np.float32),
            "decoder/align/kernel": np.zeros((8, 32), dtype=np.float32),
        },
    )

    converted = load_stage1_bottleneck_state_dict(checkpoint, memory_encoder, strict=False)

    assert converted["proj_cosmo.weight"].shape == (8, 48)
    assert np.array_equal(
        converted["proj_cosmo.weight"].numpy(),
        np.arange(48 * 8, dtype=np.float32).reshape(48, 8).T,
    )
    assert "decoder/align/kernel" not in converted


def test_converter_rejects_missing_bottleneck_parameter(
    tmp_path, memory_encoder: MemoryBottleneckEncoder
) -> None:
    checkpoint = tmp_path / "incomplete.npz"
    np.savez(
        checkpoint,
        **{"bottleneck/proj_cosmo/bias": np.zeros(8, dtype=np.float32)},
    )

    with pytest.raises(KeyError, match="missing Stage-1 parameters"):
        load_stage1_bottleneck_state_dict(checkpoint, memory_encoder, strict=True)


def _as_flax_stage1_checkpoint(memory_encoder: MemoryBottleneckEncoder) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for torch_key, tensor in memory_encoder.state_dict().items():
        value = tensor.detach().cpu().numpy()
        parts = torch_key.split(".")
        if torch_key == "latents":
            output["bottleneck/latents"] = value
            continue
        if len(parts) >= 5 and parts[0] == "blocks" and parts[2] in {"attn", "self_attn"}:
            projection = {
                "q_proj": "query",
                "k_proj": "key",
                "v_proj": "value",
                "out_proj": "out",
            }[parts[3]]
            if parts[4] == "weight":
                if projection in {"query", "key", "value"}:
                    value = value.T.reshape(value.shape[1], memory_encoder.config.num_heads, -1)
                else:
                    value = value.T.reshape(memory_encoder.config.num_heads, -1, value.shape[0])
                leaf = "kernel"
            else:
                if projection in {"query", "key", "value"}:
                    value = value.reshape(memory_encoder.config.num_heads, -1)
                leaf = "bias"
            output[f"bottleneck/blocks/{parts[1]}/{parts[2]}/{projection}/{leaf}"] = value
            continue
        if len(parts) >= 4 and parts[0] == "blocks" and parts[2] in {"norm_q", "norm_kv", "norm_s"}:
            leaf = "scale" if parts[3] == "weight" else "bias"
            output[f"bottleneck/blocks/{parts[1]}/{parts[2]}/{leaf}"] = value
            continue
        if len(parts) >= 4 and parts[0] == "blocks" and parts[2] == "ffn":
            leaf = "kernel" if parts[3] == "weight" else "bias"
            output[f"bottleneck/blocks/{parts[1]}/ffn/{leaf}"] = value.T if leaf == "kernel" else value
            continue

        leaf = "kernel" if parts[-1] == "weight" else "bias"
        output[f"bottleneck/{'/'.join(parts[:-1])}/{leaf}"] = value.T if leaf == "kernel" else value
    return output


def test_converter_loads_every_stage1_parameter_strictly(tmp_path, memory_encoder: MemoryBottleneckEncoder) -> None:
    checkpoint = tmp_path / "complete.npz"
    np.savez(checkpoint, **_as_flax_stage1_checkpoint(memory_encoder))

    converted = load_stage1_bottleneck_state_dict(checkpoint, memory_encoder, strict=True)
    clone = MemoryBottleneckEncoder(memory_encoder.config)
    clone.load_state_dict(converted, strict=True)

    assert set(converted) == set(memory_encoder.state_dict())
    for key, expected in memory_encoder.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[key], expected)


def test_converter_rejects_stage1_parameter_shape_mismatch(
    tmp_path, memory_encoder: MemoryBottleneckEncoder
) -> None:
    checkpoint = tmp_path / "wrong-shape.npz"
    values = _as_flax_stage1_checkpoint(memory_encoder)
    values["bottleneck/proj_cosmo/kernel"] = np.zeros((47, 8), dtype=np.float32)
    np.savez(checkpoint, **values)

    with pytest.raises(KeyError, match="shape mismatch"):
        load_stage1_bottleneck_state_dict(checkpoint, memory_encoder, strict=True)


def test_converter_rejects_non_canonical_block_index(
    tmp_path, memory_encoder: MemoryBottleneckEncoder
) -> None:
    checkpoint = tmp_path / "duplicate-block.npz"
    values = _as_flax_stage1_checkpoint(memory_encoder)
    source_key = "bottleneck/blocks/0/attn/query/kernel"
    values["bottleneck/blocks/00/attn/query/kernel"] = values[source_key]
    np.savez(checkpoint, **values)

    with pytest.raises(KeyError, match="non-canonical block index"):
        load_stage1_bottleneck_state_dict(checkpoint, memory_encoder, strict=True)


def test_converter_rejects_non_floating_stage1_parameter(
    tmp_path, memory_encoder: MemoryBottleneckEncoder
) -> None:
    checkpoint = tmp_path / "integer-weight.npz"
    values = _as_flax_stage1_checkpoint(memory_encoder)
    values["bottleneck/proj_cosmo/kernel"] = np.zeros((48, 8), dtype=np.int64)
    np.savez(checkpoint, **values)

    with pytest.raises(ValueError, match="must use a floating dtype"):
        load_stage1_bottleneck_state_dict(checkpoint, memory_encoder, strict=True)
