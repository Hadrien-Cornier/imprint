"""Strict conversion of Pi0.5 Stage-1 memory checkpoints to PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from olmo.nn.memory_bottleneck import MemoryBottleneckEncoder


_ATTENTION_PROJECTIONS = {
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "out": "out_proj",
}


def flax_key_to_torch_key(key: str) -> tuple[str, bool]:
    """Map a ``bottleneck/...`` Flax leaf to a PyTorch state-dict key.

    The boolean reports whether a regular dense kernel requires a matrix
    transpose. Multi-head attention kernels also transpose, but need an
    additional reshape handled by ``_convert_flax_value``.
    """

    if not key.startswith("bottleneck/"):
        raise ValueError(f"Expected Stage-1 bottleneck key, got {key!r}")
    parts = key.split("/")[1:]
    if parts == ["latents"]:
        return "latents", False
    if len(parts) < 2:
        raise ValueError(f"Unsupported Stage-1 bottleneck key {key!r}")

    if parts[0] == "blocks":
        if len(parts) < 4 or not parts[1].isdigit():
            raise ValueError(f"Unsupported Stage-1 block key {key!r}")
        if str(int(parts[1])) != parts[1]:
            raise ValueError(f"non-canonical block index in Stage-1 key {key!r}")
        block_prefix = f"blocks.{int(parts[1])}"
        module = parts[2]
        if module in {"attn", "self_attn"}:
            if len(parts) != 5 or parts[3] not in _ATTENTION_PROJECTIONS:
                raise ValueError(f"Unsupported Stage-1 attention key {key!r}")
            leaf = "weight" if parts[4] == "kernel" else "bias" if parts[4] == "bias" else None
            if leaf is None:
                raise ValueError(f"Unsupported Stage-1 attention leaf {key!r}")
            return f"{block_prefix}.{module}.{_ATTENTION_PROJECTIONS[parts[3]]}.{leaf}", leaf == "weight"
        if module in {"norm_q", "norm_kv", "norm_s"}:
            if len(parts) != 4 or parts[3] not in {"scale", "bias"}:
                raise ValueError(f"Unsupported Stage-1 LayerNorm key {key!r}")
            leaf = "weight" if parts[3] == "scale" else "bias"
            return f"{block_prefix}.{module}.{leaf}", False
        if module == "ffn":
            if len(parts) != 4 or parts[3] not in {"kernel", "bias"}:
                raise ValueError(f"Unsupported Stage-1 feed-forward key {key!r}")
            leaf = "weight" if parts[3] == "kernel" else "bias"
            return f"{block_prefix}.ffn.{leaf}", leaf == "weight"
        raise ValueError(f"Unsupported Stage-1 block module {key!r}")

    if parts[0] in {"proj_cosmo", "proj_recent", "lang_film"}:
        if len(parts) != 2 or parts[1] not in {"kernel", "bias"}:
            raise ValueError(f"Unsupported Stage-1 projection key {key!r}")
        leaf = "weight" if parts[1] == "kernel" else "bias"
        return f"{parts[0]}.{leaf}", leaf == "weight"

    if parts[0] in {"tok_action", "tok_proprio"}:
        if len(parts) != 3 or parts[1] not in {"l1", "l2"} or parts[2] not in {"kernel", "bias"}:
            raise ValueError(f"Unsupported Stage-1 tokenizer key {key!r}")
        leaf = "weight" if parts[2] == "kernel" else "bias"
        return f"{parts[0]}.{parts[1]}.{leaf}", leaf == "weight"

    raise ValueError(f"Unsupported Stage-1 bottleneck key {key!r}")


def _convert_flax_value(key: str, value: np.ndarray, *, transpose: bool) -> Tensor:
    array = np.asarray(value)
    if array.dtype not in {np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"Stage-1 parameter {key!r} must use a floating dtype, got {array.dtype}")
    attention_key = "/attn/" in key or "/self_attn/" in key
    if attention_key:
        parts = key.split("/")
        projection = parts[-2]
        if parts[-1] == "kernel":
            if projection in {"query", "key", "value"}:
                array = array.reshape(array.shape[0], -1).T
            elif projection == "out":
                array = array.reshape(-1, array.shape[-1]).T
            else:  # pragma: no cover - guarded by flax_key_to_torch_key
                raise ValueError(f"Unsupported attention projection {key!r}")
        elif parts[-1] == "bias" and projection in {"query", "key", "value"}:
            array = array.reshape(-1)
    elif transpose:
        if array.ndim != 2:
            raise ValueError(f"Expected rank-2 dense kernel for {key!r}, got {array.shape}")
        array = array.T
    return torch.from_numpy(np.ascontiguousarray(array))


def load_stage1_bottleneck_state_dict(
    checkpoint_path: str | Path,
    encoder: MemoryBottleneckEncoder,
    *,
    strict: bool = True,
) -> dict[str, Tensor]:
    """Return the exact memory-encoder state dict represented by Stage-1 weights."""

    checkpoint_path = Path(checkpoint_path)
    converted: dict[str, Tensor] = {}
    unexpected: list[str] = []
    seen_source_keys: set[str] = set()
    expected_state = encoder.state_dict()
    with np.load(checkpoint_path, allow_pickle=False) as source:
        for key in source.files:
            if not key.startswith("bottleneck/"):
                continue
            if key in seen_source_keys:
                unexpected.append(f"duplicate Stage-1 parameter {key!r}")
                continue
            seen_source_keys.add(key)
            try:
                torch_key, transpose = flax_key_to_torch_key(key)
            except ValueError as exc:
                unexpected.append(str(exc))
                continue
            if torch_key in converted:
                unexpected.append(
                    f"duplicate Stage-1 destination {torch_key!r} from source key {key!r}"
                )
                continue
            converted[torch_key] = _convert_flax_value(
                key,
                source[key],
                transpose=transpose,
            ).to(dtype=expected_state.get(torch_key, torch.empty((), dtype=torch.float32)).dtype)

    expected = set(expected_state)
    matched = {key: value for key, value in converted.items() if key in expected}
    unexpected.extend(sorted(set(converted) - expected))
    shape_mismatches = {
        key: (tuple(value.shape), tuple(expected_state[key].shape))
        for key, value in matched.items()
        if tuple(value.shape) != tuple(expected_state[key].shape)
    }
    for key in shape_mismatches:
        del matched[key]
    missing = sorted(expected - set(matched))
    if strict and (missing or unexpected or shape_mismatches):
        raise KeyError(
            "missing Stage-1 parameters="
            f"{missing} unexpected Stage-1 parameters={sorted(unexpected)} "
            f"shape mismatch={shape_mismatches}"
        )
    return matched
