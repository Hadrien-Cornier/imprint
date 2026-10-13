from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "imprint_validate_store",
    ROOT / "scripts/reproduce/validate_store.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATE_STORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE_STORE)
validate_episode = VALIDATE_STORE.validate_episode


def write_episode(path, *, globally_indexed_causal: bool) -> None:
    path.mkdir()
    action = np.arange(12, dtype=np.float32).reshape(6, 2)
    payload = {
        "action": action,
        "proprio": np.zeros((6, 3), dtype=np.float32),
    }
    np.save(path / "action_proprio.npy", payload)
    np.save(path / "dinov3.npy", np.zeros((6, 3, 64, 768), dtype=np.float16))
    (path / "chunk_index.json").write_text(json.dumps({"chunk_frames": 2}))

    frame_idx = np.asarray([0, 2, 4], dtype=np.int64)
    variant_index = {}
    global_causal = np.zeros_like(action)
    global_causal[1:] = action[:-1]
    for variant in range(8):
        variant_index[f"v{variant}"] = {
            "frame_idx": frame_idx.tolist(),
            "n_frames": len(frame_idx),
            "n_chunks": 1,
        }
        sampled = action[frame_idx]
        causal = np.zeros_like(sampled)
        causal[1:] = sampled[:-1]
        if globally_indexed_causal:
            causal = global_causal[frame_idx]
        np.save(path / f"action_causal_v{variant}.npy", causal)
        np.save(
            path / f"latents_v{variant}.npy",
            np.zeros((1, 12, 48), dtype=np.float32),
        )
    (path / "variant_index.json").write_text(json.dumps(variant_index))


def test_validate_episode_accepts_variant_local_causal_actions(tmp_path) -> None:
    episode = tmp_path / "episode_0"
    write_episode(episode, globally_indexed_causal=False)

    validate_episode(episode, action_dim=2, proprio_dim=3)


def test_validate_episode_rejects_globally_indexed_causal_actions(tmp_path) -> None:
    episode = tmp_path / "episode_0"
    write_episode(episode, globally_indexed_causal=True)

    with pytest.raises(ValueError, match="sampled variant"):
        validate_episode(episode, action_dim=2, proprio_dim=3)


def test_validate_episode_rejects_unfinalized_terminal_chunk(tmp_path) -> None:
    episode = tmp_path / "episode_0"
    write_episode(episode, globally_indexed_causal=False)
    np.save(
        episode / "latents_v0.npy",
        np.zeros((2, 12, 48), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="expected 1 chunks"):
        validate_episode(episode, action_dim=2, proprio_dim=3)
