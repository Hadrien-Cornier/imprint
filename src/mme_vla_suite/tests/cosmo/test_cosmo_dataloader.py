"""CPU unit test for the cosmo_bottleneck training-dataloader gather path.

Fabricates a TINY synthetic per-episode Wan store on disk (no checkpoint, no GPU,
no real dataset) and exercises the standalone gather method
``RoboMMEDataset._gather_cosmo_bottleneck`` directly.

We test the standalone method (not full ``__getitem__``) because full construction
of ``RoboMMEDataset`` requires a ``DataConfig`` with norm_stats and pulls the whole
transform/model stack; the gather logic is pure Python/numpy over the on-disk store,
so we build the object via ``object.__new__`` and attach only the attributes the
method reads (``feature_dir``, ``history_config``). This keeps the test hermetic and
CPU-only per the task guidance ("Prefer the standalone-method approach").
"""
import json
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.cosmo import shapes as S


# --------------------------------------------------------------------------- #
# Synthetic tiny store                                                        #
# --------------------------------------------------------------------------- #
NUM_STEPS = 10
DOWNSAMPLE = 1
CHUNK_FRAMES = 4
NUM_VIEWS = 2
SPATIAL = 64            # V*spatial tokens per chunk row = NUM_VIEWS * SPATIAL
WAN_DIM = 48
SIGLIP_TOK = 64
SIGLIP_DIM = 2048
ACTION_DIM = 12
PROPRIO_DIM = 16
NUM_QUERY = 32


def _step_to_chunk(num_steps: int, chunk_frames: int, num_chunks: int) -> list[int]:
    # Mirrors build_wan_store: downsampled step i belongs to chunk (i//chunk_frames),
    # or -1 if that chunk is not (yet) finalized.
    return [
        (i // chunk_frames) if (i // chunk_frames) < num_chunks else -1
        for i in range(num_steps)
    ]


def _build_synthetic_store(root, num_chunks=2):
    """Write one tiny episode_0 under <root>/features/, plus a data/0.pkl sample."""
    epi_dir = os.path.join(root, "features", "episode_0")
    os.makedirs(epi_dir, exist_ok=True)

    # latents: (num_chunks, V*spatial, 48). Fill each chunk with its own index so the
    # gathered prefix is identity-checkable.
    v_spatial = NUM_VIEWS * SPATIAL
    latents = np.stack(
        [np.full((v_spatial, WAN_DIM), float(c), np.float32) for c in range(num_chunks)],
        axis=0,
    )
    np.save(os.path.join(epi_dir, "latents.npy"), latents)

    step_to_chunk = _step_to_chunk(NUM_STEPS, CHUNK_FRAMES, num_chunks)
    chunk_index = {
        "num_steps": NUM_STEPS,
        "chunk_frames": CHUNK_FRAMES,
        "downsample": DOWNSAMPLE,
        "num_views": NUM_VIEWS,
        "step_to_chunk": step_to_chunk,
        "num_chunks": num_chunks,
    }
    with open(os.path.join(epi_dir, "chunk_index.json"), "w") as f:
        json.dump(chunk_index, f)

    # recent_siglip: (steps, V, 64, 2048); encode step index into value.
    recent = np.stack(
        [np.full((NUM_VIEWS, SIGLIP_TOK, SIGLIP_DIM), float(t), np.float32)
         for t in range(NUM_STEPS)],
        axis=0,
    )
    np.save(os.path.join(epi_dir, "recent_siglip.npy"), recent)

    # align_prefix: (steps, 64, 2048); encode step index.
    align = np.stack(
        [np.full((SIGLIP_TOK, SIGLIP_DIM), float(t), np.float32) for t in range(NUM_STEPS)],
        axis=0,
    )
    np.save(os.path.join(epi_dir, "align_prefix.npy"), align)

    # action_proprio dict: action/da (steps,12); proprio/dp (steps,16).
    rng = np.random.default_rng(0)
    ap = {
        "action": rng.standard_normal((NUM_STEPS, ACTION_DIM)).astype(np.float32),
        "proprio": rng.standard_normal((NUM_STEPS, PROPRIO_DIM)).astype(np.float32),
        "da": rng.standard_normal((NUM_STEPS, ACTION_DIM)).astype(np.float32),
        "dp": rng.standard_normal((NUM_STEPS, PROPRIO_DIM)).astype(np.float32),
    }
    np.save(os.path.join(epi_dir, "action_proprio.npy"), ap, allow_pickle=True)
    return step_to_chunk, latents, recent, align, ap


def _cosmo_history_config(max_prefix_chunks=8):
    return OmegaConf.create(
        {
            "representation_type": "cosmo_bottleneck",
            "num_views": NUM_VIEWS,
            "streaming_obs_horizon": 16,
            "cosmo_bottleneck_memory": {
                "num_query_samples": NUM_QUERY,
                "max_prefix_chunks": max_prefix_chunks,
                "cosmos_latent_dim": WAN_DIM,
                "siglip_dim": SIGLIP_DIM,
                "action_in_dim": 24,
                "proprio_in_dim": 32,
            },
        }
    )


def _make_dataset(root, history_config):
    """Build a RoboMMEDataset shell exposing only what the gather method reads."""
    from mme_vla_suite.training.dataset import RoboMMEDataset

    ds = object.__new__(RoboMMEDataset)
    ds.history_config = history_config
    ds.feature_dir = os.path.join(root, "features")
    return ds


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_gather_cosmo_bottleneck_shapes_and_invariant(tmp_path):
    root = str(tmp_path)
    num_chunks = 2
    step_to_chunk, latents, recent, align, ap = _build_synthetic_store(root, num_chunks)
    hc = _cosmo_history_config()
    ds = _make_dataset(root, hc)

    step_idx = NUM_STEPS - 1  # t = 9
    d = ds._gather_cosmo_bottleneck(epis_idx=0, step_idx=step_idx)

    # --- feature dims ---
    assert d["cosmo_latent_emb"].shape[-1] == 48
    assert d["cosmo_recent_emb"].shape[-1] == 2048
    assert d["cosmo_action_emb"].shape[-1] == 24
    assert d["cosmo_proprio_emb"].shape[-1] == 32

    # --- query sampling / targets ---
    assert d["query_steps"].shape[0] == d["align_target"].shape[0]  # K matches
    assert d["align_target"].shape[1:] == (64, 2048)
    assert d["action_target"].shape[-1] == 12 and d["proprio_target"].shape[-1] == 16
    assert d["action_target"].shape[0] == d["query_steps"].shape[0]
    assert d["proprio_target"].shape[0] == d["query_steps"].shape[0]

    # K = min(32, num available steps in [0, t]) = min(32, 10) = 10
    assert d["query_steps"].shape[0] == min(NUM_QUERY, step_idx + 1)
    assert np.all(d["query_steps"] >= 0) and np.all(d["query_steps"] <= step_idx)

    # --- INVARIANT: gathered latent prefix == latents[:c(t)+1] flattened ---
    # last finalized chunk <= t: step_to_chunk has -1 for the terminal partial chunk,
    # so the effective c(t) is the max finalized index over [0, t].
    finalized = [c for c in step_to_chunk[: step_idx + 1] if c >= 0]
    c = max(finalized)
    expected_prefix = latents[: c + 1].reshape(-1, 48)
    expected_valid_tokens = expected_prefix.shape[0]

    assert d["cosmo_mem_mask"].dtype == bool
    assert int(d["cosmo_mem_mask"].sum()) == expected_valid_tokens
    # valid rows equal the true prefix (order preserved).
    valid = d["cosmo_latent_emb"][d["cosmo_mem_mask"]]
    assert valid.shape == expected_prefix.shape
    assert np.array_equal(valid, expected_prefix)
    # padded rows are zero.
    if d["cosmo_latent_emb"].shape[0] > expected_valid_tokens:
        assert np.all(d["cosmo_latent_emb"][~d["cosmo_mem_mask"]] == 0.0)

    # --- action/proprio emb == concat(raw, delta) for steps <= t ---
    exp_action_emb = np.concatenate([ap["action"][: step_idx + 1], ap["da"][: step_idx + 1]], -1)
    exp_proprio_emb = np.concatenate([ap["proprio"][: step_idx + 1], ap["dp"][: step_idx + 1]], -1)
    assert np.allclose(d["cosmo_action_emb"], exp_action_emb)
    assert np.allclose(d["cosmo_proprio_emb"], exp_proprio_emb)

    # --- targets read the RAW (12/16) executed action/proprio at query_steps ---
    qs = d["query_steps"]
    assert np.allclose(d["action_target"], ap["action"][qs])
    assert np.allclose(d["proprio_target"], ap["proprio"][qs])
    assert np.allclose(d["align_target"], align[qs])

    # --- recent window: from last finalized boundary to t, all V views ---
    boundary = (c + 1) * CHUNK_FRAMES
    n_recent = step_idx - boundary + 1
    assert d["cosmo_recent_emb"].shape[0] == n_recent * NUM_VIEWS * SIGLIP_TOK
    exp_recent = recent[boundary : step_idx + 1].reshape(-1, SIGLIP_DIM)
    assert np.array_equal(d["cosmo_recent_emb"], exp_recent)


def test_query_steps_capped_by_K(tmp_path):
    """When available steps exceed K, K == num_query_samples."""
    root = str(tmp_path)
    _build_synthetic_store(root, num_chunks=2)
    hc = _cosmo_history_config()
    hc.cosmo_bottleneck_memory.num_query_samples = 4
    ds = _make_dataset(root, hc)
    d = ds._gather_cosmo_bottleneck(epis_idx=0, step_idx=NUM_STEPS - 1)
    assert d["query_steps"].shape[0] == 4
    assert d["align_target"].shape[0] == 4
