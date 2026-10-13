"""Tests for the offline Wan-latent memory store builder (sub-project A).

test_build_episode_shapes (CPU): drives build_episode with FAKE feature
extractors on a tiny synthetic episode; proves the per-episode packing/layout
(4 output files + chunk_index.json) without any checkpoint.

test_store_prefix_equals_online_buffer (GPU): THE core invariant. The offline
store's finalized Wan-latent chunk prefix up to root step t must equal the online
CosmoMemoryBuffer's consolidated latents at step t, byte-for-byte (fp tol). Uses
the REAL WanFrontend.
"""
import json
import os

import numpy as np
import pytest

from tests.cosmo import shapes as S


# --------------------------------------------------------------------------- #
# PART B: CPU shape test with FAKES                                           #
# --------------------------------------------------------------------------- #
def test_build_episode_shapes(tmp_path):
    from mme_vla_suite.dataset_builder.build_wan_store import build_episode

    T, V, H, W = 12, 3, 128, 128
    downsample = 1
    chunk_frames = S.WAN_TEMPORAL_FACTOR  # 4
    rng = np.random.default_rng(0)
    frames_v = rng.integers(0, 256, (T, V, H, W, 3), dtype=np.uint8)
    states = rng.standard_normal((T, S.STATE_DIM)).astype(np.float32)
    actions = rng.standard_normal((T, 12)).astype(np.float32)

    class _FakeWan:
        temporal_factor = chunk_frames
        latent_dim = S.WAN_LATENT_DIM

        def finalized_encode_multiview(self, fv):
            # (T', V, H, W, 3) -> (num_chunks, V*spatial, 48); drop-terminal
            n = fv.shape[0] // self.temporal_factor
            n = max(n - 1, 0)
            Vv = fv.shape[1]
            return np.full(
                (n, Vv * S.WAN_SPATIAL_TOKENS_128, self.latent_dim), 0.5, np.float32
            )

    def fake_siglip_fn(x):  # (t, V, H, W, 3) -> (t, V, 64, 2048)
        t, v = x.shape[0], x.shape[1]
        return np.zeros((t, v, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

    def fake_prefix_fn(x):  # (t, H, W, 3) front view -> (t, 64, 2048)
        t = x.shape[0]
        return np.zeros((t, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

    out_dir = tmp_path / "features" / "episode_0"
    samples = build_episode(
        frames_v=frames_v,
        states=states,
        actions=actions,
        wan_frontend=_FakeWan(),
        siglip_fn=fake_siglip_fn,
        prefix_fn=fake_prefix_fn,
        downsample=downsample,
        chunk_frames=chunk_frames,
        out_dir=str(out_dir),
        episode_id=0,
    )

    # steps after downsampling
    n_steps = len(range(0, T, downsample))

    latents = np.load(out_dir / "latents.npy")
    recent = np.load(out_dir / "recent_siglip.npy")
    align = np.load(out_dir / "align_prefix.npy")
    ap = np.load(out_dir / "action_proprio.npy", allow_pickle=True).item()
    with open(out_dir / "chunk_index.json") as f:
        ci = json.load(f)

    # finalized chunk count: n_steps//chunk_frames - 1 (drop-terminal), >=0
    exp_chunks = max(n_steps // chunk_frames - 1, 0)
    assert latents.shape == (exp_chunks, V * S.WAN_SPATIAL_TOKENS_128, S.WAN_LATENT_DIM)
    assert recent.shape == (n_steps, V, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM)
    assert align.shape == (n_steps, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM)

    # action/proprio + deltas
    assert ap["action"].shape == (n_steps, 12)
    assert ap["proprio"].shape == (n_steps, S.STATE_DIM)
    assert ap["da"].shape == (n_steps, 12)
    assert ap["dp"].shape == (n_steps, S.STATE_DIM)

    # chunk_index metadata
    assert ci["num_steps"] == n_steps
    assert ci["chunk_frames"] == chunk_frames
    assert ci["downsample"] == downsample
    assert ci["num_views"] == V
    assert ci["num_chunks"] == exp_chunks
    assert len(ci["step_to_chunk"]) == n_steps

    # samples enumerate (episode, root_step_t) for every step
    assert len(samples) == n_steps
    assert samples[0] == (0, 0)


# --------------------------------------------------------------------------- #
# PART C: THE CORE INVARIANT (GPU)                                            #
# --------------------------------------------------------------------------- #
def _run_store_vs_buffer(reference):
    """Shared driver. Builds a synthetic 24-frame 2-view episode, runs the OFFLINE
    store Wan-latent path (encode-once + slice) and the ONLINE incremental path, and
    returns (offline_prefix, online, max_abs_diff, n_online_chunks).

    reference selects the ONLINE path:
      "buffer"  -> CosmoMemoryBuffer.get_history_feats(t)["cosmo_latent_emb"]
      "causal"  -> WanFrontend.streaming_encode (gate-blessed causal-incremental path)
    """
    from mme_vla_suite.models.representation.wan_frontend import WanFrontend
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    from mme_vla_suite.dataset_builder.build_wan_store import _downsample_frames

    fe = WanFrontend(device="cuda")
    chunk_frames = fe.temporal_factor  # 4
    ld = fe.latent_dim  # 48
    downsample = 2
    V = 2
    H = W = 128
    T = 24  # raw frames

    rng = np.random.default_rng(0)
    frames_v = rng.integers(0, 256, (T, V, H, W, 3), dtype=np.uint8)
    states = np.zeros((T, S.STATE_DIM), np.float32)
    actions = np.zeros((T, 12), np.float32)

    # OFFLINE store path: downsample once, finalized_encode_multiview once.
    ds_frames = _downsample_frames(frames_v, downsample)  # (t, V, H, W, 3)
    offline = fe.finalized_encode_multiview(ds_frames)    # (num_chunks, V*spatial, 48)
    offline_flat = offline.reshape(-1, ld)

    if reference == "causal":
        # Gate-blessed causal-incremental path, per view, concatenated on tokens,
        # exactly like finalized_encode_multiview but built chunk-by-chunk.
        per_view = [fe.streaming_encode(np.ascontiguousarray(ds_frames[:, v]))
                    for v in range(V)]
        nc = per_view[0].shape[0]
        online = np.concatenate(per_view, axis=1).reshape(-1, ld) if nc > 0 else per_view[0].reshape(-1, ld)
    else:
        def siglip_fn(x):
            t, v = x.shape[0], x.shape[1]
            return np.zeros((t, v, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

        buf = CosmoMemoryBuffer(
            cosmo=fe, siglip_fn=siglip_fn, chunk_frames=chunk_frames,
            downsample=downsample, num_views=V, state_dim=S.STATE_DIM,
            action_dim=12, latent_dim=ld,
        )
        buf.set_language(None)
        for step in range(T):
            buf.add_buffer(
                images=frames_v[step][None],
                states=states[step][None],
                actions=actions[step][None],
                step_idx_list=[step],
            )
        online = buf.get_history_feats(T - 1)["cosmo_latent_emb"]

    n_online_chunks = online.shape[0] // (V * S.WAN_SPATIAL_TOKENS_128)
    offline_prefix = offline_flat[: online.shape[0]]
    max_abs_diff = float(np.max(np.abs(offline_prefix - online))) if online.shape[0] else float("nan")
    return offline, offline_prefix, online, max_abs_diff, n_online_chunks


@pytest.mark.gpu
def test_store_prefix_equals_causal_stream():
    """CORE INVARIANT (builder side): the OFFLINE store's encode-once-and-slice Wan-latent
    prefix EQUALS the causal-incremental path (WanFrontend.streaming_encode) byte-for-byte.

    This proves the store builder's encode-once path reproduces the causally-correct
    incremental consolidation the online memory is *supposed* to produce. Wan is
    fp32-deterministic + finalized-chunk-stable, so expect ~0.
    """
    offline, offline_prefix, online, mad, nck = _run_store_vs_buffer("causal")
    print(f"\n[INVARIANT store-vs-causal] max_abs_diff = {mad:.3e} "
          f"(offline chunks={offline.shape[0]}, online chunks={nck})")
    assert nck > 0, "no finalized chunks produced"
    assert offline_prefix.shape == online.shape
    assert np.allclose(offline_prefix, online, rtol=2e-2, atol=2e-2), (
        f"store encode-once != causal-incremental: max_abs_diff={mad}"
    )


@pytest.mark.gpu
def test_store_prefix_equals_online_buffer():
    """CORE INVARIANT (online side): the OFFLINE store's encode-once-and-slice Wan-latent prefix
    EQUALS the ONLINE CosmoMemoryBuffer's consolidated latents byte-for-byte (fp tol).

    Previously xfail(strict): the buffer encoded each finalized chunk as a STANDALONE clip (zero
    temporal history), breaking Wan2.2's strict cross-chunk causal continuity -- the leading chunk
    matched (max_abs_diff 0.0) but every subsequent chunk diverged (~1.1). The buffer now
    consolidates via the GROWING-PREFIX causal encode (frontend.finalized_encode_multiview over all
    downsampled frames so far), matching the store's encode-once path, so this now PASSES.
    Prints the measured max-abs-diff so the (now ~0) agreement is visible in CI logs."""
    offline, offline_prefix, online, mad, nck = _run_store_vs_buffer("buffer")
    print(f"\n[INVARIANT store-vs-buffer] max_abs_diff = {mad:.3e} "
          f"(offline chunks={offline.shape[0]}, online chunks={nck})")
    assert nck > 0, "no finalized chunks produced"
    assert offline_prefix.shape == online.shape
    assert np.allclose(offline_prefix, online, rtol=2e-2, atol=2e-2), (
        f"invariant broken (buffer uses standalone-chunk encoding): max_abs_diff={mad}"
    )
