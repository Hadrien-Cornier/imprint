import numpy as np
from tests.cosmo import shapes as S

class _FakeCosmo:
    temporal_factor = 8
    def encode(self, frames):  # (n*8,H,W,3) -> (n, 784, 16)
        n = frames.shape[0] // 8
        return np.full((n, S.COSMOS_SPATIAL_TOKENS, S.COSMOS_LATENT_DIM), float(frames.shape[0]), np.float32)

class _FakeSiglip:
    def __call__(self, x):  # (t,v,224,224,3) -> (t,v,64,2048)
        t, v = x.shape[0], x.shape[1]
        return np.zeros((t, v, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

class _CausalFakeCosmo:
    """STRICTLY-CAUSAL, CONTENT-DEPENDENT fake tokenizer used to prove latent VALUES (test A).

    Chunk c's latent = a scalar function of the SUM of every frame in the growing prefix up to
    and including chunk c (frames[0:(c+1)*8]), broadcast across all spatial tokens/dims. This is:
      * history-dependent: encoding chunk c's 8 frames STANDALONE (what the OLD buffer did) sees
        only frames[c*8:(c+1)*8], giving a DIFFERENT value for every c>=1 than the growing-prefix
        encode -> the old standalone code FAILS this test at chunk 1+.
      * causally stable: chunk c never depends on frames after (c+1)*8, so its finalized value is
        invariant to future appends -> the growing-prefix buffer equals the full-episode encode.
    """
    temporal_factor = 8

    def encode(self, frames):  # (N,H,W,3) -> (N//8, 784, 16)
        n = frames.shape[0] // self.temporal_factor
        vals = np.asarray(frames, np.float64)
        out = np.zeros((n, S.COSMOS_SPATIAL_TOKENS, S.COSMOS_LATENT_DIM), np.float32)
        for c in range(n):
            out[c] = np.float32(vals[: (c + 1) * self.temporal_factor].sum())
        return out

    def encode_multiview(self, frames_v):  # (N,V,H,W,3) -> (N//8, V*784, 16)
        V = frames_v.shape[1]
        per_view = [self.encode(np.ascontiguousarray(frames_v[:, v])) for v in range(V)]
        return np.concatenate(per_view, axis=1)


class _CosmosAccountingFake:
    """Mimics the REAL Cosmos CV8x8x8 chunk accounting to reproduce the IndexError regression
    on CPU (no cosmos_tokenizer / GPU needed).

    Cosmos emits 1 + T//tf raw temporal latents and drops the leading warm-up latent, so encode
    yields T//tf chunks; finalized_encode drops the terminal (provisional) chunk on top of that,
    yielding max(0, T//tf - 1) finalized chunks. This is ONE LESS than a naive pop cadence that
    finalizes a chunk as soon as a single next-chunk frame exists -- exactly the off-by-one that
    made the old buffer index past finalized_encode's output and raise IndexError."""
    temporal_factor = 8
    latent_dim = S.COSMOS_LATENT_DIM

    def encode(self, frames):  # (N,H,W,3) -> (N//tf, 784, 16)
        n = frames.shape[0] // self.temporal_factor
        return np.full((n, S.COSMOS_SPATIAL_TOKENS, self.latent_dim), float(frames.shape[0]), np.float32)

    def finalized_encode(self, frames):  # drop-terminal: max(0, N//tf - 1) chunks
        n = max(0, frames.shape[0] // self.temporal_factor - 1)
        return np.full((n, S.COSMOS_SPATIAL_TOKENS, self.latent_dim), float(frames.shape[0]), np.float32)

    def finalized_encode_multiview(self, frames_v):
        V = frames_v.shape[1]
        per_view = [self.finalized_encode(np.ascontiguousarray(frames_v[:, v])) for v in range(V)]
        n = per_view[0].shape[0]
        return np.concatenate(per_view, axis=1) if n > 0 else per_view[0]


def test_consolidation_no_index_overrun():
    """REGRESSION (guards the fix to commit 3a0b1f3): a Cosmos-accounting frontend whose
    finalized_encode drops a leading warm-up latent must NOT trigger an IndexError. The old
    buffer used an independent pop-loop counter that committed chunk index c before
    finalized_encode(all)[c] existed -> IndexError at shared/cosmo_buffer.py `return finalized[c]`.
    Drives the exact e2e config (chunk_frames=8, downsample=2) to 20 root steps == 10 downsampled
    frames, where finalized_encode yields max(0, 10//8 - 1) = 0 chunks."""
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    for num_views in (1, 2):
        buf = CosmoMemoryBuffer(cosmo=_CosmosAccountingFake(), siglip_fn=_FakeSiglip(),
                                chunk_frames=8, downsample=2, num_views=num_views,
                                state_dim=S.STATE_DIM, action_dim=12, latent_dim=S.COSMOS_LATENT_DIM)
        rng = np.random.default_rng(0)
        for step in range(20):  # must not raise IndexError at any step
            buf.add_buffer(images=rng.integers(0, 256, (1, num_views, 128, 128, 3)).astype(np.uint8),
                           states=np.zeros((1, S.STATE_DIM), np.float32),
                           actions=np.zeros((1, 12), np.float32), step_idx_list=[step])
        feats = buf.get_history_feats(19)
        # 10 downsampled frames -> max(0, 10//8 - 1) = 0 finalized chunks (matches offline store).
        assert feats["cosmo_latent_emb"].shape[0] == 0
        assert feats["cosmo_recent_emb"].shape[0] == 10 * num_views * S.SIGLIP_TOKENS_PER_FRAME

    # And push far enough to actually finalize chunks, still no overrun and correct counts:
    buf = CosmoMemoryBuffer(cosmo=_CosmosAccountingFake(), siglip_fn=_FakeSiglip(),
                            chunk_frames=8, downsample=1, num_views=1,
                            state_dim=S.STATE_DIM, action_dim=12, latent_dim=S.COSMOS_LATENT_DIM)
    rng = np.random.default_rng(1)
    for step in range(24):  # 24 frames -> finalized = 24//8 - 1 = 2 chunks
        buf.add_buffer(images=rng.integers(0, 256, (1, 1, 128, 128, 3)).astype(np.uint8),
                       states=np.zeros((1, S.STATE_DIM), np.float32),
                       actions=np.zeros((1, 12), np.float32), step_idx_list=[step])
    assert buf.get_history_feats(23)["cosmo_latent_emb"].shape[0] == 2 * S.COSMOS_SPATIAL_TOKENS


def _mk_buf(chunk_frames=8, downsample=1):
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    return CosmoMemoryBuffer(cosmo=_FakeCosmo(), siglip_fn=_FakeSiglip(),
                             chunk_frames=chunk_frames, downsample=downsample, num_views=1,
                             state_dim=S.STATE_DIM, action_dim=12, latent_dim=S.COSMOS_LATENT_DIM)

def _step(buf, step):
    buf.add_buffer(images=np.zeros((1,1,128,128,3), np.uint8),
                   states=np.zeros((1,S.STATE_DIM), np.float32),
                   actions=np.zeros((1,12), np.float32), step_idx_list=[step])

def test_drop_terminal_cadence_and_recency():
    """DROP-TERMINAL cadence, defined by the SINGLE source of truth: the frontend's finalized
    encode over the growing prefix (encode()[:-1] for a plain-encode fake). A chunk finalizes
    only once the FULL NEXT chunk exists (that terminal chunk is dropped), i.e. chunk 0 finalizes
    at 2*chunk_frames=16 downsampled frames, NOT at chunk_frames+1=9.

    NOTE (regression fix, commit follow-up to 3a0b1f3): the previous expectation "9th frame -> 1
    consolidated / 1 recent" came from an independent pop-loop counter that finalized a chunk as
    soon as ONE next-chunk frame arrived. That counter disagreed with the frontend's own terminal
    drop (Cosmos drops a leading warm-up latent), causing an IndexError on the real-Cosmos e2e.
    The buffer now derives BOTH the consolidated count and the recency window from the frontend's
    finalized_encode count, so the correct drop-terminal cadence is 8->0/8recent, 15->0/15recent,
    16->1/8recent. Counts updated to this correct accounting; token VALUES remain causal."""
    buf = _mk_buf(chunk_frames=8, downsample=1)
    for s in range(8):
        _step(buf, s)
    f8 = buf.get_history_feats(7)
    assert f8["cosmo_latent_emb"].shape[0] == 0                          # 8 frames: no full next chunk yet
    assert f8["cosmo_recent_emb"].shape[0] == 8 * S.SIGLIP_TOKENS_PER_FRAME
    assert f8["cosmo_action_emb"].shape[1] == S.ACTION_IN_DIM
    assert f8["cosmo_proprio_emb"].shape[1] == S.PROPRIO_IN_DIM
    # 9th..15th frame: chunk 0 is complete but still TERMINAL (next chunk not yet full) -> 0 finalized.
    for s in range(8, 15):
        _step(buf, s)
    f15 = buf.get_history_feats(14)
    assert f15["cosmo_latent_emb"].shape[0] == 0                          # still terminal, dropped
    assert f15["cosmo_recent_emb"].shape[0] == 15 * S.SIGLIP_TOKENS_PER_FRAME
    # 16th frame completes chunk 1 -> chunk 0 becomes non-terminal -> exactly 1 finalized chunk,
    # recency window resets to the 8 frames of the (now provisional) chunk 1.
    _step(buf, 15)
    f16 = buf.get_history_feats(15)
    assert f16["cosmo_latent_emb"].shape[0] == S.COSMOS_SPATIAL_TOKENS    # 1 finalized chunk
    assert f16["cosmo_recent_emb"].shape[0] == 8 * S.SIGLIP_TOKENS_PER_FRAME

def test_language_passthrough():
    buf = _mk_buf()
    buf.set_language(np.arange(S.LANG_EMB_DIM, dtype=np.float32))
    _step(buf, 0)
    f = buf.get_history_feats(0)
    assert f["cosmo_lang_emb"].shape == (S.LANG_EMB_DIM,)
    buf.set_language(None)
    _step(buf, 1)
    assert buf.get_history_feats(1)["cosmo_lang_emb"] is None

def test_delta_values():
    buf = _mk_buf()
    buf.add_buffer(images=np.zeros((2,1,128,128,3),np.uint8),
                   states=np.stack([np.full((S.STATE_DIM,),1.0,np.float32), np.full((S.STATE_DIM,),3.0,np.float32)]),
                   actions=np.stack([np.full((12,),5.0,np.float32), np.full((12,),8.0,np.float32)]),
                   step_idx_list=[0,1])
    f = buf.get_history_feats(1)
    a = f["cosmo_action_emb"]
    assert np.allclose(a[0, 12:], 0.0)
    assert np.allclose(a[1, 12:], 3.0)
    p = f["cosmo_proprio_emb"]
    assert np.allclose(p[1, 16:], 2.0)


def _mk_causal_buf(num_views=1, chunk_frames=8, downsample=1):
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    return CosmoMemoryBuffer(cosmo=_CausalFakeCosmo(), siglip_fn=_FakeSiglip(),
                             chunk_frames=chunk_frames, downsample=downsample, num_views=num_views,
                             state_dim=S.STATE_DIM, action_dim=12, latent_dim=S.COSMOS_LATENT_DIM)


def _drive(buf, frames_v):
    """Feed a (T, V, H, W, 3) uint8 clip to the buffer one root step at a time (downsample=1)."""
    T, V = frames_v.shape[0], frames_v.shape[1]
    for t in range(T):
        buf.add_buffer(images=frames_v[t][None],
                       states=np.zeros((1, S.STATE_DIM), np.float32),
                       actions=np.zeros((1, 12), np.float32), step_idx_list=[t])


def test_consolidation_is_causal_value_invariant_single_view():
    """Test A (CPU): with a strictly-causal CONTENT-dependent fake, each consolidated chunk c
    must equal the c-th chunk of a causal encode over the WHOLE downsampled clip. This FAILS on
    the old standalone-chunk code (chunk 1+ diverges) and PASSES with growing-prefix consolidation."""
    T, V, H, W = 40, 1, 8, 8  # 40 frames / cf=8 -> 5 encode chunks, 4 finalized after drop-terminal
    rng = np.random.default_rng(0)
    frames_v = rng.integers(0, 256, (T, V, H, W, 3), dtype=np.uint8)

    buf = _mk_causal_buf(num_views=V)
    _drive(buf, frames_v)
    online = buf.get_history_feats(T - 1)["cosmo_latent_emb"]  # (nchunks*784, 16)

    ref = _CausalFakeCosmo().encode_multiview(frames_v)        # (5, 784, 16) causally-correct
    n_online = online.shape[0] // S.COSMOS_SPATIAL_TOKENS
    assert n_online >= 2, f"need >=2 finalized chunks to expose the bug, got {n_online}"
    ref_prefix = ref[:n_online].reshape(-1, S.COSMOS_LATENT_DIM)
    assert online.shape == ref_prefix.shape
    max_abs = float(np.max(np.abs(online - ref_prefix)))
    assert np.allclose(online, ref_prefix), f"buffer consolidation not causal: max_abs_diff={max_abs}"


def test_consolidation_is_causal_value_invariant_multiview():
    """Test A, multiview (V=2): same causal-value invariant through the encode_multiview path."""
    T, V, H, W = 40, 2, 8, 8
    rng = np.random.default_rng(1)
    frames_v = rng.integers(0, 256, (T, V, H, W, 3), dtype=np.uint8)

    buf = _mk_causal_buf(num_views=V)
    _drive(buf, frames_v)
    online = buf.get_history_feats(T - 1)["cosmo_latent_emb"]  # (nchunks*V*784, 16)

    ref = _CausalFakeCosmo().encode_multiview(frames_v)        # (5, V*784, 16)
    tokens_per_chunk = V * S.COSMOS_SPATIAL_TOKENS
    n_online = online.shape[0] // tokens_per_chunk
    assert n_online >= 2
    ref_prefix = ref[:n_online].reshape(-1, S.COSMOS_LATENT_DIM)
    assert online.shape == ref_prefix.shape
    assert np.allclose(online, ref_prefix), "multiview buffer consolidation not causal"
