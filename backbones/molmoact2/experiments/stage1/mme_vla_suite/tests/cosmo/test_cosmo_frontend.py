import numpy as np
import pytest
from tests.cosmo import shapes as S

torch = pytest.importorskip("torch")

def test_frontend_encode_shapes_and_frozen(tmp_path):
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    fe = CosmoFrontend(variant="CV8x8x8", device="cuda")
    # 16 downsampled frames -> 2 complete chunks at temporal factor 8
    frames = (np.random.default_rng(0).integers(0, 256, (16, 224, 224, 3))).astype(np.uint8)
    lat = fe.encode(frames)                       # (num_chunks, spatial_tokens, latent_dim)
    assert lat.ndim == 3
    assert lat.shape[0] == 16 // S.COSMOS_TEMPORAL_FACTOR
    assert lat.shape[2] == S.COSMOS_LATENT_DIM
    # frozen: no parameter requires grad
    assert all(not p.requires_grad for p in fe.parameters())
    # deterministic: same input -> same latents
    lat2 = fe.encode(frames)
    np.testing.assert_allclose(lat, lat2, rtol=0, atol=0)


def test_prefix_consistency_finalized():
    """Finalized (non-terminal) chunks must not change when future frames are appended (true causality)."""
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    fe = CosmoFrontend(variant="CV8x8x8", device="cuda")
    rng = np.random.default_rng(1)
    full = rng.integers(0, 256, (40, 224, 224, 3)).astype(np.uint8)   # 5 chunks -> 4 finalized
    lat_full_fin = fe.finalized_encode(full)                          # 4 finalized chunks
    lat_prefix_fin = fe.finalized_encode(full[:24])                   # 3 chunks -> 2 finalized
    print(f"\n[prefix_consistency_finalized] max_abs_diff={np.max(np.abs(lat_full_fin[:2] - lat_prefix_fin)):.6e}")
    np.testing.assert_allclose(lat_full_fin[:2], lat_prefix_fin, rtol=1e-2, atol=1e-2)


def test_streaming_equals_full_pass_finalized():
    """Stateful streaming of finalized chunks must equal the full-pass finalized chunks (within bf16 noise)."""
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    fe = CosmoFrontend(variant="CV8x8x8", device="cuda")
    rng = np.random.default_rng(2)
    full = rng.integers(0, 256, (40, 224, 224, 3)).astype(np.uint8)
    lat_full_fin = fe.finalized_encode(full)
    lat_stream = fe.streaming_encode(full, chunk_frames=8)            # your stateful impl, finalized chunks
    assert lat_stream.shape[0] == lat_full_fin.shape[0]
    print(f"\n[streaming_equals_full_finalized] max_abs_diff={np.max(np.abs(lat_full_fin - lat_stream)):.6e}")
    np.testing.assert_allclose(lat_full_fin, lat_stream, rtol=2e-2, atol=2e-2)


def test_compiled_matches_eager():
    """Compiled encode must match eager encode within bf16 tolerance, same shape."""
    import numpy as np
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    rng = np.random.default_rng(3)
    frames = rng.integers(0,256,(16,224,224,3)).astype(np.uint8)
    fe_eager = CosmoFrontend(variant="CV8x8x8", device="cuda", compile=False)
    fe_comp  = CosmoFrontend(variant="CV8x8x8", device="cuda", compile=True)
    le = fe_eager.encode(frames)
    lc = fe_comp.encode(frames)
    assert le.shape == lc.shape
    np.testing.assert_allclose(le, lc, rtol=2e-2, atol=2e-2)


def test_warmup_then_encode():
    """warmup(shape) must not break a subsequent real encode of that shape."""
    import numpy as np
    from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
    fe = CosmoFrontend(variant="CV8x8x8", device="cuda", compile=True)
    fe.warmup((8, 224, 224))
    frames = np.random.default_rng(4).integers(0,256,(8,224,224,3)).astype(np.uint8)
    lat = fe.encode(frames)
    assert lat.shape[0] == 8 // fe.temporal_factor
    # determinism cache still honored after warmup
    lat2 = fe.encode(frames)
    np.testing.assert_allclose(lat, lat2, rtol=0, atol=0)
