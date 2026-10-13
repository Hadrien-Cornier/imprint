import numpy as np
import pytest
from tests.cosmo import shapes as S

torch = pytest.importorskip("torch")

def test_wan_encode_multiview_concats_token_axis():
    from mme_vla_suite.models.representation.wan_frontend import WanFrontend
    fe = WanFrontend(device="cuda")
    rng = np.random.default_rng(0)
    T = 8  # 2 finalized chunks at temporal_factor 4
    v1 = rng.integers(0,256,(T,128,128,3)).astype(np.uint8)
    lat1 = fe.encode(v1)                       # (nc, spatial, 48)
    v3 = rng.integers(0,256,(T,3,128,128,3)).astype(np.uint8)
    lat3 = fe.encode_multiview(v3)             # (nc, 3*spatial, 48)
    assert lat3.shape[0] == lat1.shape[0]                       # same #chunks
    assert lat3.shape[1] == 3 * lat1.shape[1]                   # V x spatial tokens
    assert lat3.shape[2] == lat1.shape[2] == S.WAN_LATENT_DIM
    # view 0 slice of the multiview equals the single-view encode of that view
    v3_view0 = np.ascontiguousarray(v3[:,0])
    lat_v0 = fe.encode(v3_view0)
    np.testing.assert_allclose(lat3[:, :lat1.shape[1]], lat_v0, rtol=2e-2, atol=2e-2)

def test_wan_finalized_multiview():
    from mme_vla_suite.models.representation.wan_frontend import WanFrontend
    fe = WanFrontend(device="cuda")
    rng = np.random.default_rng(1)
    v3 = rng.integers(0,256,(12,3,128,128,3)).astype(np.uint8)   # 3 chunks -> 2 finalized
    fin = fe.finalized_encode_multiview(v3)
    full = fe.encode_multiview(v3)
    assert fin.shape[0] == full.shape[0] - 1                     # dropped 1 terminal chunk
    assert fin.shape[1] == full.shape[1]                         # same token width (3*spatial)
