import numpy as np
from tests.cosmo import shapes as S

class _FakeCosmo:
    temporal_factor = 8
    def encode(self, frames):
        n = frames.shape[0] // 8
        return np.full((n, S.COSMOS_SPATIAL_TOKENS, S.COSMOS_LATENT_DIM), 1.0, np.float32)
def _fake_siglip(x):
    t, v = x.shape[0], x.shape[1]
    return np.zeros((t, v, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

def _buf(mode):
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    return CosmoMemoryBuffer(cosmo=_FakeCosmo(), siglip_fn=_fake_siglip, chunk_frames=8,
                             downsample=1, num_views=1, state_dim=S.STATE_DIM, action_dim=12,
                             latent_dim=S.COSMOS_LATENT_DIM, action_encoding=mode)

def _step(buf, step, action):
    buf.add_buffer(images=np.zeros((1,1,128,128,3),np.uint8),
                   states=np.zeros((1,S.STATE_DIM),np.float32),
                   actions=action[None].astype(np.float32), step_idx_list=[step])

def test_raw_delta_default_unchanged():
    # default mode must be raw_delta, width 24, delta semantics as before
    buf = _buf("raw_delta")
    _step(buf, 0, np.full((12,), 5.0)); _step(buf, 1, np.full((12,), 8.0))
    a = buf.get_history_feats(1)["cosmo_action_emb"]
    assert a.shape[1] == 24
    assert np.allclose(a[1,12:], 3.0)   # delta = 8-5

def test_geometric_mode_shape_and_6d():
    buf = _buf("geometric")
    # eef_pos=[0.1,0.2,0.3], eef_rot(rotvec)=[0,0,0] (identity), gripper=0.7, base=[1,2,3,4], mode=1
    act = np.array([0.1,0.2,0.3, 0,0,0, 0.7, 1,2,3,4, 1.0], np.float32)
    _step(buf, 0, act)
    a = buf.get_history_feats(0)["cosmo_action_emb"]
    assert a.shape[1] == 15                      # 3 + 6 + 1 + 4 + 1
    # identity rotvec -> 6D = first two cols of I = [1,0,0, 0,1,0]
    np.testing.assert_allclose(a[0, 3:9], [1,0,0, 0,1,0], atol=1e-5)
    np.testing.assert_allclose(a[0, 0:3], [0.1,0.2,0.3], atol=1e-5)  # translation passthrough
    assert np.isclose(a[0, 9], 0.7)              # gripper absolute passthrough
    np.testing.assert_allclose(a[0, 10:14], [1,2,3,4], atol=1e-5)    # base passthrough
    assert np.isclose(a[0, 14], 1.0)             # control_mode passthrough

def test_geometric_6d_roundtrip_nontrivial_rotation():
    # a 90deg rotation about z: rotvec=[0,0,pi/2]
    buf = _buf("geometric")
    act = np.zeros((12,), np.float32); act[3:6] = [0,0,np.pi/2]
    _step(buf, 0, act)
    sixd = buf.get_history_feats(0)["cosmo_action_emb"][0, 3:9]
    # first column of Rz(90) = [0,1,0]; second column = [-1,0,0]
    np.testing.assert_allclose(sixd, [0,1,0, -1,0,0], atol=1e-5)
