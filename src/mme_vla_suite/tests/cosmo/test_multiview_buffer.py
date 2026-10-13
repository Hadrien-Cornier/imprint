import numpy as np
import pytest
from tests.cosmo import shapes as S


def test_buffer_multiview():
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    class _FakeCosmo:
        temporal_factor = 4
        def encode(self, frames):                 # (n*4,H,W,3)->(n,64,48)
            n = frames.shape[0] // 4
            return np.full((n, S.WAN_SPATIAL_TOKENS_128, S.WAN_LATENT_DIM), 1.0, np.float32)
        def encode_multiview(self, fv):            # (T,V,H,W,3)->(n, V*64, 48)
            V = fv.shape[1]; n = fv.shape[0] // 4
            return np.full((n, V*S.WAN_SPATIAL_TOKENS_128, S.WAN_LATENT_DIM), 1.0, np.float32)
    def _sig(x):                                   # (t,v,224,224,3)->(t,v,64,2048)
        return np.zeros((x.shape[0], x.shape[1], S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)
    buf = CosmoMemoryBuffer(cosmo=_FakeCosmo(), siglip_fn=_sig, chunk_frames=4, downsample=1,
                            num_views=3, state_dim=S.STATE_DIM, action_dim=12, latent_dim=S.WAN_LATENT_DIM)
    for step in range(9):   # 9 frames -> at least 1 finalized 4-frame chunk (drop-terminal)
        buf.add_buffer(images=np.zeros((1,3,128,128,3), np.uint8),
                       states=np.zeros((1,S.STATE_DIM), np.float32),
                       actions=np.zeros((1,12), np.float32), step_idx_list=[step])
    f = buf.get_history_feats(8)
    Lc = f["cosmo_latent_emb"].shape[0]
    assert Lc > 0 and Lc % (3*S.WAN_SPATIAL_TOKENS_128) == 0    # consolidated carries 3 views/chunk
    assert f["cosmo_latent_emb"].shape[1] == S.WAN_LATENT_DIM
    # recent SigLIP carries all 3 views per remaining frame
    Lr = f["cosmo_recent_emb"].shape[0]
    assert Lr % 3 == 0 and Lr > 0
    assert f["cosmo_recent_emb"].shape[1] == S.SIGLIP_DIM


def test_buffer_singleview_backcompat():
    # num_views=1 must still work via encode() fallback (existing behavior)
    from mme_vla_suite.shared.cosmo_buffer import CosmoMemoryBuffer
    class _FakeCosmo1:
        temporal_factor = 4
        def encode(self, frames):
            n = frames.shape[0] // 4
            return np.full((n, S.WAN_SPATIAL_TOKENS_128, S.WAN_LATENT_DIM), 1.0, np.float32)
    def _sig(x):
        return np.zeros((x.shape[0], x.shape[1], S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)
    buf = CosmoMemoryBuffer(cosmo=_FakeCosmo1(), siglip_fn=_sig, chunk_frames=4, downsample=1,
                            num_views=1, state_dim=S.STATE_DIM, action_dim=12, latent_dim=S.WAN_LATENT_DIM)
    for step in range(9):
        buf.add_buffer(images=np.zeros((1,1,128,128,3), np.uint8),
                       states=np.zeros((1,S.STATE_DIM), np.float32),
                       actions=np.zeros((1,12), np.float32), step_idx_list=[step])
    f = buf.get_history_feats(8)
    assert f["cosmo_latent_emb"].shape[0] % S.WAN_SPATIAL_TOKENS_128 == 0
    assert f["cosmo_recent_emb"].shape[1] == S.SIGLIP_DIM
