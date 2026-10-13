import numpy as np
import types
from tests.cosmo import shapes as S

class _FakeCosmo:
    temporal_factor = 8
    def encode(self, frames):
        n = frames.shape[0] // 8
        return np.full((n, S.COSMOS_SPATIAL_TOKENS, S.COSMOS_LATENT_DIM), 1.0, np.float32)

def _fake_siglip(x):
    t, v = x.shape[0], x.shape[1]
    return np.zeros((t, v, S.SIGLIP_TOKENS_PER_FRAME, S.SIGLIP_DIM), np.float32)

def _cfg():
    return types.SimpleNamespace(
        representation_type="cosmo_bottleneck",
        num_views=1,
        memory_feature=types.SimpleNamespace(state=types.SimpleNamespace(input_dim=16)),
        cosmo_bottleneck_memory=types.SimpleNamespace(
            variant="CV8x8x8", temporal_factor=8, downsample=1,
            action_in_dim=24, cosmos_latent_dim=16),
    )

def test_build_cosmo_buffer(monkeypatch):
    from mme_vla_suite.policies import policy as P
    monkeypatch.setattr(P, "_load_cosmo_frontend", lambda variant: _FakeCosmo())
    buf = P._build_cosmo_buffer(_cfg(), siglip_fn=_fake_siglip)
    assert buf.__class__.__name__ == "CosmoMemoryBuffer"
    assert buf.chunk_frames == 8
    assert buf.action_dim == 12
    assert buf.latent_dim == 16
