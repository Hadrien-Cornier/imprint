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


def test_policy_reset_prefers_archived_language_embedding(monkeypatch):
    from mme_vla_suite.policies import policy as P

    supplied = np.arange(8, dtype=np.float32)
    received = []
    buffer = types.SimpleNamespace(
        set_language=lambda value: received.append(value.copy())
    )
    policy = P.MME_VLA_Policy.__new__(P.MME_VLA_Policy)
    policy.mem_buffer = None
    policy.config = types.SimpleNamespace(
        representation_type="cosmo_bottleneck",
        cosmo_bottleneck_memory=types.SimpleNamespace(lang_emb_dim=8),
    )
    policy._seed = 42
    policy._uses_gated_delta = False
    monkeypatch.setattr(
        policy,
        "_prepare_mem_buffer",
        lambda: setattr(policy, "mem_buffer", buffer),
    )
    monkeypatch.setattr(
        policy,
        "_pool_prompt_text_emb",
        lambda prompt: (_ for _ in ()).throw(
            AssertionError("stored language embedding must be preferred")
        ),
    )

    policy.reset(prompt="ignored", lang_emb=supplied)

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], supplied)


def test_missing_checkpoint_history_keeps_requested_config(tmp_path):
    from mme_vla_suite.policies.policy_config import _effective_history_config

    checkpoint = tmp_path / "stage2" / "100000"
    checkpoint.mkdir(parents=True)

    assert (
        _effective_history_config(checkpoint, "portable.yaml")
        == "portable.yaml"
    )


def test_checkpoint_mapping_keys_follow_model_state_types():
    from openpi.models.model import _align_mapping_keys_to_expected

    expected = {
        "mem_encoder": {
            "blocks": {
                0: {"attention": {"kernel": object()}},
                1: {"attention": {"kernel": object()}},
            }
        }
    }
    restored = {
        "mem_encoder": {
            "blocks": {
                "0": {"attention": {"kernel": "first"}},
                "1": {"attention": {"kernel": "second"}},
            }
        }
    }

    aligned = _align_mapping_keys_to_expected(expected, restored)

    assert tuple(aligned["mem_encoder"]["blocks"]) == (0, 1)
    assert aligned["mem_encoder"]["blocks"][0]["attention"]["kernel"] == "first"


def test_requested_history_can_override_legacy_absolute_metadata(
    monkeypatch,
    tmp_path,
):
    from mme_vla_suite.policies.policy_config import _effective_history_config

    checkpoint = tmp_path / "stage2" / "100000"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "history_config.txt").write_text(
        "/retired/runtime/history.yaml\n"
    )
    monkeypatch.setenv("IMPRINT_PREFER_REQUESTED_HISTORY_CONFIG", "1")

    assert (
        _effective_history_config(checkpoint, "portable.yaml")
        == "portable.yaml"
    )
