from types import SimpleNamespace

from mme_vla_suite.shared.memory_caps import max_recent_tokens


def test_max_recent_tokens_uses_explicit_config_value():
    assert max_recent_tokens(SimpleNamespace(max_recent_tokens=768)) == 768


def test_max_recent_tokens_keeps_legacy_default_when_unset():
    assert max_recent_tokens(SimpleNamespace()) == 1024
