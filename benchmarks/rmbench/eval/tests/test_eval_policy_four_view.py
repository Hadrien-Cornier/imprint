from types import SimpleNamespace

import pytest

from rmbench_l40_eval.runtime import eval_policy_four_view
from rmbench_l40_eval.runtime.eval_policy_four_view import (
    configure_canonical_client_timeout,
    configure_canonical_renderer,
    install_four_view_evaluation_loop,
)


def test_configure_canonical_renderer_uses_available_hook() -> None:
    calls = []
    canonical = SimpleNamespace(
        configure_renderer_device=lambda: calls.append("configured")
    )

    configure_canonical_renderer(canonical)

    assert calls == ["configured"]


def test_configure_canonical_renderer_supports_legacy_evaluator() -> None:
    configure_canonical_renderer(SimpleNamespace())


def test_configure_canonical_client_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class ModelClient:
        def __init__(self, *, port, timeout=30):
            captured.update(port=port, timeout=timeout)

    canonical = SimpleNamespace(ModelClient=ModelClient)
    monkeypatch.setenv("RMBENCH_MODEL_CLIENT_TIMEOUT", "600")

    configure_canonical_client_timeout(canonical)
    canonical.ModelClient(port=8234)

    assert captured == {"port": 8234, "timeout": 600}


@pytest.mark.parametrize("timeout", ["0", "-1", "invalid"])
def test_configure_canonical_client_timeout_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    timeout: str,
) -> None:
    monkeypatch.setenv("RMBENCH_MODEL_CLIENT_TIMEOUT", timeout)
    with pytest.raises(ValueError, match="RMBENCH_MODEL_CLIENT_TIMEOUT"):
        configure_canonical_client_timeout(
            SimpleNamespace(ModelClient=object),
        )


def test_evaluation_loop_uses_requested_trial_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_eval(*args, **kwargs):
        captured["trial_count"] = args[8]
        return 100001, 1

    monkeypatch.setattr(
        eval_policy_four_view,
        "_eval_policy_with_four_view",
        fake_eval,
    )
    loop = install_four_view_evaluation_loop(
        SimpleNamespace(),
        tmp_path,
        trial_count=1,
    )

    loop("cover_blocks", object(), {}, object(), 100000, test_num=100)

    assert captured["trial_count"] == 1


@pytest.mark.parametrize("trial_count", [0, -1, True, 1.5])
def test_evaluation_loop_rejects_invalid_trial_count(
    tmp_path,
    trial_count,
) -> None:
    with pytest.raises(ValueError, match="trial_count"):
        install_four_view_evaluation_loop(
            SimpleNamespace(),
            tmp_path,
            trial_count=trial_count,
        )
