from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from rmbench_eval.runtime.patch_policy_client import patch_policy_client


CANONICAL_CLIENT = """\
import os
from pathlib import Path

test_num = 100

class ModelClient:
    def __init__(self, host='localhost', port=9999, timeout=30):
        pass

    def predict(self, response):
        return response['res']

def run(usr_args, args, save_dir, parent_directory):
    save_dir.mkdir(parents=True, exist_ok=True)
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
"""


def test_patch_policy_client_applies_the_worker_contract(tmp_path):
    source = tmp_path / "eval_policy_client.py"
    output = tmp_path / "patched_eval_policy_client.py"
    source.write_text(CANONICAL_CLIENT, encoding="utf-8")

    patch_policy_client(source, output)

    patched = output.read_text(encoding="utf-8")
    assert 'test_num = int(usr_args.get("test_num", 100))' in patched
    assert (
        "timeout=int(os.environ.get('RMBENCH_POLICY_RPC_TIMEOUT', '180'))"
        in patched
    )
    assert 'save_dir = Path(usr_args["result_dir"])' in patched
    assert 'args["log_file"] = str(save_dir / "eval_log.txt")' in patched
    assert 'args["trial_results_path"] = str(save_dir / "trials.jsonl")' in patched
    assert 'os.environ.get("RMBENCH_TASK_CONFIG_PATH",' in patched
    assert 'if "res" not in response:' in patched
    assert 'Policy server error: {response.get(\'error\', \'unknown\')}\\n' in patched
    assert 'f"{response.get(\'traceback\')}"' in patched
    assert "def configure_renderer_device():" in patched
    assert 'os.environ.get("RMBENCH_RENDER_DEVICE")' in patched
    assert "sapien_core.Device(renderer_device)" in patched


def test_patch_policy_client_names_missing_canonical_marker(tmp_path):
    source = tmp_path / "eval_policy_client.py"
    output = tmp_path / "patched_eval_policy_client.py"
    source.write_text(
        CANONICAL_CLIENT.replace("        return response['res']", ""),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="RPC response"):
        patch_policy_client(source, output)

    assert not output.exists()


def test_patch_policy_client_rejects_duplicate_marker_without_overwriting_target(tmp_path):
    source = tmp_path / "eval_policy_client.py"
    output = tmp_path / "patched_eval_policy_client.py"
    source.write_text(
        CANONICAL_CLIENT.replace(
            "test_num = 100",
            "test_num = 100\ntest_num = 100",
        ),
        encoding="utf-8",
    )
    output.write_text("existing target contents", encoding="utf-8")

    with pytest.raises(ValueError, match="test count marker"):
        patch_policy_client(source, output)

    assert output.read_text(encoding="utf-8") == "existing target contents"


def test_patch_policy_client_rejects_in_place_output_without_mutating_source(tmp_path):
    source = tmp_path / "eval_policy_client.py"
    source.write_text(CANONICAL_CLIENT, encoding="utf-8")

    with pytest.raises(ValueError, match="source and target must differ"):
        patch_policy_client(source, source)

    assert source.read_text(encoding="utf-8") == CANONICAL_CLIENT


def test_patch_policy_client_keeps_target_on_atomic_replace_failure(tmp_path, monkeypatch):
    source = tmp_path / "eval_policy_client.py"
    output = tmp_path / "patched_eval_policy_client.py"
    source.write_text(CANONICAL_CLIENT, encoding="utf-8")
    output.write_text("existing target contents", encoding="utf-8")

    def raise_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "rmbench_eval.runtime.patch_policy_client.os.replace",
        raise_replace,
    )

    with pytest.raises(OSError, match="simulated replace failure"):
        patch_policy_client(source, output)

    assert output.read_text(encoding="utf-8") == "existing target contents"
    assert not list(tmp_path.glob(".patched_eval_policy_client.py.*.tmp"))


def test_patch_policy_client_cli_writes_output(tmp_path):
    source = tmp_path / "eval_policy_client.py"
    output = tmp_path / "patched_eval_policy_client.py"
    source.write_text(CANONICAL_CLIENT, encoding="utf-8")
    patcher = (
        Path(__file__).resolve().parents[1]
        / "rmbench_eval"
        / "runtime"
        / "patch_policy_client.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(patcher),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
