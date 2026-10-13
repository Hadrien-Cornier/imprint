"""Patch the canonical RMBench policy client for managed evaluation runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


TEST_NUM_MARKER = "test_num = 100"
RPC_TIMEOUT_MARKER = "    def __init__(self, host='localhost', port=9999, timeout=30):"
OUTPUT_MARKER = "    save_dir.mkdir(parents=True, exist_ok=True)"
TASK_CONFIG_MARKER = (
    'camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")'
)
RPC_RESPONSE_MARKER = "        return response['res']"

REPLACEMENTS = (
    (
        "test count",
        TEST_NUM_MARKER,
        'test_num = int(usr_args.get("test_num", 100))',
    ),
    (
        "RPC timeout",
        RPC_TIMEOUT_MARKER,
        "    def __init__(self, host='localhost', port=9999, "
        "timeout=int(os.environ.get('RMBENCH_POLICY_RPC_TIMEOUT', '180'))):",
    ),
    (
        "output directory",
        OUTPUT_MARKER,
        """    if usr_args.get("result_dir"):
        save_dir = Path(usr_args["result_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    args["log_file"] = str(save_dir / "eval_log.txt")
    args["trial_results_path"] = str(save_dir / "trials.jsonl")""",
    ),
    (
        "task config",
        TASK_CONFIG_MARKER,
        'camera_config_path = os.environ.get("RMBENCH_TASK_CONFIG_PATH", '
        'os.path.join(parent_directory, "../task_config/_camera_config.yml"))',
    ),
    (
        "RPC response",
        RPC_RESPONSE_MARKER,
        """        if "res" not in response:
            raise RuntimeError(
                f"Policy server error: {response.get('error', 'unknown')}\\n"
                f"{response.get('traceback')}"
            )
        return response["res"]""",
    ),
)

RENDERER_CONFIGURATION = """

def configure_renderer_device():
    import os
    renderer_device = os.environ.get("RMBENCH_RENDER_DEVICE")
    if renderer_device:
        import sapien.core as sapien_core
        sapien_core.SapienRenderer = lambda: sapien_core.render.SapienRenderer(
            sapien_core.Device(renderer_device)
        )
"""


def _validate_markers(source: str) -> None:
    for name, marker, _replacement in REPLACEMENTS:
        marker_count = source.count(marker)
        if marker_count != 1:
            raise ValueError(
                f"unrecognized canonical RMBench policy client: expected exactly "
                f"one {name} marker, found {marker_count}"
            )


def patch_policy_client(source: Path, target: Path) -> None:
    """Write a patched policy client without modifying the canonical source."""
    source = Path(source)
    target = Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("source and target must differ; canonical client is read-only")

    canonical_source = source.read_text(encoding="utf-8")
    _validate_markers(canonical_source)

    patched_source = canonical_source
    for _name, marker, replacement in REPLACEMENTS:
        patched_source = patched_source.replace(marker, replacement, 1)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(patched_source + RENDERER_CONFIGURATION)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch a canonical RMBench policy client for managed evaluation."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    patch_policy_client(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
