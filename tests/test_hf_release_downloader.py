from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = ROOT / "scripts" / "reproduce" / "download_release.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def run_downloader(
    tmp_path: Path,
    benchmark: str,
    architecture: str,
    selection: str,
) -> tuple[list[str], list[str]]:
    hf_log = tmp_path / "hf.log"
    python_log = tmp_path / "python.log"
    fake_hf = tmp_path / "hf"
    fake_python = tmp_path / "python"
    write_executable(
        fake_hf,
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$HF_LOG"\n'
        'local_dir=""\n'
        'include=""\n'
        'while [[ "$#" -gt 0 ]]; do\n'
        '  case "$1" in\n'
        '    --local-dir) local_dir="$2"; shift 2 ;;\n'
        '    --include) include="$2"; shift 2 ;;\n'
        '    *) shift ;;\n'
        '  esac\n'
        'done\n'
        'if [[ -n "$local_dir" && -n "$include" ]]; then\n'
        '  mkdir -p "$local_dir/${include%/**}"\n'
        'fi\n',
    )
    write_executable(
        fake_python,
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$PYTHON_LOG"\n',
    )
    env = os.environ.copy()
    env.update(
        {
            "HF_LOG": str(hf_log),
            "PYTHON_LOG": str(python_log),
            "IMPRINT_HF_CLI": str(fake_hf),
            "IMPRINT_PYTHON": str(fake_python),
            "IMPRINT_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "IMPRINT_DOWNLOAD_WORKERS": "7",
        }
    )
    subprocess.run(
        ["bash", str(DOWNLOADER), benchmark, architecture, selection],
        cwd=ROOT,
        env=env,
        check=True,
    )
    hf_calls = hf_log.read_text().splitlines() if hf_log.exists() else []
    python_calls = python_log.read_text().splitlines() if python_log.exists() else []
    return hf_calls, python_calls


@pytest.mark.parametrize(
    ("benchmark", "architecture", "prefix"),
    [
        ("rmbench", "full_attention", "simulation/rmbench/full_attention"),
        ("rmbench", "recurrent", "simulation/rmbench/recurrent"),
        ("rmbench", "pi0", "simulation/rmbench/pi0"),
        ("robocasa", "full_attention", "simulation/robocasa/full_attention"),
        ("robocasa", "recurrent", "simulation/robocasa/recurrent"),
    ],
)
def test_model_downloads_use_amazon_far(
    tmp_path: Path,
    benchmark: str,
    architecture: str,
    prefix: str,
) -> None:
    hf_calls, python_calls = run_downloader(
        tmp_path, benchmark, architecture, "--model-only"
    )
    assert len(hf_calls) == 1
    assert hf_calls[0].startswith(
        "download Amazon-FAR/imprint_models --revision main "
    )
    assert f"--include {prefix}/**" in hf_calls[0]
    assert "--max-workers 7" in hf_calls[0]
    assert python_calls == []


def test_dataset_download_and_extraction_use_hugging_face(tmp_path: Path) -> None:
    hf_calls, python_calls = run_downloader(
        tmp_path, "rmbench", "full_attention", "--dataset-only"
    )
    assert len(hf_calls) == 1
    assert hf_calls[0].startswith("download Amazon-FAR/imprint_datasets ")
    assert "--repo-type dataset" in hf_calls[0]
    assert "--revision main" in hf_calls[0]
    assert "--include simulation/rmbench/450/**" in hf_calls[0]
    assert len(python_calls) == 1
    assert "extract_dataset_store.py" in python_calls[0]
    assert "imprint_datasets/simulation/rmbench/450" in python_calls[0]
    assert "--skip-existing" in python_calls[0]
