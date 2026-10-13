from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "rmbench_eval"
WORKER = PACKAGE_ROOT / "scripts" / "run_worker.sh"
LAUNCHER = PACKAGE_ROOT / "scripts" / "launch_eval.sh"
WRAPPER = ROOT / "scripts" / "run_molmoact2_rmbench_smoke_worker.sh"
BOOTSTRAP = ROOT / "scripts" / "bootstrap_molmoact2.sh"


def test_versioned_worker_owns_overlay_and_client_patcher_contract() -> None:
    source = WORKER.read_text(encoding="utf-8")

    assert 'EVAL_PACKAGE_ROOT=' in source
    assert '$EVAL_PACKAGE_ROOT/runtime/eval_policy_four_view.py' in source
    assert 'rmbench_eval.runtime.patch_policy_client' in source
    assert "rmbench_l40_harness" not in source
    assert 'serve_rmbench_molmoact2_tcp.py' in source
    assert 'RMBENCH_EXECUTION_HORIZON' in source
    assert 'online_wan_and_dinov3' in source
    assert 'stored Pi0.5 language embeddings' in source


def test_legacy_worker_delegates_to_versioned_worker() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'rmbench_eval/scripts/run_worker.sh' in source
    assert 'exec bash' in source


def test_launcher_executes_checked_in_worker_directly() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "RMBENCH_MOLMOACT2_INPUT_REPO" in source
    assert "RMBENCH_MOLMOACT2_EVAL_ASSETS_REPO" in source
    assert "--task" in source
    assert "--trials" in source
    assert "--template" not in source
    assert "exec bash" in source
    assert 'scripts/run_worker.sh' in source


def test_evaluation_readme_documents_runtime_transport_and_parity() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "RMBench Stage-2 Evaluation",
        "RMBENCH_RUNTIME_ARCHIVE_DIR",
        "Amazon-FAR/imprint_models",
        "simulation/rmbench/molmoact2",
        "serve_rmbench_molmoact2_tcp.py",
        "patch_policy_client.py",
        "online DINOv3",
        "stored Pi0.5 language embeddings",
        "head_camera",
        "left_camera",
        "right_camera",
        "RGB-to-BGR",
        "EVAL_CONSOLIDATE_ALL=1",
        "trials.jsonl",
        "WANDB_API_KEY",
    ):
        assert required in readme


def test_portable_bootstrap_clones_from_github_and_checks_artifacts() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert "https://github.com/amazon-far/imprint.git" in source
    assert 'BRANCH="${MOLMOACT2_BRANCH:-molmoact2}"' in source
    assert "git ls-remote" in source
    assert "git clone --branch" in source
    assert "submodule update --init" in source
    assert "--recursive" not in source
    assert "arn:aws" not in source
    assert "gh auth" not in source
    assert "stage1_step160000.npz" in source
    assert "164c94f3ff8fcd84aee61e88ec0feb97faf99d203eac1be0f4b4146d464d2044" in source


def test_portable_bootstrap_doc_uses_huggingface_stage1_checkpoint() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Amazon-FAR/imprint_models" in readme
    assert "simulation/rmbench/full_attention" in readme
    assert "RMBENCH_RUNTIME_ARCHIVE_DIR" in readme
