import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "serve_rmbench_molmoact2.py"
)
LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "launch_rmbench_molmoact2_eval.sh"
)
TCP_SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "serve_rmbench_molmoact2_tcp.py"
)
DATASET_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "lerobot"
    / "src"
    / "lerobot"
    / "datasets"
    / "utils.py"
)
LEROBOT_SOURCE_ROOT = DATASET_UTILS_PATH.parents[2]
RUNTIME_UTILS_PATH = DATASET_UTILS_PATH.parents[1] / "utils" / "utils.py"
MOLMOACT2_PACKAGE_PATH = (
    LEROBOT_SOURCE_ROOT / "lerobot" / "policies" / "molmoact2" / "__init__.py"
)
POLICIES_PACKAGE_PATH = LEROBOT_SOURCE_ROOT / "lerobot" / "policies" / "__init__.py"
PRETRAINED_POLICY_PATH = LEROBOT_SOURCE_ROOT / "lerobot" / "policies" / "pretrained.py"
POLICY_UTILS_PATH = LEROBOT_SOURCE_ROOT / "lerobot" / "policies" / "utils.py"
PROCESSOR_PACKAGE_PATH = LEROBOT_SOURCE_ROOT / "lerobot" / "processor" / "__init__.py"
OLMO_IO_PATH = Path(__file__).resolve().parents[1] / "olmo" / "io.py"
OLMO_MODEL_PATH = Path(__file__).resolve().parents[1] / "olmo" / "models" / "model.py"
OLMO_CHECKPOINTER_PATH = Path(__file__).resolve().parents[1] / "olmo" / "train" / "checkpointer.py"
OLMO_UTIL_PATH = Path(__file__).resolve().parents[1] / "olmo" / "util.py"
VIDEO_LOADER_PATH = Path(__file__).resolve().parents[1] / "olmo" / "data" / "video_loader.py"


def load_module():
    spec = importlib.util.spec_from_file_location("serve_rmbench_molmoact2", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_config(module, tmp_path, **overrides):
    paths = {
        "checkpoint": tmp_path / "checkpoint",
        "stage2_root": tmp_path / "stage2",
        "feature_root": tmp_path / "features",
        "dino_model_path": tmp_path / "dino",
    }
    for path in paths.values():
        path.mkdir()
    paths.update(overrides)
    return module.ServerConfig(**paths)


def write_memory_layout(stage2_root, **overrides):
    layout = {
        "temporal_factor": 4,
        "downsample": 2,
        "latent_cap": 4096,
        "max_recent_tokens": 768,
        "max_step_tokens": 1244,
        "latent_dim": 48,
        "recent_dim": 768,
        "stream_dim": 16,
        "num_views": 3,
        "pool_h": 2,
        "pool_w": 2,
        "wan_h": 256,
        "wan_w": 256,
    }
    layout.update(overrides)
    meta_dir = stage2_root / "meta"
    meta_dir.mkdir()
    path = meta_dir / "memory_layout.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    return layout


def test_load_stage2_memory_layout_requires_canonical_metadata(tmp_path):
    module = load_module()
    stage2_root = tmp_path / "stage2"
    stage2_root.mkdir()

    with pytest.raises(FileNotFoundError, match="stage2/meta/memory_layout.json"):
        module.load_stage2_memory_layout(stage2_root)


def test_load_stage2_memory_layout_reads_exact_canonical_metadata(tmp_path):
    module = load_module()
    stage2_root = tmp_path / "stage2"
    stage2_root.mkdir()
    expected = write_memory_layout(stage2_root, latent_cap=2048, max_step_tokens=512)

    layout = module.load_stage2_memory_layout(stage2_root)

    assert {
        field: getattr(layout, field)
        for field in expected
    } == expected


def test_server_config_rejects_offline_visual_features(tmp_path):
    module = load_module()
    config = make_config(module, tmp_path, offline_feature_root="/tmp/legacy-features")

    with pytest.raises(ValueError, match="offline visual features are forbidden"):
        config.validate()


def test_runtime_provenance_records_online_visual_contract(tmp_path):
    module = load_module()
    stage1_source = tmp_path / "pi05"
    stage1_source.mkdir()
    (stage1_source / "pyproject.toml").write_text(
        '[project]\nname = "mme-vla-suite"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    wan_module_file = (
        stage1_source
        / "mme_vla_suite"
        / "models"
        / "representation"
        / "wan_frontend.py"
    )
    wan_module_file.parent.mkdir(parents=True)
    wan_module_file.write_text("", encoding="utf-8")
    config = make_config(module, tmp_path, stage1_source_root=str(stage1_source))

    provenance = config.runtime_provenance()

    assert provenance == {
        "checkpoint": str(tmp_path / "checkpoint"),
        "stage2_metadata_root": str(tmp_path / "stage2"),
        "language_feature_root": str(tmp_path / "features"),
        "dino_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dino_model_revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
        "dino_model_path": str(tmp_path / "dino"),
        "camera_order": ["head_camera", "left_camera", "right_camera"],
        "prediction_horizon": 50,
        "flow_seed": None,
        "memory_enabled": True,
        "visual_history": "online_wan_and_dinov3",
        "offline_visual_features": False,
        "language_source": "stored_pi05_embeddings",
        "stage1_source_root": str(stage1_source),
        "wan_frontend_variant": "Wan2.2-TI2V-5B",
        "wan_stage1_source_root": str(stage1_source),
        "wan_resolved_source_path": str(stage1_source.resolve()),
        "wan_source_version": "1.2.3",
        "wan_frontend_module_file": str(wan_module_file.resolve()),
    }


def test_no_memory_config_only_requires_checkpoint_and_records_native_prompt(tmp_path):
    module = load_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = module.ServerConfig(
        checkpoint=str(checkpoint),
        memory_enabled=False,
    )

    config.validate()

    assert config.runtime_provenance() == {
        "checkpoint": str(checkpoint),
        "camera_order": ["head_camera", "left_camera", "right_camera"],
        "prediction_horizon": 50,
        "flow_seed": None,
        "memory_enabled": False,
        "visual_history": "none",
        "offline_visual_features": False,
        "language_source": "molmoact2_prompt",
    }


def test_get_model_no_memory_does_not_require_memory_paths(tmp_path, monkeypatch):
    module = load_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured = []
    monkeypatch.setattr(
        module,
        "build_server_model",
        lambda config: captured.append(config) or "model",
    )

    result = module.get_model(
        {
            "checkpoint": str(checkpoint),
            "memory_enabled": False,
            "prediction_horizon": 50,
        }
    )

    assert result == "model"
    assert captured[0].memory_enabled is False
    assert captured[0].stage2_root is None
    assert captured[0].feature_root is None
    assert captured[0].dino_model_path is None
    assert captured[0].stage1_source_root is None


def test_get_model_rejects_bad_config_before_model_import(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "build_server_model",
        lambda config: pytest.fail("build_server_model must not run for invalid config"),
    )
    directories = {
        name: tmp_path / name
        for name in ("checkpoint", "stage2", "features", "dino", "pi05")
    }
    for directory in directories.values():
        directory.mkdir()

    with pytest.raises(ValueError, match="prediction_horizon"):
        module.get_model(
            {
                "checkpoint": str(directories["checkpoint"]),
                "stage2_root": str(directories["stage2"]),
                "feature_root": str(directories["features"]),
                "dino_model_path": str(directories["dino"]),
                "stage1_source_root": str(directories["pi05"]),
                "prediction_horizon": 49,
            }
        )


def test_smoke_launcher_requires_online_inputs_and_fixed_horizons():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    for variable in (
        "MOLMOACT2_CHECKPOINT",
        "RMBENCH_STAGE2_ROOT",
        "RMBENCH_STAGE1_FEATURE_ROOT",
        "RMBENCH_DINOV3_MODEL_PATH",
        "RMBENCH_PI05_SOURCE_ROOT",
        "RMBENCH_TASK",
        "RMBENCH_POLICY_MODEL_SERVER",
    ):
        assert variable in source

    assert "MEMFT_DINOV3_REQUIRED=1" in source
    assert "EVAL_CONSOLIDATE_ALL=1" in source
    assert "RMBENCH_CAMERA_ORDER=head,left,right" in source
    assert "RMBENCH_PREDICTION_HORIZON=50" in source
    assert "RMBENCH_EXECUTION_HORIZON=50" in source
    assert 'meta/memory_layout.json' in source
    assert 'meta/episodes.jsonl' in source
    assert "offline_feature_root" not in source
    assert "scripts.serve_rmbench_molmoact2" in source
    assert "--prediction-horizon" in source
    assert "sha256sum" in source


def test_smoke_launcher_treats_task_as_a_literal_selector():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert 'require_file RMBENCH_TASK "$RMBENCH_TASK"' not in source
    assert 'task_sha256="$(printf \'%s\' "$RMBENCH_TASK" | sha256sum' in source


def test_dedicated_tcp_server_avoids_simulator_environment_imports():
    assert TCP_SERVER_PATH.is_file()

    source = TCP_SERVER_PATH.read_text(encoding="utf-8")

    assert "from serve_rmbench_molmoact2 import get_model" in source
    assert "import sapien" not in source.lower()
    assert "from sapien" not in source.lower()
    assert "recv_exact" in source
    assert "numpy_to_json" in source


def test_policy_import_dependency_parses_with_pinned_python_311_runtime():
    source = DATASET_UTILS_PATH.read_text(encoding="utf-8")

    ast.parse(source, filename=str(DATASET_UTILS_PATH), feature_version=(3, 11))


def test_lerobot_source_parses_with_pinned_python_311_runtime():
    for source_path in LEROBOT_SOURCE_ROOT.rglob("*.py"):
        ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
            feature_version=(3, 11),
        )


def test_policy_runtime_does_not_import_accelerate_for_type_annotations():
    source = RUNTIME_UTILS_PATH.read_text(encoding="utf-8")

    assert "from typing import TYPE_CHECKING" in source
    assert "if TYPE_CHECKING:\n    from accelerate import Accelerator" in source


def test_molmoact2_processor_factory_is_lazy_for_policy_serving():
    source = MOLMOACT2_PACKAGE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(MOLMOACT2_PACKAGE_PATH))

    assert all(
        not isinstance(statement, ast.ImportFrom)
        or statement.module != "processor_molmoact2"
        for statement in module.body
    )
    assert 'if name == "make_molmoact2_pre_post_processors":' in source


def test_policy_registry_does_not_eagerly_import_unrelated_policy_backends():
    source = POLICIES_PACKAGE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(POLICIES_PACKAGE_PATH))

    assert all(
        not isinstance(statement, ast.ImportFrom) or statement.level == 0
        for statement in module.body
    )
    assert "_LAZY_EXPORTS" in source


def test_pretrained_policy_does_not_import_training_config_for_serving():
    source = PRETRAINED_POLICY_PATH.read_text(encoding="utf-8")

    assert "from typing import TYPE_CHECKING" in source
    assert "if TYPE_CHECKING:\n    from lerobot.configs.train import TrainPipelineConfig" in source


def test_policy_utils_imports_core_types_without_initializing_hardware_processors():
    source = POLICY_UTILS_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(POLICY_UTILS_PATH))

    assert any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "lerobot.processor.core"
        and {alias.name for alias in statement.names}
        == {"PolicyAction", "RobotAction", "RobotObservation"}
        for statement in module.body
    )
    assert all(
        not (
            isinstance(statement, ast.ImportFrom)
            and statement.module == "lerobot.processor"
        )
        for statement in module.body
    )


def test_processor_package_defers_hardware_processor_imports_for_serving():
    source = PROCESSOR_PACKAGE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(PROCESSOR_PACKAGE_PATH))

    eager_modules = {
        statement.module
        for statement in module.body
        if isinstance(statement, ast.ImportFrom) and statement.level == 1
    }

    assert "gym_action_processor" not in eager_modules
    assert "hil_processor" not in eager_modules
    assert "_HARDWARE_LAZY_EXPORTS" in source
    assert "def __getattr__(name: str):" in source


def test_olmo_io_supports_local_paths_without_cached_path_dependency():
    source = OLMO_IO_PATH.read_text(encoding="utf-8")

    assert "except ModuleNotFoundError:" in source
    assert "def cached_path(path, quiet=True, **kwargs):" in source
    assert 'if "://" in str(path):' in source
    assert "cached_path is required for remote paths" in source


def test_olmo_model_does_not_import_training_metrics_for_serving():
    source = OLMO_MODEL_PATH.read_text(encoding="utf-8")

    assert "from typing import TYPE_CHECKING" in source
    assert "if TYPE_CHECKING:\n    import torchmetrics" in source


def test_olmo_checkpointer_reuses_local_cached_path_fallback():
    source = OLMO_CHECKPOINTER_PATH.read_text(encoding="utf-8")

    assert "from cached_path import cached_path" not in source
    assert "cached_path," in source


def test_olmo_util_resource_path_reuses_local_cached_path_fallback():
    source = OLMO_UTIL_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(OLMO_UTIL_PATH))

    assert "from cached_path import cached_path" not in source
    assert any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "io"
        and statement.level == 1
        and "cached_path" in {alias.name for alias in statement.names}
        for statement in module.body
    )


def test_video_loader_defers_optional_video_decoders_for_serving():
    source = VIDEO_LOADER_PATH.read_text(encoding="utf-8")

    assert "try:\n    import decord" in source
    assert "except ModuleNotFoundError:\n    decord = None" in source
    assert "try:\n    import av" in source
    assert "if decord is not None:\n    decord.logging.set_level(2)" in source
