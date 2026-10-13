from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
WRAPPER = REPOSITORY_ROOT / "scripts" / "reproduce" / "train_molmoact2_stage2.sh"
LAUNCHER = (
    REPOSITORY_ROOT
    / "backbones"
    / "molmoact2"
    / "experiments"
    / "launch_scripts"
    / "launch_rmbench_molmoact2_stage2_memory_ablation.bash"
)
PROFILE = (
    REPOSITORY_ROOT
    / "configs"
    / "paper"
    / "rmbench"
    / "molmoact2"
    / "experiment.env"
)


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_paper_profile_records_exact_molmoact2_shape() -> None:
    profile = _read_env(PROFILE)

    assert profile["IMPRINT_BACKBONE"] == "molmoact2"
    assert profile["IMPRINT_STAGE1_SELECTED_STEP"] == "160000"
    assert profile["MOLMOACT2_STAGE2_STEPS"] == "100000"
    assert profile["MOLMOACT2_ACTION_HORIZON"] == "50"
    assert profile["MOLMOACT2_ACTION_DIM"] == "16"
    assert profile["MOLMOACT2_NUM_LATENTS"] == "256"
    assert profile["MOLMOACT2_MEMORY_DIM"] == "1024"


def test_wrapper_forwards_profile_shape_to_native_launcher() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    for variable in (
        "MOLMOACT2_ACTION_HORIZON",
        "MOLMOACT2_ACTION_DIM",
        "MOLMOACT2_NUM_LATENTS",
        "MOLMOACT2_MEMORY_DIM",
    ):
        assert variable in source


def test_native_launcher_passes_explicit_shape_flags() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    for flag in (
        '--max_action_horizon="${MAX_ACTION_HORIZON}"',
        '--max_action_dim="${MAX_ACTION_DIM}"',
        '--memory_num_latents="${MEMORY_NUM_LATENTS}"',
        '--memory_dim="${MEMORY_DIM}"',
    ):
        assert flag in source
