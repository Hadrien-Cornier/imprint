from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "rmbench_eval" / "scripts" / "run_worker.sh"
WRAPPER = ROOT / "scripts" / "run_molmoact2_rmbench_smoke_worker.sh"
PREPARE_BUNDLE = ROOT / "scripts" / "prepare_rmbench_molmoact2_smoke_bundle.sh"
REBUILD_CUROBO = ROOT / "rmbench_eval" / "scripts" / "rebuild_curobo_b200_in_place.sh"


def test_molmoact2_smoke_worker_restores_online_only_bundle_on_two_gpus() -> None:
    assert WORKER.is_file()
    assert WRAPPER.is_file()

    script = WORKER.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    for argument in (
        "--input-repo",
        "--input-revision",
        "--input-prefix",
        "--eval-assets-prefix",
        "--task",
        "--trials",
    ):
        assert argument in script
    assert "run_python() {" in script
    assert 'if [[ -x "$POLICY_PY" ]]; then' in script
    assert '"$POLICY_PY" "$@"' in script
    assert "ensure_hf_cli() {" in script
    assert "stage_hf() {" in script
    assert "RMBench Hugging Face stage:" in script
    assert 'run_python - "$MANIFEST_PATH" <<\'PY\'' in script
    assert "cover_blocks" in script
    for task in (
        "battery_try",
        "blocks_ranking_try",
        "cover_blocks",
        "observe_and_pickup",
        "press_button",
        "put_back_block",
        "rearrange_blocks",
        "swap_T",
        "swap_blocks",
    ):
        assert f'"{task}"' in script
    assert "MolmoAct2 RMBench worker only supports official all9 tasks" in script
    assert 'trial_count == 1' not in script
    assert 'START_SEED="${RMBENCH_START_SEED:-100000}"' in script
    assert '"start_seed": ${START_SEED}' in script
    assert '--start_seed "$START_SEED"' in script

    for artifact in (
        "manifest.json",
        "model-release",
        "eval-assets",
        '${EVAL_ASSETS_ROOT}/stage2',
        '${EVAL_ASSETS_ROOT}/stage1-language',
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ):
        assert artifact in script
    assert "molmoact2-src.tar.gz" not in script
    assert "pi05-src.tar.gz" not in script
    assert "meta/episodes.jsonl" in script
    assert "meta/memory_layout.json" in script
    assert "offline_visual_features" in script
    for offline_array in (
        "dinov3.npy",
        "latents_v*.npy",
        "memory_cosmo_latents*.npy",
        "memory_recent_embeddings*.npy",
    ):
        assert offline_array in script
    assert 'rm -rf "$INPUT_ROOT"' not in script
    assert '"$MODEL_DOWNLOAD_ROOT"' in script
    assert '"$EVAL_ASSETS_DOWNLOAD_ROOT"' in script
    assert '--include "${MODEL_PREFIX}/**"' in script
    assert '--include "${EVAL_ASSETS_PREFIX}/**"' in script

    assert 'CUDA_VISIBLE_DEVICES=${POLICY_GPU}' in script
    assert 'CUDA_VISIBLE_DEVICES=${EVALUATOR_GPU}' in script
    assert 'WORKER_GPU="${RMBENCH_WORKER_GPU:-0}"' in script
    assert 'POLICY_GPU="$WORKER_GPU"' in script
    assert 'EVALUATOR_GPU="$WORKER_GPU"' in script
    assert "RMBENCH_WORKER_GPU must be a non-negative integer" in script
    assert "RMBENCH_POLICY_RPC_TIMEOUT" in script
    assert '"RMBENCH_POLICY_RPC_TIMEOUT=${RMBENCH_POLICY_RPC_TIMEOUT:-180}"' in script
    assert "WANDB_BASE_URL=" not in script
    assert "serve_rmbench_molmoact2" in script
    assert "serve_rmbench_molmoact2_tcp.py" in script
    assert "eval_policy_client.py" in script
    assert "eval_policy_four_view.py" in script
    assert "$EVAL_PACKAGE_ROOT/four_view.py" in script
    assert "rmbench_l40_harness" not in script
    assert "rmbench_eval.runtime.patch_policy_client" in script
    assert "cached_path" not in script
    assert "POLICY_DEPS" not in script
    assert 'DINO_TRANSFORMERS_DEPS="${WORK_ROOT}/dino_transformers_deps"' in script
    assert '"transformers==5.15.0" "tokenizers==0.22.2"' in script
    assert '"$UV_BIN" --no-config pip install --index-url https://pypi.org/simple' in script
    assert "AutoImageProcessor.from_pretrained" in script
    assert "AutoModel.from_pretrained" in script
    assert "--no-deps" not in script
    assert 'server_pythonpath="${DINO_TRANSFORMERS_DEPS}:${server_pythonpath}:${PI05_ROOT}"' in script
    assert "RMBENCH_RENDER_DEVICE=cuda:0" in script
    assert '"test_num": ${trial_count}' in script
    assert 'EXECUTION_HORIZON="${RMBENCH_EXECUTION_HORIZON:-50}"' in script
    assert 'FLOW_SEED="${RMBENCH_FLOW_SEED:-0}"' in script
    assert '"execution_horizon": ${EXECUTION_HORIZON}' in script
    assert '"RMBENCH_EXECUTION_HORIZON=${EXECUTION_HORIZON}"' in script
    assert '--execution_horizon "$EXECUTION_HORIZON"' in script
    assert '"flow_seed": int(flow_seed)' in script
    assert "--prepare-only" in script
    assert '"effective_execution_horizon": int(execution_horizon)' in script
    assert '"effective_flow_seed": int(flow_seed)' in script
    assert "dump_failure_logs" in script
    assert 'for log_path in "${OUTPUT_ROOT}/evaluator.log" "${OUTPUT_ROOT}/policy.log"' in script
    assert 'tail -240 "$log_path"' in script
    assert 'RESULT_HF_REPO="${RMBENCH_RESULT_HF_REPO:-}"' in script
    assert "publish_results" in script
    assert 'hf upload "$RESULT_HF_REPO"' in script
    assert 'WANDB_REQUIRED="${RMBENCH_WANDB_REQUIRED:-0}"' in script
    assert "start_wandb_logger" in script
    assert "wandb.Video" in script
    assert '"eval/completed_trials"' in script
    assert '"eval/success_rate"' in script
    assert '"eval/videos/{view}"' in script
    assert 'SKIP_RUNTIME_STAGE="${RMBENCH_SKIP_RUNTIME_STAGE:-1}"' in script
    assert 'if (( ! SKIP_RUNTIME_STAGE )); then' in script
    assert 'Skipping runtime archive staging; using pre-staged runtime:' in script
    assert "run_as_root() {" in script
    assert 'run_as_root tar --extract --file "${RUNTIME_STAGE}/uv-python.tar" --directory /' in script
    assert 'run_as_root tar --extract --file "${RUNTIME_STAGE}/runtime.tar" --directory /' in script
    assert 'run_as_root env RMBENCH_BUILD_GPU="$EVALUATOR_GPU" bash "$rebuild_script"' in script
    assert 'UV_BIN="${RMBENCH_UV_BIN:-}"' in script
    assert 'UV_BIN="${HOME}/.local/bin/uv"' in script
    assert 'USE_STAGED_INPUT="${RMBENCH_USE_STAGED_INPUT:-0}"' in script
    assert 'if (( USE_STAGED_INPUT )); then' in script
    assert 'MEMORY_ENABLED="${RMBENCH_MEMORY_ENABLED:-1}"' in script
    assert 'memory_enabled = manifest.get("memory_enabled", True)' in script
    assert '"memory_enabled": bool(int(memory_enabled))' in script
    assert '"memory_enabled": memory_enabled' in script
    assert '"molmoact2_prompt"' in script
    assert 'if (( MEMORY_ENABLED )); then' in script
    assert 'MOLMOACT2_SOURCE="$MOLMOACT2_MOUNT"' in script
    assert 'Using staged evaluation inputs rooted at:' in script

    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "rmbench_eval/scripts/run_worker.sh" in wrapper
    assert "exec bash" in wrapper


def test_smoke_worker_gives_spawned_curobo_a_writable_private_cache() -> None:
    script = WORKER.read_text(encoding="utf-8")

    assert 'SIMULATOR_CACHE_ROOT="${WORK_ROOT}/simulator_cache"' in script
    assert '"$SIMULATOR_CACHE_ROOT/warp" "$SIMULATOR_CACHE_ROOT/torch_extensions"' in script
    assert '"XDG_CACHE_HOME=${SIMULATOR_CACHE_ROOT}"' in script
    assert '"WARP_CACHE_PATH=${SIMULATOR_CACHE_ROOT}/warp"' in script
    assert '"TORCH_EXTENSIONS_DIR=${SIMULATOR_CACHE_ROOT}/torch_extensions"' in script


def test_smoke_worker_bootstraps_ffmpeg_for_four_view_recording() -> None:
    script = WORKER.read_text(encoding="utf-8")

    assert "ensure_ffmpeg() {" in script
    assert 'apt-get install -y -qq --no-install-recommends ffmpeg' in script
    assert "ffmpeg installation did not provide ffmpeg" in script
    assert "ensure_ffmpeg" in script.split('if (( PREPARE_ONLY )); then', 1)[1]


def test_smoke_bundle_publishes_model_only_hf_artifacts() -> None:
    source = PREPARE_BUNDLE.read_text(encoding="utf-8")

    assert "Amazon-FAR/imprint_models" in source
    assert "Amazon-FAR/imprint_datasets" in source
    assert "simulation/rmbench/molmoact2" in source
    assert "simulation/rmbench/molmoact2_eval_assets" in source
    assert "model.pt" in source
    assert "optim.pt" not in source
    assert "model_and_optim" not in source
    assert "hf upload-large-folder" in source
    assert "molmoact2-src.tar.gz" not in source
    assert "pi05-src.tar.gz" not in source


def test_b200_curobo_rebuild_restores_missing_cuda_apt_source() -> None:
    source = REBUILD_CUROBO.read_text(encoding="utf-8")

    assert "cuda-keyring_1.1-1_all.deb" in source
    assert "dpkg --force-confmiss -i /tmp/cuda-keyring.deb" in source
    assert 'apt-get install -y -qq --no-install-recommends "cuda-toolkit-${CUDA_TOOLKIT_VERSION}"' in source


def test_molmoact2_b200_worker_uses_the_b200_runtime_contract() -> None:
    script = WORKER.read_text(encoding="utf-8")

    assert "ensure_b200_curobo_runtime() {" in script
    assert "MolmoAct2 B200 simulator requires two physical GPUs" not in script
    assert "(( REQUIRED_GPU_COUNT == 2 ))" not in script
    assert 'RMBENCH_BUILD_GPU="$EVALUATOR_GPU"' in script
    assert "rebuild_curobo_b200_in_place.sh" in script
    assert "MotionGenConfig.load_from_robot_config" in script
    assert "curobo_sm100_motion_gen=ok" in script


def test_single_gpu_mode_runs_policy_and_evaluator_on_one_gpu() -> None:
    script = WORKER.read_text(encoding="utf-8")

    assert 'SINGLE_GPU="${RMBENCH_SINGLE_GPU:-0}"' in script
    assert '"CUDA_VISIBLE_DEVICES=${EVALUATOR_GPU}"' in script
    assert 'EVALUATOR_GPU="$WORKER_GPU"' in script
    assert '"RMBENCH_RENDER_DEVICE=cuda:0"' in script
    assert "B200 CuRobo preflight passed" in script
