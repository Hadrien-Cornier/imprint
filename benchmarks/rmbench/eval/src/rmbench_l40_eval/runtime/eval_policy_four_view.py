"""Four-view overlay for the canonical RMBench ``eval_policy_client.py``.

This module loads the canonical evaluator from an explicit source path and
patches only its evaluation loop. It does not import or duplicate
``deploy_policy.py`` and never changes policy observations or image channels.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import functools
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any
import uuid

import numpy as np

try:
    from ..four_view import (
        FourViewRecorder,
        WriterFactory,
        configured_video_codec,
        video_encoder_args,
        video_encoder_subprocess_env,
    )
except ImportError:
    from rmbench_l40_eval.four_view import (
        FourViewRecorder,
        WriterFactory,
        configured_video_codec,
        video_encoder_args,
        video_encoder_subprocess_env,
    )


@contextmanager
def _canonical_working_directory(working_directory: Path) -> Iterator[None]:
    original_directory = Path.cwd()
    sys.path.insert(0, str(working_directory))
    os.chdir(working_directory)
    try:
        yield
    finally:
        os.chdir(original_directory)
        del sys.path[0]


def load_canonical_evaluator(
    source_path: str | Path,
    working_directory: str | Path,
) -> ModuleType:
    """Load a canonical evaluator with its required working directory."""
    source = Path(source_path).resolve()
    working_directory = Path(working_directory).resolve()
    if not source.is_file():
        raise ValueError(f"canonical eval_policy source is missing: {source}")
    if not working_directory.is_dir():
        raise ValueError(
            f"canonical evaluator working directory is missing: {working_directory}"
        )

    module_name = f"_rmbench_canonical_eval_policy_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _canonical_working_directory(working_directory):
            spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def configure_canonical_renderer(canonical: Any) -> None:
    """Apply an optional renderer hook exposed by newer evaluator revisions.

    Older RMBench checkouts create the SAPIEN renderer inside their task base
    class and select the simulator GPU through ``CUDA_VISIBLE_DEVICES``. Newer
    checkouts expose an explicit helper. Supporting both keeps the overlay
    compatible without modifying either canonical evaluator.
    """
    configure_renderer_device = getattr(
        canonical, "configure_renderer_device", None
    )
    if configure_renderer_device is not None:
        configure_renderer_device()


def configure_canonical_client_timeout(canonical: Any) -> None:
    """Increase the official TCP client's timeout for first-call compilation."""
    model_client = getattr(canonical, "ModelClient", None)
    if model_client is None:
        return
    raw_timeout = os.environ.get("RMBENCH_MODEL_CLIENT_TIMEOUT", "900")
    try:
        timeout = int(raw_timeout)
    except ValueError as error:
        raise ValueError(
            "RMBENCH_MODEL_CLIENT_TIMEOUT must be a positive integer"
        ) from error
    if timeout < 1:
        raise ValueError("RMBENCH_MODEL_CLIENT_TIMEOUT must be a positive integer")
    canonical.ModelClient = functools.partial(model_client, timeout=timeout)


def record_policy_observation(
    recorder: FourViewRecorder,
    observation: Any,
    third_person: Any,
) -> None:
    """Record an evaluator observation without transforming any policy input."""
    recorder.write(observation, third_person)


def _cache_clear_frequency(args: Mapping[str, Any]) -> int:
    configured = os.environ.get("RMBENCH_CLEAR_CACHE_FREQ")
    raw_frequency = configured if configured is not None else args["clear_cache_freq"]
    try:
        frequency = int(raw_frequency)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"cache clear frequency must be a positive integer, got {raw_frequency!r}"
        ) from error
    if frequency < 1:
        raise ValueError(
            f"cache clear frequency must be a positive integer, got {frequency}"
        )
    return frequency


def _recreate_task_env_per_trial() -> bool:
    return os.environ.get("RMBENCH_RECREATE_TASK_ENV_PER_TRIAL") == "1"


def install_four_view_evaluation_loop(
    canonical: Any,
    output: str | Path,
    writer_factory: WriterFactory | None = None,
    *,
    seed_start: int | None = None,
    seed_end: int | None = None,
    timing_output: str | Path | None = None,
    trial_count: int | None = None,
) -> Callable[..., tuple[int, int]]:
    """Replace canonical ``eval_policy`` with a four-view recording overlay."""
    if trial_count is not None and (
        not isinstance(trial_count, int)
        or isinstance(trial_count, bool)
        or trial_count < 1
    ):
        raise ValueError("trial_count must be a positive integer or null")
    if seed_start is None:
        if seed_end is not None:
            raise ValueError("seed_end requires start_seed")
    else:
        if (
            not isinstance(seed_start, int)
            or isinstance(seed_start, bool)
        ):
            raise ValueError("start_seed must be an integer")
        if seed_end is not None and (
            not isinstance(seed_end, int)
            or isinstance(seed_end, bool)
            or seed_end <= seed_start
        ):
            raise ValueError("seed window must be increasing integers")

    def eval_policy_with_four_view(
        task_name: str,
        task_env: Any,
        args: dict[str, Any],
        model: Any,
        st_seed: int,
        test_num: int = 100,
        video_size: str | None = None,
        instruction_type: str | None = None,
        policy_conda_env: str | None = None,
        end_seed: int | None = None,
    ) -> tuple[int, int]:
        del end_seed, policy_conda_env
        return _eval_policy_with_four_view(
            canonical,
            Path(output),
            writer_factory,
            task_name,
            task_env,
            args,
            model,
            st_seed if seed_start is None else seed_start,
            test_num if trial_count is None else trial_count,
            video_size,
            instruction_type,
            seed_end=seed_end,
            timing_output=timing_output,
        )

    canonical.eval_policy = eval_policy_with_four_view
    return eval_policy_with_four_view


def _eval_policy_with_four_view(
    canonical: Any,
    output: Path,
    writer_factory: WriterFactory | None,
    task_name: str,
    task_env: Any,
    args: dict[str, Any],
    model: Any,
    st_seed: int,
    test_num: int,
    video_size: str | None,
    instruction_type: str | None,
    *,
    seed_end: int | None = None,
    timing_output: str | Path | None = None,
) -> tuple[int, int]:
    """Mirror canonical seed acceptance and policy execution around raw capture."""
    expert_check = True
    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = _cache_clear_frequency(args)
    args["eval_mode"] = True
    timing_path = Path(timing_output) if timing_output is not None else None

    eval_func = canonical.eval_function_decorator(args["policy_name"], "eval")
    reset_func = canonical.eval_function_decorator(args["policy_name"], "reset_model")

    while succ_seed < test_num:
        preflight_seconds = 0.0
        if seed_end is not None and now_seed >= seed_end:
            raise RuntimeError("seed window exhausted before accepted trial")
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            preflight_started = time.perf_counter() if timing_path is not None else 0.0
            try:
                task_env.setup_demo(
                    now_ep_num=now_id,
                    seed=now_seed,
                    is_test=True,
                    **args,
                )
                episode_info = task_env.play_once()
                task_env.close_env()
                if timing_path is not None:
                    preflight_seconds = time.perf_counter() - preflight_started
            except canonical.UnStableError:
                task_env.close_env()
                if timing_path is not None:
                    preflight_seconds = time.perf_counter() - preflight_started
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
                print(
                    f"RMBench preflight setup failed for seed={now_seed}:\n"
                    f"{traceback.format_exc().rstrip()}",
                    flush=True,
                )
                task_env.close_env()
                if timing_path is not None:
                    preflight_seconds = time.perf_counter() - preflight_started
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if not task_env.plan_success or not task_env.check_success():
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        succ_seed += 1
        args["render_freq"] = render_freq

        trial_started = time.perf_counter() if timing_path is not None else 0.0
        phase_seconds = {
            "preflight": preflight_seconds,
            "setup": 0.0,
            "canonical_video_start": 0.0,
            "recorder_start": 0.0,
            "reset_model": 0.0,
            "get_obs": 0.0,
            "third_view": 0.0,
            "recorder_write": 0.0,
            "policy_eval": 0.0,
            "artifact_write": 0.0,
            "recorder_close": 0.0,
            "canonical_video_close": 0.0,
            "environment_close": 0.0,
        }
        call_seconds = {
            "get_obs": [],
            "third_view": [],
            "recorder_write": [],
            "policy_eval": [],
        }
        setup_started = time.perf_counter() if timing_path is not None else 0.0
        task_env.setup_demo(
            now_ep_num=now_id,
            seed=now_seed,
            is_test=True,
            **args,
        )
        results = canonical.generate_episode_descriptions(
            args["task_name"],
            [episode_info["info"]],
            test_num,
        )
        task_env.set_instruction(
            instruction=np.random.choice(results[0][instruction_type])
        )
        if timing_path is not None:
            phase_seconds["setup"] = time.perf_counter() - setup_started
        canonical_video_enabled = task_env.eval_video_path is not None
        original_get_obs = task_env.get_obs
        recorder: FourViewRecorder | None = None
        try:
            if canonical_video_enabled:
                canonical_video_started = (
                    time.perf_counter() if timing_path is not None else 0.0
                )
                _start_canonical_video(task_env, video_size)
                if timing_path is not None:
                    phase_seconds["canonical_video_start"] = (
                        time.perf_counter() - canonical_video_started
                    )
            recorder_started = time.perf_counter() if timing_path is not None else 0.0
            recorder = FourViewRecorder(output, now_id, writer_factory)
            recorder.start()
            if timing_path is not None:
                phase_seconds["recorder_start"] = time.perf_counter() - recorder_started

            def recording_get_obs(*get_obs_args: Any, **get_obs_kwargs: Any) -> Any:
                if timing_path is None:
                    observation = original_get_obs(*get_obs_args, **get_obs_kwargs)
                    record_policy_observation(
                        recorder,
                        observation,
                        task_env.cameras.get_third_view_rgb(),
                    )
                    return observation
                get_obs_started = time.perf_counter()
                observation = original_get_obs(*get_obs_args, **get_obs_kwargs)
                get_obs_seconds = time.perf_counter() - get_obs_started
                phase_seconds["get_obs"] += get_obs_seconds
                call_seconds["get_obs"].append(get_obs_seconds)
                third_view_started = time.perf_counter()
                third_person = task_env.cameras.get_third_view_rgb()
                third_view_seconds = time.perf_counter() - third_view_started
                phase_seconds["third_view"] += third_view_seconds
                call_seconds["third_view"].append(third_view_seconds)
                recorder_write_started = time.perf_counter()
                record_policy_observation(
                    recorder,
                    observation,
                    third_person,
                )
                recorder_write_seconds = time.perf_counter() - recorder_write_started
                phase_seconds["recorder_write"] += recorder_write_seconds
                call_seconds["recorder_write"].append(recorder_write_seconds)
                return observation

            task_env.get_obs = recording_get_obs
            succ = False
            reset_started = time.perf_counter() if timing_path is not None else 0.0
            reset_func(model)
            if timing_path is not None:
                phase_seconds["reset_model"] = time.perf_counter() - reset_started
            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                policy_eval_started = time.perf_counter() if timing_path is not None else 0.0
                eval_func(task_env, model, observation)
                if timing_path is not None:
                    policy_eval_seconds = time.perf_counter() - policy_eval_started
                    phase_seconds["policy_eval"] += policy_eval_seconds
                    call_seconds["policy_eval"].append(policy_eval_seconds)
                if task_env.eval_success:
                    succ = True
                    break
            if succ:
                task_env.suc += 1
            artifact_write_started = time.perf_counter() if timing_path is not None else 0.0
            _append_canonical_trial_artifacts(
                args,
                trial_results_path=output / "trials.jsonl",
                task=task_name,
                trial=now_id,
                seed=now_seed,
                success=succ,
                max_reward=task_env.max_reward,
                steps=task_env.take_action_cnt,
            )
            if timing_path is not None:
                phase_seconds["artifact_write"] = (
                    time.perf_counter() - artifact_write_started
                )
        finally:
            try:
                if recorder is not None:
                    recorder_close_started = (
                        time.perf_counter() if timing_path is not None else 0.0
                    )
                    recorder.close()
                    if timing_path is not None:
                        phase_seconds["recorder_close"] = (
                            time.perf_counter() - recorder_close_started
                        )
            finally:
                try:
                    if canonical_video_enabled:
                        canonical_video_close_started = (
                            time.perf_counter() if timing_path is not None else 0.0
                        )
                        task_env._del_eval_video_ffmpeg()
                        if timing_path is not None:
                            phase_seconds["canonical_video_close"] = (
                                time.perf_counter() - canonical_video_close_started
                            )
                finally:
                    try:
                        task_env.get_obs = original_get_obs
                    finally:
                        environment_close_started = (
                            time.perf_counter() if timing_path is not None else 0.0
                        )
                        task_env.close_env(
                            clear_cache=((succ_seed + 1) % clear_cache_freq == 0)
                        )
                        if timing_path is not None:
                            phase_seconds["environment_close"] = (
                                time.perf_counter() - environment_close_started
                            )

        if timing_path is not None:
            phase_seconds["total_wall"] = time.perf_counter() - trial_started
            _append_timing_record(
                timing_path,
                {
                    "task": task_name,
                    "trial": now_id,
                    "seed": now_seed,
                    "success": bool(succ),
                    "steps": int(task_env.take_action_cnt),
                    "phase_seconds": phase_seconds,
                    "call_counts": {
                        name: len(samples) for name, samples in call_seconds.items()
                    },
                    "call_summaries": {
                        name: _summarize_timing_samples(samples)
                        for name, samples in call_seconds.items()
                    },
                    "call_samples_seconds": call_seconds,
                },
            )

        now_id += 1
        if task_env.render_freq:
            task_env.viewer.close()
        task_env.test_num += 1
        now_seed += 1
        if succ_seed < test_num and _recreate_task_env_per_trial():
            previous_successes = task_env.suc
            task_env = canonical.class_decorator(task_name)
            task_env.suc = previous_successes
            task_env.test_num = now_id

    (output / "summary.json").write_text(
        json.dumps(
            {
                "task": task_name,
                "trials": test_num,
                "successes": int(task_env.suc),
                "success_rate": float(task_env.suc) / float(test_num),
                "start_seed": st_seed,
                "next_seed": now_seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return now_seed, task_env.suc


def _summarize_timing_samples(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    ordered = sorted(samples)
    last_index = len(ordered) - 1
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[round(last_index * 0.50)],
        "p95": ordered[round(last_index * 0.95)],
    }


def _append_timing_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")


def _append_canonical_trial_artifacts(
    args: Mapping[str, Any],
    *,
    trial_results_path: Path,
    task: str,
    trial: int,
    seed: int,
    success: bool,
    max_reward: Any,
    steps: Any,
) -> None:
    """Append canonical-compatible trial and log records before env teardown."""
    result = "Success" if success else "Fail"
    normalized_reward = float(max_reward)
    normalized_steps = int(steps)
    log_file = args.get("log_file")
    if log_file is not None:
        try:
            with Path(log_file).open("a", encoding="utf-8") as output_file:
                output_file.write(
                    f"episode_id={trial}, seed={seed}, result={result}, "
                    f"max_reward={normalized_reward}, steps={normalized_steps}\n"
                )
        except Exception as error:
            print(f"[Log Warning] Failed to write log: {error}")
    with trial_results_path.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(
                {
                    "task": task,
                    "trial": trial,
                    "seed": seed,
                    "success": bool(success),
                    "max_reward": normalized_reward,
                    "steps": normalized_steps,
                }
            )
            + "\n"
        )


def _start_canonical_video(task_env: Any, video_size: str | None) -> None:
    """Preserve the canonical evaluator's optional single-view video behavior."""
    codec = configured_video_codec()
    encoder_args = (
        ["-vcodec", "libx264", "-crf", "23"]
        if codec == "libx264"
        else video_encoder_args(codec)
    )
    popen_kwargs: dict[str, Any] = {"stdin": subprocess.PIPE}
    child_env = video_encoder_subprocess_env(codec)
    if child_env is not None:
        popen_kwargs["env"] = child_env
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            *encoder_args,
            f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
        ],
        **popen_kwargs,
    )
    task_env._set_eval_video_ffmpeg(ffmpeg)


def run_canonical_overlay(
    canonical_source: str | Path,
    working_directory: str | Path,
    user_args: Mapping[str, Any],
    output: str | Path,
) -> None:
    """Load, patch, and run canonical ``main`` from explicit worker inputs."""
    seed_start = user_args.get("start_seed")
    seed_end = user_args.get("end_seed")
    if not isinstance(seed_start, int) or isinstance(seed_start, bool):
        raise ValueError("start_seed must be an integer")
    if seed_end is not None and (
        not isinstance(seed_end, int)
        or isinstance(seed_end, bool)
        or seed_end <= seed_start
    ):
        raise ValueError("end_seed must be an increasing integer or null")
    working_directory = Path(working_directory).resolve()
    canonical = load_canonical_evaluator(canonical_source, working_directory)
    configure_canonical_renderer(canonical)
    configure_canonical_client_timeout(canonical)
    timing_output_setting = os.environ.get("RMBENCH_TIMING_OUTPUT")
    timing_output = (
        Path(timing_output_setting) if timing_output_setting is not None else None
    )
    install_four_view_evaluation_loop(
        canonical,
        output,
        seed_start=seed_start,
        seed_end=seed_end,
        timing_output=timing_output,
        trial_count=user_args.get("test_num"),
    )
    with _canonical_working_directory(working_directory):
        canonical.main(dict(user_args))


def main(argv: list[str] | None = None) -> None:
    """Run the overlay from a worker-provided JSON argument mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-eval-policy", type=Path, required=True)
    parser.add_argument("--canonical-working-directory", type=Path, required=True)
    parser.add_argument("--user-args-json", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--execution_horizon", type=int, required=True)
    parser.add_argument("--start_seed", type=int, required=True)
    parser.add_argument("--end_seed", type=int)
    parsed = parser.parse_args(argv)
    with parsed.user_args_json.open(encoding="utf-8") as input_file:
        user_args = json.load(input_file)
    if not isinstance(user_args, dict):
        raise ValueError("--user-args-json must contain an object")
    if user_args.get("task_name") != parsed.task_name:
        raise ValueError("--task_name must match the evaluator argument mapping")
    if user_args.get("execution_horizon") != parsed.execution_horizon:
        raise ValueError(
            "--execution_horizon must match the evaluator argument mapping"
        )
    if user_args.get("start_seed") != parsed.start_seed:
        raise ValueError("--start_seed must match the evaluator argument mapping")
    if user_args.get("end_seed") != parsed.end_seed:
        raise ValueError("--end_seed must match the evaluator argument mapping")
    run_canonical_overlay(
        parsed.canonical_eval_policy,
        parsed.canonical_working_directory,
        user_args,
        parsed.video_output,
    )


if __name__ == "__main__":
    main()
