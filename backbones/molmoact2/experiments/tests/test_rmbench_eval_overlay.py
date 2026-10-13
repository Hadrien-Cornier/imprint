from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rmbench_eval.four_view import FourViewRecorder, VIEW_NAMES
from rmbench_eval.runtime.eval_policy_four_view import install_four_view_evaluation_loop


def test_overlay_direct_cli_help_works_from_repository_root():
    experiments_root = Path(__file__).resolve().parents[1]
    repository_root = experiments_root.parent
    overlay_path = (
        experiments_root / "rmbench_eval" / "runtime" / "eval_policy_four_view.py"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(overlay_path), "--help"],
        check=False,
        capture_output=True,
        cwd=repository_root,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--canonical-eval-policy" in result.stdout


class FakeWriter:
    def __init__(
        self,
        *,
        start_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.started = False
        self.closed = False
        self.start_error = start_error
        self.close_error = close_error
        self.frames: list[np.ndarray] = []

    def start(self) -> None:
        self.started = True
        if self.start_error is not None:
            raise self.start_error

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def test_four_view_recorder_writes_all_views_and_preserves_raw_bgr_frames(tmp_path):
    writers: dict[str, FakeWriter] = {}

    def writer_factory(name, _output):
        writer = FakeWriter()
        writers[name] = writer
        return writer

    frames = {
        "head_camera": np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8),
        "left_camera": np.full((4, 6, 3), [40, 50, 60], dtype=np.uint8),
        "right_camera": np.full((4, 6, 3), [70, 80, 90], dtype=np.uint8),
    }
    third_person = np.full((4, 6, 3), [100, 110, 120], dtype=np.uint8)
    recorder = FourViewRecorder(tmp_path, trial_index=4, writer_factory=writer_factory)

    recorder.start()
    recorder.write({"observation": {name: {"rgb": frame} for name, frame in frames.items()}}, third_person)
    recorder.close()

    assert tuple(writers) == VIEW_NAMES
    assert all(writer.started and writer.closed for writer in writers.values())
    np.testing.assert_array_equal(writers["head_camera"].frames, [frames["head_camera"]])
    np.testing.assert_array_equal(writers["left_camera"].frames, [frames["left_camera"]])
    np.testing.assert_array_equal(writers["right_camera"].frames, [frames["right_camera"]])
    np.testing.assert_array_equal(writers["third_person"].frames, [third_person])
    mosaic = writers["mosaic"].frames[0]
    np.testing.assert_array_equal(mosaic[:4, :6], frames["head_camera"])
    np.testing.assert_array_equal(mosaic[:4, 6:], frames["left_camera"])
    np.testing.assert_array_equal(mosaic[4:, :6], frames["right_camera"])
    np.testing.assert_array_equal(mosaic[4:, 6:], third_person)


def test_four_view_recorder_closes_attempted_writers_when_start_fails(tmp_path):
    writers: dict[str, FakeWriter] = {}
    start_error = RuntimeError("start failure")

    def writer_factory(name, _output):
        writer = FakeWriter(
            start_error=start_error if name == "right_camera" else None,
        )
        writers[name] = writer
        return writer

    recorder = FourViewRecorder(tmp_path, trial_index=4, writer_factory=writer_factory)

    with pytest.raises(RuntimeError, match="start failure"):
        recorder.start()

    attempted_names = ("head_camera", "left_camera", "right_camera")
    assert all(writers[name].started and writers[name].closed for name in attempted_names)
    assert not writers["third_person"].started
    assert not writers["third_person"].closed
    assert not writers["mosaic"].started
    assert not writers["mosaic"].closed


def test_four_view_recorder_closes_all_writers_and_surfaces_first_error(tmp_path):
    writers: dict[str, FakeWriter] = {}
    first_error = RuntimeError("first close failure")
    second_error = ValueError("second close failure")

    def writer_factory(name, _output):
        writer = FakeWriter(
            close_error={
                "left_camera": first_error,
                "mosaic": second_error,
            }.get(name),
        )
        writers[name] = writer
        return writer

    recorder = FourViewRecorder(tmp_path, trial_index=4, writer_factory=writer_factory)
    recorder.start()

    with pytest.raises(RuntimeError, match="first close failure"):
        recorder.close()

    assert all(writer.closed for writer in writers.values())


class FakeCameras:
    def get_third_view_rgb(self):
        return np.full((2, 2, 3), [100, 110, 120], dtype=np.uint8)


class FakeTaskEnvironment:
    def __init__(self) -> None:
        self.cameras = FakeCameras()
        self.eval_video_path = None
        self.suc = 0
        self.test_num = 0
        self.plan_success = True
        self.take_action_cnt = 0
        self.step_lim = 1
        self.eval_success = False
        self.max_reward = 0.0
        self.render_freq = 0
        self.closed = 0
        self.instructions = []

    def setup_demo(self, **_kwargs) -> None:
        self.take_action_cnt = 0
        self.eval_success = False

    def play_once(self):
        return {"info": {"episode": "fake"}}

    def close_env(self, **_kwargs) -> None:
        self.closed += 1

    def check_success(self) -> bool:
        return True

    def set_instruction(self, instruction) -> None:
        self.instructions.append(instruction)

    def get_obs(self):
        frame = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        return {
            "observation": {
                name: {"rgb": frame}
                for name in ("head_camera", "left_camera", "right_camera")
            }
        }


class FakeCanonical:
    class UnStableError(Exception):
        pass

    @staticmethod
    def generate_episode_descriptions(_task_name, _infos, _test_num):
        return [{"default": ["place block"]}]

    @staticmethod
    def eval_function_decorator(_policy_name, mode):
        if mode == "reset_model":
            return lambda _model: None

        def evaluate(task_env, _model, _observation):
            task_env.take_action_cnt += 1
            task_env.eval_success = True
            task_env.max_reward = 1.0

        return evaluate


def test_overlay_writes_canonical_trial_record_and_restores_get_obs(tmp_path):
    writers: dict[str, FakeWriter] = {}

    def writer_factory(name, _output):
        writer = FakeWriter()
        writers[name] = writer
        return writer

    canonical = FakeCanonical()
    task_env = FakeTaskEnvironment()
    original_get_obs = task_env.get_obs
    trial_results_path = tmp_path / "trials.jsonl"
    install_four_view_evaluation_loop(
        canonical,
        tmp_path,
        writer_factory,
        seed_start=7,
        seed_end=8,
    )

    next_seed, successes = canonical.eval_policy(
        "cover_block",
        task_env,
        {
            "policy_name": "fake",
            "task_name": "cover_block",
            "clear_cache_freq": 1,
            "render_freq": 0,
            "trial_results_path": trial_results_path,
        },
        object(),
        st_seed=0,
        test_num=1,
        instruction_type="default",
    )

    assert (next_seed, successes) == (8, 1)
    assert task_env.get_obs.__func__ is original_get_obs.__func__
    assert task_env.closed == 2
    assert all(writer.closed for writer in writers.values())
    assert trial_results_path.read_text(encoding="utf-8") == (
        '{"task": "cover_block", "trial": 0, "seed": 7, "success": true, '
        '"max_reward": 1.0, "steps": 1}\n'
    )
