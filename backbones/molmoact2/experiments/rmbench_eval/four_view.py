"""Raw four-camera video recording for RMBench evaluation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Protocol

import numpy as np


VIEW_NAMES = (
    "head_camera",
    "left_camera",
    "right_camera",
    "third_person",
    "mosaic",
)
_SOURCE_VIEW_NAMES = VIEW_NAMES[:3]


class VideoWriter(Protocol):
    """Writer interface used by :class:`FourViewRecorder`."""

    def start(self) -> None:
        """Open the output stream."""

    def write(self, frame: np.ndarray) -> None:
        """Append one raw uint8 HWC frame."""

    def close(self) -> None:
        """Finalize the output stream."""


WriterFactory = Callable[[str, Path], VideoWriter]


class FfmpegVideoWriter:
    """Encode untouched BGR renderer frames to a deterministic H.264 MP4 stream."""

    def __init__(
        self,
        output: Path,
        popen: Callable[..., Any] = subprocess.Popen,
        fps: int = 20,
        wait_timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(fps, int) or isinstance(fps, bool) or fps < 1:
            raise ValueError("fps must be a positive integer")
        if (
            not isinstance(wait_timeout_seconds, (int, float))
            or isinstance(wait_timeout_seconds, bool)
            or not math.isfinite(wait_timeout_seconds)
            or wait_timeout_seconds <= 0
        ):
            raise ValueError("wait_timeout_seconds must be a finite positive number")
        self.output = Path(output)
        self.popen = popen
        self.fps = fps
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self._started = False
        self._closed = False
        self._frame_shape: tuple[int, int, int] | None = None
        self._process: Any | None = None

    def start(self) -> None:
        """Mark the writer ready; FFmpeg launches after dimensions are known."""
        if self._closed:
            raise RuntimeError("cannot start a closed FFmpeg writer")
        self._started = True

    def write(self, frame: np.ndarray) -> None:
        """Send one untouched uint8 HWC BGR frame to FFmpeg stdin."""
        if not self._started:
            raise RuntimeError("FfmpegVideoWriter.write requires start() first")
        if self._closed:
            raise RuntimeError("cannot write to a closed FFmpeg writer")
        self._validate_frame(frame)
        if self._frame_shape is None:
            self._frame_shape = frame.shape
            self.output.parent.mkdir(parents=True, exist_ok=True)
            process = self.popen(self._command(frame.shape), stdin=subprocess.PIPE)
            stdin = getattr(process, "stdin", None)
            if (
                stdin is None
                or not callable(getattr(stdin, "write", None))
                or not callable(getattr(stdin, "close", None))
            ):
                cleanup_error: BaseException | None = None
                if callable(getattr(stdin, "close", None)):
                    try:
                        stdin.close()
                    except BaseException as error:
                        cleanup_error = error
                reaping_error = self._terminate_and_wait(process)
                if cleanup_error is None:
                    cleanup_error = reaping_error
                self._closed = True
                if cleanup_error is not None:
                    raise RuntimeError(
                        "FFmpeg process has no usable stdin and could not be reaped"
                    ) from cleanup_error
                raise RuntimeError("FFmpeg process has no usable stdin")
            self._process = process
        elif frame.shape != self._frame_shape:
            raise ValueError(
                f"FFmpeg frame dimensions changed from {self._frame_shape} "
                f"to {frame.shape}"
            )
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        """Close FFmpeg stdin, wait for completion, and report encoder failure."""
        if self._closed:
            return
        if self._process is None:
            self._closed = True
            return
        try:
            first_error: BaseException | None = None
            try:
                self._process.stdin.close()
            except BaseException as error:
                first_error = error
            try:
                returncode = self._process.wait(timeout=self.wait_timeout_seconds)
                if returncode != 0 and first_error is None:
                    first_error = RuntimeError(f"FFmpeg exited with status {returncode}")
            except subprocess.TimeoutExpired as error:
                self._raise_after_timeout(self._process, error)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            if first_error is not None:
                raise first_error
        finally:
            self._closed = True

    def _raise_after_timeout(
        self,
        process: Any,
        initial_timeout: subprocess.TimeoutExpired,
    ) -> None:
        terminate_error: BaseException | None = None
        try:
            process.terminate()
        except BaseException as error:
            terminate_error = error
        try:
            process.wait(timeout=self.wait_timeout_seconds)
        except subprocess.TimeoutExpired:
            kill_error: BaseException | None = None
            try:
                process.kill()
            except BaseException as error:
                kill_error = error
            try:
                process.wait(timeout=self.wait_timeout_seconds)
            except subprocess.TimeoutExpired as kill_timeout:
                raise RuntimeError(
                    "FFmpeg remained unresponsive after kill"
                ) from kill_timeout
            except BaseException as error:
                raise RuntimeError(
                    "FFmpeg failed while waiting after kill"
                ) from error
            message = (
                "FFmpeg close timed out after "
                f"{self.wait_timeout_seconds} seconds and required kill"
            )
            if kill_error is not None:
                message += " after kill raised an error"
            raise RuntimeError(message) from initial_timeout
        except BaseException as error:
            raise RuntimeError(
                "FFmpeg close timed out and termination cleanup failed"
            ) from error
        message = (
            "FFmpeg close timed out after "
            f"{self.wait_timeout_seconds} seconds; terminated child process"
        )
        if terminate_error is not None:
            message += " after terminate raised an error"
        raise RuntimeError(message) from initial_timeout

    def _terminate_and_wait(self, process: Any) -> BaseException | None:
        cleanup_error: BaseException | None = None
        try:
            process.terminate()
        except BaseException as error:
            cleanup_error = error
        try:
            process.wait(timeout=self.wait_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                process.wait(timeout=self.wait_timeout_seconds)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        return cleanup_error

    def _command(self, shape: tuple[int, int, int]) -> list[str]:
        height, width, _ = shape
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            str(self.output),
        ]

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError("FFmpeg frame must have dtype uint8")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("FFmpeg frame must have HWC shape with three channels")


def _ffmpeg_writer_factory(_name: str, output: Path) -> FfmpegVideoWriter:
    return FfmpegVideoWriter(output)


class FourViewRecorder:
    """Write three raw policy cameras, a raw third-person view, and a mosaic."""

    def __init__(
        self,
        output: Path,
        trial_index: int,
        writer_factory: WriterFactory | None = None,
    ) -> None:
        if not isinstance(trial_index, int) or isinstance(trial_index, bool):
            raise ValueError("trial_index must be an integer")
        if trial_index < 0:
            raise ValueError("trial_index must be non-negative")
        self.output = Path(output)
        self.output.joinpath("videos").mkdir(parents=True, exist_ok=True)
        factory = writer_factory or _ffmpeg_writer_factory
        self.writers = {
            name: factory(
                name,
                self.output / "videos" / f"trial_{trial_index:03d}_{name}.mp4",
            )
            for name in VIEW_NAMES
        }
        self._started = False
        self._frame_shape: tuple[int, int, int] | None = None

    def start(self) -> None:
        """Start every configured video writer."""
        attempted_writers: list[VideoWriter] = []
        try:
            for writer in self.writers.values():
                attempted_writers.append(writer)
                writer.start()
        except BaseException:
            for writer in attempted_writers:
                try:
                    writer.close()
                except BaseException:
                    pass
            raise
        self._started = True

    def write(self, observation: object, third_person: object) -> None:
        """Record raw source frames and a ``[head|left; right|third]`` mosaic."""
        if not self._started:
            raise RuntimeError("FourViewRecorder.write requires start() first")
        sources = {
            name: self._source_frame(observation, name) for name in _SOURCE_VIEW_NAMES
        }
        third_frame = self._validated_frame(third_person, "third_person")
        reference_shape = sources["head_camera"].shape
        for name, frame in (*sources.items(), ("third_person", third_frame)):
            if frame.shape != reference_shape:
                raise ValueError(
                    f"{name} dimensions {frame.shape} do not match "
                    f"head_camera {reference_shape}"
                )
        if self._frame_shape is None:
            self._frame_shape = reference_shape
        elif reference_shape != self._frame_shape:
            raise ValueError(
                f"frame dimensions changed from {self._frame_shape} "
                f"to {reference_shape}"
            )

        height, width, _ = reference_shape
        mosaic = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
        mosaic[:height, :width] = sources["head_camera"]
        mosaic[:height, width:] = sources["left_camera"]
        mosaic[height:, :width] = sources["right_camera"]
        mosaic[height:, width:] = third_frame

        for name, frame in sources.items():
            self.writers[name].write(frame)
        self.writers["third_person"].write(third_frame)
        self.writers["mosaic"].write(mosaic)

    def close(self) -> None:
        """Close every writer, preserving the first close failure."""
        first_error: BaseException | None = None
        for writer in self.writers.values():
            try:
                writer.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._started = False
        if first_error is not None:
            raise first_error

    @classmethod
    def _source_frame(cls, observation: object, name: str) -> np.ndarray:
        try:
            frame = observation["observation"][name]["rgb"]  # type: ignore[index]
        except (KeyError, TypeError) as error:
            raise ValueError(f"missing raw camera frame: {name}") from error
        return cls._validated_frame(frame, name)

    @staticmethod
    def _validated_frame(frame: object, name: str) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"{name} must be a numpy uint8 HWC frame")
        if frame.dtype != np.uint8:
            raise ValueError(f"{name} must have dtype uint8")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name} must have HWC shape with three channels")
        return frame


def validate_mp4_output(
    output: str | Path,
    *,
    width: int,
    height: int,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    """Verify an FFmpeg recorder artifact has the expected stream contract."""
    path = Path(output)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"MP4 output is missing or empty: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed with status {result.returncode}: {result.stderr}")
    try:
        stream = json.loads(result.stdout)["streams"][0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"ffprobe returned no usable video stream for {path}") from error
    if (
        stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or stream.get("width") != width
        or stream.get("height") != height
    ):
        raise ValueError(f"MP4 output contract mismatch: {path}")
