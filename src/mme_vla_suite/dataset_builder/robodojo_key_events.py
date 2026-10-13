from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    temporal_downsample: int = 2
    event_radius: int = 5
    derivative_threshold: float = 0.02
    minimum_excursion: float = 0.25
    reopen_fraction: float = 0.70
    stationary_delta: float = 0.005
    stationary_frames: int = 2
    sustained_motion_frames: int = 3


@dataclass(frozen=True)
class GripperTransition:
    raw_start: int
    raw_end: int
    arm: str
    kind: str
    excursion: float


_GRIPPER_CHANNELS = {"left": 6, "right": 13}
_ARM_POSE_SLICES = {"left": slice(0, 6), "right": slice(7, 13)}


def raw_to_training_step(raw_frame: int, downsample: int) -> int:
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    return int(math.floor(raw_frame / downsample + 0.5))


def union_event_windows(
    events: list[dict], *, episode_steps: int, radius: int
) -> list[int]:
    if not isinstance(episode_steps, Integral) or episode_steps <= 0:
        raise ValueError("episode_steps must be a positive integer")
    if not isinstance(radius, Integral) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")

    weighted: set[int] = set()
    for index, event in enumerate(events):
        center = event.get("training_step") if isinstance(event, dict) else None
        if (
            not isinstance(center, Integral)
            or isinstance(center, bool)
            or not 0 <= center < episode_steps
        ):
            raise ValueError(
                f"event {index} center must be an integer in "
                f"[0, {episode_steps}); got {center!r}"
            )
        weighted.update(
            range(
                max(0, int(center) - int(radius)),
                min(int(episode_steps), int(center) + int(radius) + 1),
            )
        )
    return sorted(weighted)


def detect_gripper_transitions(
    actions: np.ndarray, config: DetectorConfig
) -> list[GripperTransition]:
    action_array = _validate_trajectory_array(actions, "actions")
    _validate_config(config)
    transitions: list[GripperTransition] = []

    for arm, channel in _GRIPPER_CHANNELS.items():
        values = action_array[:, channel]
        derivatives = np.diff(values, prepend=values[0])
        active = np.flatnonzero(
            np.abs(derivatives) > config.derivative_threshold
        )
        for group in _contiguous_groups(active):
            excursion = float(np.sum(derivatives[group]))
            if abs(excursion) <= config.minimum_excursion:
                continue
            transitions.append(
                GripperTransition(
                    raw_start=int(group[0]),
                    raw_end=int(group[-1]),
                    arm=arm,
                    kind="open" if excursion > 0 else "close",
                    excursion=excursion,
                )
            )

    return sorted(transitions, key=lambda transition: transition.raw_start)


def detect_key_events(
    actions: np.ndarray,
    states: np.ndarray,
    config: DetectorConfig,
) -> list[dict]:
    action_array = _validate_trajectory_array(actions, "actions")
    state_array = _validate_trajectory_array(states, "states")
    if action_array.shape != state_array.shape:
        raise ValueError(
            "actions and states must have matching shape (T, 14); "
            f"got {action_array.shape} and {state_array.shape}"
        )
    _validate_config(config)

    transitions = detect_gripper_transitions(action_array, config)
    closes = [
        transition for transition in transitions if transition.kind == "close"
    ]
    opens = [
        transition for transition in transitions if transition.kind == "open"
    ]
    if len(closes) != 6 or len(opens) != 6:
        raise ValueError(
            "expected exactly six close/open cycles (six close and six open "
            f"transitions); detected {len(closes)} close and {len(opens)} open"
        )

    pairs = _pair_close_open_transitions(closes, opens)
    uncover_pairs = pairs[-3:]
    events: list[dict] = []

    for cycle_index, (close, opening) in enumerate(uncover_pairs):
        preceding_opens = [
            transition
            for transition in opens
            if transition.raw_start < close.raw_start
        ]
        if not preceding_opens:
            raise ValueError(
                f"cycle {cycle_index} has no global open transition before "
                f"close frame {close.raw_start}"
            )
        preceding_open = max(
            preceding_opens, key=lambda transition: transition.raw_start
        )

        if preceding_open.arm == close.arm:
            base_frame, base_diagnostics = _same_arm_base_ready(
                state_array,
                preceding_open,
                close,
                cycle_index,
                config,
            )
        else:
            base_frame, base_diagnostics = _waiting_arm_base_ready(
                action_array,
                state_array,
                preceding_open,
                close,
                cycle_index,
                config,
            )

        if not base_frame < close.raw_start < opening.raw_start:
            raise ValueError(
                f"cycle {cycle_index} anchors are not ordered: base_ready="
                f"{base_frame}, pre_grasp={close.raw_start}, "
                f"pre_release={opening.raw_start}"
            )

        events.extend(
            [
                _event(
                    "base_ready",
                    cycle_index,
                    close.arm,
                    base_frame,
                    base_diagnostics,
                    config,
                ),
                _event(
                    "pre_grasp",
                    cycle_index,
                    close.arm,
                    close.raw_start,
                    _transition_diagnostics(close),
                    config,
                ),
                _event(
                    "pre_release",
                    cycle_index,
                    close.arm,
                    opening.raw_start,
                    _transition_diagnostics(opening),
                    config,
                ),
            ]
        )

    if len(events) != 9:
        raise ValueError(
            f"detector contract requires exactly nine events; got {len(events)}"
        )
    return events


def _validate_trajectory_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 14 or array.shape[0] < 2:
        raise ValueError(
            f"{name} must have shape (T, 14) with T >= 2; got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(
            f"{name} must be a real numeric array; got dtype {array.dtype}"
        )
    float_array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(float_array)):
        raise ValueError(f"{name} must contain only finite values")
    return float_array


def _validate_config(config: DetectorConfig) -> None:
    if config.temporal_downsample <= 0:
        raise ValueError("temporal_downsample must be positive")
    if config.event_radius < 0:
        raise ValueError("event_radius must be nonnegative")
    if config.derivative_threshold < 0:
        raise ValueError("derivative_threshold must be nonnegative")
    if config.minimum_excursion < 0:
        raise ValueError("minimum_excursion must be nonnegative")
    if not 0 <= config.reopen_fraction <= 1:
        raise ValueError("reopen_fraction must be between zero and one")
    if config.stationary_delta < 0:
        raise ValueError("stationary_delta must be nonnegative")
    if (
        not isinstance(config.stationary_frames, Integral)
        or isinstance(config.stationary_frames, bool)
        or config.stationary_frames < 2
    ):
        raise ValueError("stationary_frames must be an integer at least 2")
    if config.sustained_motion_frames <= 0:
        raise ValueError("sustained_motion_frames must be positive")


def _contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    return list(np.split(indices, split_points))


def _pair_close_open_transitions(
    closes: list[GripperTransition],
    opens: list[GripperTransition],
) -> list[tuple[GripperTransition, GripperTransition]]:
    pairs: list[tuple[GripperTransition, GripperTransition]] = []
    used_opens: set[GripperTransition] = set()

    for close in closes:
        candidates = [
            opening
            for opening in opens
            if opening.arm == close.arm
            and opening.raw_start > close.raw_end
        ]
        if not candidates:
            raise ValueError(
                f"cannot pair {close.arm} close at frame {close.raw_start}: "
                "no later same-arm open transition"
            )
        opening = min(candidates, key=lambda transition: transition.raw_start)
        if opening in used_opens:
            raise ValueError(
                f"{opening.arm} open transition at frame {opening.raw_start} "
                "would pair with more than one close"
            )
        used_opens.add(opening)
        pairs.append((close, opening))

    if len(used_opens) != len(opens):
        unused = sorted(
            (opening.raw_start, opening.arm)
            for opening in opens
            if opening not in used_opens
        )
        raise ValueError(f"unpaired open transitions at frames {unused}")
    return pairs


def _arm_motion_deltas(states: np.ndarray, arm: str) -> np.ndarray:
    pose = states[:, _ARM_POSE_SLICES[arm]]
    return np.max(np.abs(np.diff(pose, axis=0)), axis=1)


def _same_arm_base_ready(
    states: np.ndarray,
    preceding_open: GripperTransition,
    close: GripperTransition,
    cycle_index: int,
    config: DetectorConfig,
) -> tuple[int, dict]:
    motion_deltas = _arm_motion_deltas(states, close.arm)
    interval_start = preceding_open.raw_end
    interval_stop = close.raw_start
    motion_indices = (
        np.flatnonzero(
            motion_deltas[interval_start:interval_stop]
            > config.stationary_delta
        )
        + interval_start
    )
    sustained_runs = [
        group
        for group in _contiguous_groups(motion_indices)
        if group.size >= config.sustained_motion_frames
    ]
    if not sustained_runs:
        raise ValueError(
            f"cycle {cycle_index} cannot find same-arm base-ready anchor: "
            f"no sustained {close.arm} approach before frame {close.raw_start}"
        )

    approach = sustained_runs[-1]
    approach_start = int(approach[0])
    stationary_start = approach_start
    while (
        stationary_start > interval_start
        and motion_deltas[stationary_start - 1] <= config.stationary_delta
    ):
        stationary_start -= 1

    stationary_frame_count = approach_start - stationary_start + 1
    if stationary_frame_count < config.stationary_frames:
        raise ValueError(
            f"cycle {cycle_index} cannot find same-arm base-ready anchor: "
            f"the turnaround before approach frame {approach_start} has only "
            f"{stationary_frame_count} stationary frames"
        )

    return stationary_start, {
        "mode": "same_arm_turnaround",
        "preceding_open_raw_frame": preceding_open.raw_start,
        "preceding_arm": preceding_open.arm,
        "approach_raw_start": approach_start,
        "approach_raw_end": int(approach[-1] + 1),
        "stationary_frame_count": stationary_frame_count,
        "stationary_max_delta": float(
            np.max(motion_deltas[stationary_start:approach_start])
        ),
    }


def _waiting_arm_base_ready(
    actions: np.ndarray,
    states: np.ndarray,
    preceding_open: GripperTransition,
    close: GripperTransition,
    cycle_index: int,
    config: DetectorConfig,
) -> tuple[int, dict]:
    preceding_channel = _GRIPPER_CHANNELS[preceding_open.arm]
    closed_index = max(0, preceding_open.raw_start - 1)
    closed_value = float(actions[closed_index, preceding_channel])
    open_value = float(actions[preceding_open.raw_end, preceding_channel])
    reopen_threshold = closed_value + config.reopen_fraction * (
        open_value - closed_value
    )
    motion_deltas = _arm_motion_deltas(states, close.arm)
    last_candidate = close.raw_start - config.stationary_frames + 1

    for candidate in range(preceding_open.raw_start, last_candidate + 1):
        observed_gripper = float(states[candidate, preceding_channel])
        reopened = (
            observed_gripper >= reopen_threshold
            if open_value >= closed_value
            else observed_gripper <= reopen_threshold
        )
        stationary_stop = candidate + config.stationary_frames - 1
        stationary = np.all(
            motion_deltas[candidate:stationary_stop]
            <= config.stationary_delta
        )
        if reopened and stationary:
            return candidate, {
                "mode": "waiting_arm",
                "preceding_open_raw_frame": preceding_open.raw_start,
                "preceding_arm": preceding_open.arm,
                "reopen_threshold": reopen_threshold,
                "observed_gripper": observed_gripper,
                "stationary_window_end": stationary_stop,
                "stationary_max_delta": float(
                    np.max(motion_deltas[candidate:stationary_stop])
                ),
            }

    raise ValueError(
        f"cycle {cycle_index} cannot find waiting-arm base-ready anchor "
        f"between open frame {preceding_open.raw_start} and close frame "
        f"{close.raw_start}: reopen threshold {reopen_threshold:.6g} and "
        f"{config.stationary_frames}-frame stationarity never coincide"
    )


def _transition_diagnostics(transition: GripperTransition) -> dict:
    return {
        "mode": "gripper_transition",
        "kind": transition.kind,
        "transition_raw_end": transition.raw_end,
        "excursion": transition.excursion,
    }


def _event(
    event_type: str,
    cycle_index: int,
    arm: str,
    raw_frame: int,
    diagnostics: dict,
    config: DetectorConfig,
) -> dict:
    return {
        "event_type": event_type,
        "cycle_index": cycle_index,
        "arm": arm,
        "raw_frame": raw_frame,
        "training_step": raw_to_training_step(
            raw_frame, config.temporal_downsample
        ),
        "detector": diagnostics,
    }
