import numpy as np
import pytest

from mme_vla_suite.dataset_builder.robodojo_key_events import (
    DetectorConfig,
    GripperTransition,
    detect_gripper_transitions,
    detect_key_events,
    raw_to_training_step,
    union_event_windows,
)


def synthetic_six_cycle_trajectory() -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((240, 14), dtype=np.float32)
    states = np.zeros((240, 14), dtype=np.float32)
    actions[:, (6, 13)] = 1.0
    states[:, (6, 13)] = 1.0

    cycles = [
        (20, 30, 6),
        (50, 60, 6),
        (80, 90, 13),
        (130, 140, 13),
        (165, 175, 6),
        (200, 210, 13),
    ]
    for close, release, gripper in cycles:
        actions[close:release, gripper] = 0.5
        # State feedback follows the command by one frame.
        states[close + 1 : release + 1, gripper] = 0.5

    # The right arm is reused after cycle 3. It retreats, holds a stationary
    # turnaround, then makes a sustained approach into the close at frame 130.
    states[95:105, 7] = np.linspace(0.0, 0.4, 10)
    states[105:116, 7] = 0.4
    states[116:131, 7] = np.linspace(0.44, 1.0, 15)
    states[131:, 7] = 1.0

    actions[:, :6] = states[:, :6]
    actions[:, 7:13] = states[:, 7:13]
    return actions, states


def test_detector_config_defaults_are_stable():
    assert DetectorConfig() == DetectorConfig(
        temporal_downsample=2,
        event_radius=5,
        derivative_threshold=0.02,
        minimum_excursion=0.25,
        reopen_fraction=0.70,
        stationary_delta=0.005,
        stationary_frames=2,
        sustained_motion_frames=3,
    )


def test_raw_to_training_step_uses_nearest_half_up():
    assert raw_to_training_step(382, 2) == 191
    assert raw_to_training_step(383, 2) == 192


def test_raw_to_training_step_rejects_nonpositive_downsample():
    with pytest.raises(ValueError, match="downsample must be positive"):
        raw_to_training_step(10, 0)


def test_union_event_windows_does_not_stack_overlaps():
    events = [{"training_step": 10}, {"training_step": 12}]

    assert union_event_windows(events, episode_steps=30, radius=5) == list(
        range(5, 18)
    )


@pytest.mark.parametrize(
    ("center", "expected"),
    [
        (0, list(range(0, 6))),
        (9, list(range(4, 10))),
    ],
)
def test_union_event_windows_clips_episode_endpoints(center, expected):
    events = [{"training_step": center}]

    assert union_event_windows(events, episode_steps=10, radius=5) == expected


@pytest.mark.parametrize(
    ("events", "episode_steps", "radius", "message"),
    [
        ([{"training_step": 0}], 0, 5, "episode_steps"),
        ([{"training_step": 0}], 10, -1, "radius"),
        ([{"training_step": 10}], 10, 1, "center"),
        ([{"training_step": 1.5}], 10, 1, "center"),
    ],
)
def test_union_event_windows_rejects_invalid_inputs(
    events, episode_steps, radius, message
):
    with pytest.raises(ValueError, match=message):
        union_event_windows(events, episode_steps=episode_steps, radius=radius)


def test_detect_gripper_transitions_finds_all_six_cycles():
    actions, _ = synthetic_six_cycle_trajectory()

    transitions = detect_gripper_transitions(actions, DetectorConfig())

    expected_starts = [20, 30, 50, 60, 80, 90, 130, 140, 165, 175, 200, 210]
    expected_arms = [
        "left",
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "left",
        "left",
        "right",
        "right",
    ]
    expected_kinds = ["close", "open"] * 6
    assert [transition.raw_start for transition in transitions] == expected_starts
    assert [transition.arm for transition in transitions] == expected_arms
    assert [transition.kind for transition in transitions] == expected_kinds
    assert all(isinstance(transition, GripperTransition) for transition in transitions)
    np.testing.assert_allclose(
        [transition.excursion for transition in transitions],
        [-0.5, 0.5] * 6,
    )


def test_transition_detector_preserves_uint8_close_direction():
    actions = np.zeros((4, 14), dtype=np.uint8)
    actions[:2, 6] = 1

    transitions = detect_gripper_transitions(actions, DetectorConfig())

    assert transitions == [
        GripperTransition(
            raw_start=2,
            raw_end=2,
            arm="left",
            kind="close",
            excursion=-1.0,
        )
    ]


def test_transition_detector_groups_contiguous_multiframe_excursion():
    actions = np.zeros((10, 14), dtype=np.float32)
    actions[:, 6] = [1.0, 1.0, 0.99, 0.84, 0.64, 0.39, 0.39, 0.39, 0.49, 0.59]
    config = DetectorConfig(
        derivative_threshold=0.05,
        minimum_excursion=0.25,
    )

    transitions = detect_gripper_transitions(actions, config)

    assert len(transitions) == 1
    assert transitions[0].raw_start == 3
    assert transitions[0].raw_end == 5
    assert transitions[0].kind == "close"
    assert transitions[0].excursion == pytest.approx(-0.60)


def test_detector_pairs_simultaneous_different_arm_opens_independently():
    actions, states = synthetic_six_cycle_trajectory()
    actions[60:90, 6] = 0.5
    states[61:91, 6] = 0.5

    transitions = detect_gripper_transitions(actions, DetectorConfig())
    simultaneous_opens = [
        transition
        for transition in transitions
        if transition.kind == "open" and transition.raw_start == 90
    ]
    events = detect_key_events(actions, states, DetectorConfig())

    assert [transition.arm for transition in simultaneous_opens] == [
        "left",
        "right",
    ]
    assert [event["raw_frame"] for event in events[1::3]] == [130, 165, 200]
    assert [event["raw_frame"] for event in events[2::3]] == [140, 175, 210]


def test_detector_rejects_fewer_than_six_cycles():
    actions = np.zeros((64, 14), dtype=np.float32)
    states = np.zeros((64, 14), dtype=np.float32)

    with pytest.raises(ValueError, match="six close/open cycles"):
        detect_key_events(actions, states, DetectorConfig())


@pytest.mark.parametrize(
    ("actions", "states"),
    [
        (np.zeros((10, 13)), np.zeros((10, 14))),
        (np.zeros((10, 14)), np.zeros((11, 14))),
    ],
)
def test_detector_rejects_invalid_shapes(actions, states):
    with pytest.raises(ValueError, match="shape"):
        detect_key_events(actions, states, DetectorConfig())


def test_transition_detector_rejects_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        detect_gripper_transitions(np.zeros((10, 13)), DetectorConfig())


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 14), dtype=np.complex64),
        np.full((2, 14), "not-a-number", dtype=object),
    ],
)
def test_transition_detector_rejects_non_real_numeric_actions(actions):
    with pytest.raises(ValueError, match="actions must be a real numeric array"):
        detect_gripper_transitions(actions, DetectorConfig())


@pytest.mark.parametrize(
    ("target", "value"),
    [("actions", np.nan), ("states", np.inf)],
)
def test_detector_rejects_nonfinite_trajectories(target, value):
    actions, states = synthetic_six_cycle_trajectory()
    trajectory = actions if target == "actions" else states
    trajectory[0, 0] = value

    with pytest.raises(
        ValueError, match=f"{target} must contain only finite values"
    ):
        detect_key_events(actions, states, DetectorConfig())


@pytest.mark.parametrize("stationary_frames", [1, 2.5])
def test_detector_rejects_invalid_stationary_frame_count(stationary_frames):
    actions = np.zeros((2, 14), dtype=np.float32)
    config = DetectorConfig(stationary_frames=stationary_frames)

    with pytest.raises(
        ValueError, match="stationary_frames must be an integer at least 2"
    ):
        detect_gripper_transitions(actions, config)


def test_detector_returns_nine_ordered_uncover_events():
    actions, states = synthetic_six_cycle_trajectory()

    events = detect_key_events(actions, states, DetectorConfig())

    assert [event["event_type"] for event in events] == [
        "base_ready",
        "pre_grasp",
        "pre_release",
        "base_ready",
        "pre_grasp",
        "pre_release",
        "base_ready",
        "pre_grasp",
        "pre_release",
    ]
    assert [event["cycle_index"] for event in events] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [event["arm"] for event in events[::3]] == ["right", "left", "right"]
    assert [event["raw_frame"] for event in events[::3]] == [104, 141, 176]
    assert [event["raw_frame"] for event in events[1::3]] == [130, 165, 200]
    assert [event["raw_frame"] for event in events[2::3]] == [140, 175, 210]
    assert [event["training_step"] for event in events[1::3]] == [65, 83, 100]
    assert [event["detector"]["mode"] for event in events[::3]] == [
        "same_arm_turnaround",
        "waiting_arm",
        "waiting_arm",
    ]
    assert all(
        set(event)
        == {
            "event_type",
            "cycle_index",
            "arm",
            "raw_frame",
            "training_step",
            "detector",
        }
        for event in events
    )


def test_detector_fails_when_waiting_arm_never_becomes_stationary():
    actions, states = synthetic_six_cycle_trajectory()
    states[140:166, 0] = np.arange(26, dtype=np.float32) * 0.01

    with pytest.raises(ValueError, match="cycle 1.*waiting-arm base-ready anchor"):
        detect_key_events(actions, states, DetectorConfig())
