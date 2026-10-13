"""Compatibility adapter for the legacy full-attention evaluation contract."""

from __future__ import annotations

import os

import robomme_tcp_policy as base


# RMBench emits BGR arrays. The successful August profile converted them to RGB
# for both the VLA and memory streams. This module is imported by the simulator
# as well as the policy server, so pin the value instead of requiring an env var
# that only the server process receives.
base.POLICY_INPUT_COLOR_ORDER = "bgr"

HISTORICAL_MEMORY_CAMERA_ORDER = (
    "head_camera",
    "left_camera",
    "right_camera",
)
configured_memory_camera_order = tuple(
    name.strip()
    for name in os.environ.get(
        "MEMFT_MEMORY_CAMERA_ORDER",
        ",".join(HISTORICAL_MEMORY_CAMERA_ORDER),
    ).split(",")
)
if configured_memory_camera_order != HISTORICAL_MEMORY_CAMERA_ORDER:
    raise ValueError(
        "legacy reference evaluation requires MEMFT_MEMORY_CAMERA_ORDER="
        + ",".join(HISTORICAL_MEMORY_CAMERA_ORDER)
    )
base.MEMORY_CAMERA_ORDER = configured_memory_camera_order
base.MEMORY_CAMERA_INDICES = tuple(
    base.CAMERA_ORDER.index(name) for name in base.MEMORY_CAMERA_ORDER
)


def get_model(usr_args):
    return base.get_model(usr_args)


eval = base.eval
reset_model = base.reset_model
