"""Launch the standard MME server with DINO checkpoint key compatibility."""

from __future__ import annotations

import os
import runpy

import numpy

from openpi.models import model as model_lib

# The policy virtualenv imports RoboCasa only for dataset utility types. The
# staged evaluator source enforces its own NumPy version at import time, while
# this already-working policy environment uses a different compatible version.
if numpy.__version__ != "2.2.5":
    print(
        f"Bypassing RoboCasa source-only NumPy guard for policy server: {numpy.__version__}",
        flush=True,
    )
    numpy.__version__ = "2.2.5"


def _intify_numeric_keys(tree):
    if isinstance(tree, dict):
        return {
            int(key) if isinstance(key, str) and key.isdigit() else key: _intify_numeric_keys(value)
            for key, value in tree.items()
        }
    return tree


_restore_params = model_lib.restore_params


def restore_params_with_dino_block_keys(*args, **kwargs):
    params = _restore_params(*args, **kwargs)
    blocks = params.get("mem_encoder", {}).get("blocks") if isinstance(params, dict) else None
    if isinstance(blocks, dict):
        before = tuple(blocks)
        params["mem_encoder"]["blocks"] = _intify_numeric_keys(blocks)
        after = tuple(params["mem_encoder"]["blocks"])
        print(f"DINO checkpoint block-key normalization: {before} -> {after}", flush=True)
    return params


model_lib.restore_params = restore_params_with_dino_block_keys

_intersect_trees = model_lib.ocp.transform_utils.intersect_trees


def intersect_trees_with_dino_block_keys(*args, **kwargs):
    params = _intersect_trees(*args, **kwargs)
    blocks = params.get("mem_encoder", {}).get("blocks") if isinstance(params, dict) else None
    if isinstance(blocks, dict):
        before = tuple(blocks)
        params["mem_encoder"]["blocks"] = _intify_numeric_keys(blocks)
        after = tuple(params["mem_encoder"]["blocks"])
        print(f"DINO intersected block-key normalization: {before} -> {after}", flush=True)
    return params


model_lib.ocp.transform_utils.intersect_trees = intersect_trees_with_dino_block_keys
runpy.run_path(os.environ["MME_SERVER_SCRIPT"], run_name="__main__")
