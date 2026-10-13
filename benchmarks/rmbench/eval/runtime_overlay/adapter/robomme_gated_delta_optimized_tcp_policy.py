"""Legacy-reference RMBench adapter with opt-in Gated DeltaNet optimizations."""

from __future__ import annotations

import types

import robomme_legacy_reference_tcp_policy as legacy
from gated_delta_eval_optimizations import (
    BatchedWanStreamingEncoder,
    install_optimized_runtime,
    optimized_warmup_runtime,
)


def get_model(usr_args):
    install_optimized_runtime()
    server = legacy.get_model(usr_args)
    if not getattr(server.policy, "_uses_gated_delta", False):
        raise RuntimeError("optimized adapter requires a Gated DeltaNet checkpoint")
    streamer = getattr(server.policy.mem_buffer, "_streamer", None)
    if not isinstance(streamer, BatchedWanStreamingEncoder):
        raise RuntimeError("policy did not construct the batched Wan streamer")
    server.warmup_runtime = types.MethodType(optimized_warmup_runtime, server)
    return server


eval = legacy.eval
reset_model = legacy.reset_model
