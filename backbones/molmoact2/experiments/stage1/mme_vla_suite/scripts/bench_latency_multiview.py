"""Multiview Option-D latency benchmark: how per-step inference latency scales with
the number of camera views V in {1,2,3} for the Wan video-tokenizer memory pipeline.

Multiple views concatenate along the token axis, so:
  * the frozen Wan tokenizer runs V times per chunk-fire (encode_multiview loops views),
  * the video-latent KV fed into the 256-token bottleneck is V x longer.
The 256 learned queries (bottleneck OUTPUT) stay fixed regardless of V*L, so the
downstream pi0.5 modulation seam and forward are V-independent; only the NEW upstream
costs (wan_encode, torch->jax handoff, bottleneck cross-attention over a V*L KV) scale.

This bench REUSES the single-view harness (scripts/bench_latency.py):
  * _t / _stats / _time_loop timing discipline (block_until_ready / cuda.synchronize,
    warmup discarded, median + p90 over >=30 iters),
  * the same CosmoBottleneckMemory config dict (cosmos_latent_dim=48 for Wan),
  * the same baseline pi0.5 sample_actions path for pi05_forward_ms / modulation_ms.

Components measured per (V, L)  -- all milliseconds, median + p90:
  wan_encode_ms_perfire   : real WanFrontend.encode_multiview on a
                            (temporal_factor, V, 128,128,3) uint8 clip (ONE chunk-fire,
                            V views). Fires once every temporal_factor steps.
  wan_encode_ms_amortized : perfire / temporal_factor (per-step amortized).
  torch_to_jax_ms         : real torch->jax handoff (device_put) of one chunk-fire's
                            Wan latents (V*64 tokens). Fires once per chunk-fire.
  bottleneck_ms           : CosmoBottleneckMemory forward on the (V,L) KV. The video
                            KV has num_chunks * V * 64 tokens (V-dependent, L-dependent);
                            output is a fixed 256 tokens. jitted + block_until_ready.
  modulation_ms           : cosmo-modulation sample_actions seam cost (V-INDEPENDENT:
                            the bottleneck emits 256 memory tokens regardless of V*L).
  pi05_forward_ms         : baseline pi0.5 sample_actions (no memory). V-INDEPENDENT.
  total_optionD_ms        : wan_encode_amortized + torch_to_jax_amortized + bottleneck
                            + modulation + pi05_forward.
  overhead_ms             : total_optionD - pi05_forward  (the headline).

modulation_ms / pi05_forward_ms are measured ONCE (V,L-independent) via the real
baseline + cosmo-modulation pi0.5 sample_actions (same code path as the single-view
bench), then reused for every (V,L) row -- see notes["pi05_modulation"]. This is
correct because those two components do not depend on the number of views: the
bottleneck compresses any V*L video KV down to a fixed 256-token memory summary
before it reaches the LLM seam. If the real pi0.5 build is unavailable (--fake or
load failure), CPU stand-ins are used and clearly marked.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _t():
    return time.perf_counter()


def _stats(ts):
    ts = sorted(ts)
    med = statistics.median(ts)
    p90 = ts[int(0.9 * len(ts)) - 1] if len(ts) >= 2 else ts[-1]
    return med, p90


def _time_loop(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        s = _t()
        fn()
        ts.append((_t() - s) * 1e3)
    return _stats(ts)


# Wan tokenizer geometry (see tests/cosmo/shapes.py + wan_frontend.py docstring).
WAN_TEMPORAL_FACTOR = 4
WAN_LATENT_DIM = 48
WAN_SPATIAL_TOKENS_128 = 64  # 8x8 grid at 128px input


def run_benchmark(views=(1, 2, 3), buffer_lengths=(64, 256, 1024, 2048),
                  iters=30, warmup=5, use_fake=False):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from omegaconf import OmegaConf

    from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory

    TF = WAN_TEMPORAL_FACTOR
    SPT = WAN_SPATIAL_TOKENS_128
    LDIM = WAN_LATENT_DIM

    # ---- Bottleneck config (mirrors cosmo-wan-bottleneck-modul.yaml) ----
    cfg = OmegaConf.create({
        "budget": 256, "memory_token_dim": 1024,
        "cosmo_bottleneck_memory": {
            "num_latents": 256, "num_cross_blocks": 2, "num_heads": 8,
            "cosmos_latent_dim": 48, "siglip_dim": 2048, "proprio_in_dim": 32,
            "action_in_dim": 24, "lang_emb_dim": 2048,
            "action_tokenizer_hidden": 512, "proprio_tokenizer_hidden": 512}})
    bneck = CosmoBottleneckMemory(config=cfg, rngs=nnx.Rngs(0), dtype=jnp.bfloat16)
    # __call__(cosmo_emb, recent_emb, action_emb, proprio_emb, cosmo_mask, lang_emb)
    bneck_fwd = nnx.jit(lambda m, a, b, c, d, e, f: m(a, b, c, d, e, f))

    # recent window: streaming_obs_horizon(16) * num_views. Recent SigLIP window is
    # per-step per-view; use a fixed modest recent length (V-scaled) as in the buffer.
    RECENT_FRAMES = 16  # streaming_obs_horizon
    SIGLIP_TOKENS_PER_FRAME = 64

    # ---------------------------------------------------------------------------
    # Real GPU components (built once): Wan frontend + pi0.5 baseline/cosmo models.
    # ---------------------------------------------------------------------------
    wan_fe = None
    baseline_sample = None
    cosmo_sample = None
    base_obs = None
    cosmo_obs_builder = None
    sample_key = None
    pi05_med = pi05_p90 = mod_med = mod_p90 = None

    if not use_fake:
        import torch
        from mme_vla_suite.models.representation.wan_frontend import WanFrontend

        wan_fe = WanFrontend(variant="Wan2.2-TI2V-5B", device="cuda")

        # pi0.5 baseline + cosmo-modulation (V-independent seam / forward), same path
        # as scripts/bench_latency.py.  Measured ONCE and reused for all (V,L).
        try:
            import dataclasses
            import openpi.shared.nnx_utils as nnx_utils
            from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
            from mme_vla_suite.models.integration.history_observation import HistAugObservation
            from mme_vla_suite.models.config.utils import get_history_config

            sample_key = jax.random.key(0)
            AH = 20
            base_cfg = HistoryPi0Config(pi05=True, action_horizon=AH, use_history=False,
                                        history_config=None, discrete_state_input=False)
            base_model = base_cfg.create(jax.random.key(0))
            baseline_sample = nnx_utils.module_jit(base_model.sample_actions)

            cosmo_cfg = HistoryPi0Config(pi05=True, action_horizon=AH, use_history=True,
                                         history_config="cosmo-wan-bottleneck-modul.yaml",
                                         discrete_state_input=False)
            cosmo_model = cosmo_cfg.create(jax.random.key(0))
            cosmo_sample = nnx_utils.module_jit(cosmo_model.sample_actions)
            cosmo_cfg_mat = dataclasses.replace(
                cosmo_cfg,
                history_config=get_history_config("cosmo-wan-bottleneck-modul.yaml"))

            base_spec, _ = base_cfg.inputs_spec(batch_size=1)

            def _materialize(spec):
                if spec is None:
                    return None
                if spec.dtype == bool or jnp.issubdtype(spec.dtype, jnp.integer):
                    return jnp.zeros(spec.shape, spec.dtype)
                return jnp.asarray(np.random.randn(*spec.shape), spec.dtype)

            base = jax.tree.map(_materialize, base_spec,
                                is_leaf=lambda x: x is None or hasattr(x, "dtype"))
            base_obs = HistAugObservation.from_base_obs(base)

            cbm = cosmo_cfg_mat.history_config.cosmo_bottleneck_memory
            recent_len = (cosmo_cfg_mat.history_config.streaming_obs_horizon
                          * cosmo_cfg_mat.history_config.num_views)
            base_obs_spec_for_cosmo, _ = base_cfg.inputs_spec(batch_size=1)

            def _build_cosmo_obs(n_cosmo):
                base2 = jax.tree.map(_materialize, base_obs_spec_for_cosmo,
                                     is_leaf=lambda x: x is None or hasattr(x, "dtype"))
                return HistAugObservation.from_base_obs(
                    base2,
                    cosmo_latent_emb=jnp.asarray(np.random.randn(1, n_cosmo, int(cbm.cosmos_latent_dim)), jnp.float32),
                    cosmo_recent_emb=jnp.asarray(np.random.randn(1, recent_len, int(cbm.siglip_dim)), jnp.float32),
                    cosmo_mem_mask=jnp.ones((1, n_cosmo), bool),
                    cosmo_action_emb=jnp.asarray(np.random.randn(1, recent_len, int(cbm.action_in_dim)), jnp.float32),
                    cosmo_proprio_emb=jnp.asarray(np.random.randn(1, recent_len, int(cbm.proprio_in_dim)), jnp.float32),
                    cosmo_lang_emb=jnp.asarray(np.random.randn(1, int(cbm.lang_emb_dim)), jnp.float32),
                )
            cosmo_obs_builder = _build_cosmo_obs

            # measure pi05 baseline once
            def _pi05():
                jax.block_until_ready(baseline_sample(sample_key, base_obs))
            pi05_med, pi05_p90 = _time_loop(_pi05, iters, warmup)

            # measure cosmo-modulation full sample_actions once (V-independent: 256 mem
            # tokens). modulation seam = full - baseline - bottleneck(that config).
            _co = cosmo_obs_builder(SPT)  # small KV; seam cost is KV-length-independent

            def _cosmo_full():
                jax.block_until_ready(cosmo_sample(sample_key, _co))
            cf_med, cf_p90 = _time_loop(_cosmo_full, iters, warmup)

            # bottleneck cost for THIS small cosmo_obs (to subtract, avoid double count)
            _c = jnp.asarray(np.random.randn(1, SPT, LDIM), jnp.bfloat16)
            _r = jnp.asarray(np.random.randn(1, recent_len, 2048), jnp.bfloat16)
            _a = jnp.asarray(np.random.randn(1, recent_len, 24), jnp.bfloat16)
            _p = jnp.asarray(np.random.randn(1, recent_len, 32), jnp.bfloat16)
            _l = jnp.asarray(np.random.randn(1, 2048), jnp.bfloat16)
            _mk = jnp.ones((1, SPT), bool)

            def _b0():
                jax.block_until_ready(bneck_fwd(bneck, _c, _r, _a, _p, _mk, _l))
            b0_med, b0_p90 = _time_loop(_b0, iters, warmup)
            mod_med = max(cf_med - pi05_med - b0_med, 0.0)
            mod_p90 = max(cf_p90 - pi05_p90 - b0_p90, 0.0)
            pi05_mode = "real"
        except Exception as e:  # noqa: BLE001
            pi05_mode = f"unavailable ({type(e).__name__}: {e}); pi05/modulation = CPU stand-in"
            _tiny = nnx.jit(lambda w, x: jnp.tanh(x @ w))
            _tw = jnp.asarray(np.random.randn(64, 64), jnp.float32)
            _tx = jnp.asarray(np.random.randn(1, 64), jnp.float32)

            def _stand():
                jax.block_until_ready(_tiny(_tw, _tx))
            pi05_med, pi05_p90 = _time_loop(_stand, iters, warmup)
            mod_med, mod_p90 = _time_loop(_stand, iters, warmup)
    else:
        pi05_mode = "fake (CPU stand-in)"
        _tiny = nnx.jit(lambda w, x: jnp.tanh(x @ w))
        _tw = jnp.asarray(np.random.randn(64, 64), jnp.float32)
        _tx = jnp.asarray(np.random.randn(1, 64), jnp.float32)

        def _stand():
            jax.block_until_ready(_tiny(_tw, _tx))
        pi05_med, pi05_p90 = _time_loop(_stand, iters, warmup)
        mod_med, mod_p90 = _time_loop(_stand, iters, warmup)

    report = {}
    for V in views:
        for L in buffer_lengths:
            key = f"V{V}_L{L}"
            # downsample already applied to L; num_chunks ~= L // temporal_factor
            num_chunks = max(1, L // TF)
            n_video_tokens = num_chunks * V * SPT     # bottleneck video KV length
            recent_len = RECENT_FRAMES * V            # recent per-view window

            # ---- bottleneck (real, both modes) on the (V,L) KV ----
            cosmo = jnp.asarray(np.random.randn(1, n_video_tokens, LDIM), jnp.bfloat16)
            recent = jnp.asarray(np.random.randn(1, recent_len * SIGLIP_TOKENS_PER_FRAME, 2048), jnp.bfloat16)
            act = jnp.asarray(np.random.randn(1, recent_len, 24), jnp.bfloat16)
            prop = jnp.asarray(np.random.randn(1, recent_len, 32), jnp.bfloat16)
            lang = jnp.asarray(np.random.randn(1, 2048), jnp.bfloat16)
            mask = jnp.ones((1, n_video_tokens), bool)

            def _bneck():
                jax.block_until_ready(bneck_fwd(bneck, cosmo, recent, act, prop, mask, lang))
            b_med, b_p90 = _time_loop(_bneck, iters, warmup)

            row = {"bottleneck_ms": b_med, "bottleneck_ms_p90": b_p90}

            if not use_fake:
                import torch
                # ---- wan_encode: ONE chunk-fire (temporal_factor frames, V views) ----
                def _wan():
                    clip = np.random.randint(0, 256, (TF, V, 128, 128, 3), np.uint8)
                    wan_fe.encode_multiview(clip)
                    torch.cuda.synchronize()
                w_med, w_p90 = _time_loop(_wan, iters, warmup)
                row["wan_encode_ms_perfire"] = w_med
                row["wan_encode_ms_perfire_p90"] = w_p90
                row["wan_encode_ms_amortized"] = w_med / TF

                # ---- torch_to_jax: one chunk-fire's Wan latents (V*64 tokens) ----
                _lat = np.asarray(
                    wan_fe.encode_multiview(
                        np.random.randint(0, 256, (TF, V, 128, 128, 3), np.uint8)),
                    np.float32)

                def _t2j():
                    x = jax.device_put(jnp.asarray(_lat))
                    jax.block_until_ready(x)
                t_med, t_p90 = _time_loop(_t2j, iters, warmup)
                row["torch_to_jax_ms"] = t_med
                row["torch_to_jax_ms_p90"] = t_p90
            else:
                # CPU stand-ins for the torch-side ops (fast smoke).
                def _wan_fake():
                    _ = np.random.randn(TF, V * SPT, LDIM).astype(np.float32)
                w_med, w_p90 = _time_loop(_wan_fake, iters, warmup)
                row["wan_encode_ms_perfire"] = w_med
                row["wan_encode_ms_perfire_p90"] = w_p90
                row["wan_encode_ms_amortized"] = w_med / TF
                _lat = np.random.randn(1, V * SPT, LDIM).astype(np.float32)

                def _t2j():
                    x = jax.device_put(jnp.asarray(_lat))
                    jax.block_until_ready(x)
                t_med, t_p90 = _time_loop(_t2j, iters, warmup)
                row["torch_to_jax_ms"] = t_med
                row["torch_to_jax_ms_p90"] = t_p90

            row["modulation_ms"] = mod_med
            row["modulation_ms_p90"] = mod_p90
            row["pi05_forward_ms"] = pi05_med
            row["pi05_forward_ms_p90"] = pi05_p90

            total = (row["wan_encode_ms_amortized"]
                     + row["torch_to_jax_ms"] / TF
                     + row["bottleneck_ms"]
                     + row["modulation_ms"]
                     + row["pi05_forward_ms"])
            row["total_optionD_ms"] = total
            row["overhead_ms"] = total - row["pi05_forward_ms"]

            row["_meta"] = {"V": V, "L": L, "num_chunks": num_chunks,
                            "n_video_tokens": n_video_tokens, "recent_len": recent_len,
                            "temporal_factor": TF, "spatial_tokens_per_view": SPT,
                            "latent_dim": LDIM}
            report[key] = row

    report["_notes"] = {
        "pi05_modulation": ("pi05_forward_ms and modulation_ms are V,L-INDEPENDENT: "
                            "the bottleneck compresses any V*L video KV down to a fixed "
                            "256-token memory summary before the pi0.5 LLM seam. Measured "
                            "ONCE via the real baseline + cosmo-modulation sample_actions "
                            "(same path as scripts/bench_latency.py) and reused for every "
                            "(V,L) row. Mode: " + str(pi05_mode)),
        "wan_encode": ("real WanFrontend.encode_multiview on (temporal_factor, V, 128,128,3) "
                       "uint8; V views encoded independently by the frozen VAE then "
                       "concatenated on the token axis. Fires once per temporal_factor "
                       "steps; amortized = perfire / temporal_factor."),
        "torch_to_jax": "device_put of one chunk-fire's V*64 Wan latents; amortized in total by /temporal_factor.",
        "bottleneck": ("real CosmoBottleneckMemory forward; video KV = num_chunks*V*64 tokens "
                       "(grows with V*L), output fixed 256 tokens. jitted, block_until_ready."),
        "total_optionD_ms": "wan_encode_amortized + torch_to_jax_amortized + bottleneck + modulation + pi05_forward",
        "overhead_ms": "total_optionD - pi05_forward (headline extra per-step latency)",
    }
    return report


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/wan_multiview_latency.json")
    ap.add_argument("--views", default="1,2,3")
    ap.add_argument("--lengths", default="64,256,1024,2048")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    views = [int(x) for x in args.views.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    rep = run_benchmark(views=views, buffer_lengths=lengths, iters=args.iters,
                        warmup=args.warmup, use_fake=args.fake)

    try:
        dev = jax.devices()[0]
        stats = dev.memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        if peak is not None:
            rep["_peak_gpu_mib"] = peak / (1024 ** 2)
    except Exception as e:  # noqa: BLE001
        rep["_peak_gpu_mib_error"] = str(e)

    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
