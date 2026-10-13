"""Component-wise latency benchmark: baseline pi0.5 vs Option-D memory overhead.

Measures the extra *synchronous* per-policy-step latency that the Option-D
(Cosmos-bottleneck) memory path adds on top of a baseline pi0.5 forward pass.

Components (all milliseconds, median over >=30 timed iters after warmup):
  cosmos_encode_ms_perfire   : encode one chunk_frames-frame clip through the
                               frozen Cosmos CV tokenizer (torch, GPU). Fires
                               once every chunk_frames steps.
  cosmos_encode_ms_amortized : perfire / chunk_frames (per-step amortized).
  cosmos_encode_ms           : == amortized (smoke-test contract key).
  siglip_recent_ms           : jitted SigLIP vision_encode on the recent
                               window (<= chunk_frames frames). Runs every step.
  torch_to_jax_ms            : in-process torch->jax handoff of one chunk's
                               Cosmos latents (device_put). Fires once per
                               chunk-fire; amortized /chunk_frames for the total.
  bottleneck_ms              : the 256-token Perceiver forward (embed_memory),
                               KV grows with history L, output is fixed 256.
  modulation_ms              : extra sample_actions cost of feeding the 256
                               memory tokens as modulation vs the baseline.
  pi05_forward_ms            : baseline pi0.5 sample_actions (no memory).
  total_optionD_ms           : cosmos_amortized + siglip + torch_to_jax_amortized
                               + bottleneck + modulation + pi05_forward.
  overhead_ms                : total_optionD - pi05_forward (the headline).

Timing discipline:
  * every timed jax result is jax.block_until_ready'd before stopping the timer.
  * every timed torch CUDA op is torch.cuda.synchronize'd before stopping.
  * first `warmup` (>=3 for real runs) iters are discarded (JIT/kernel warmup).
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

# Allow `from tests.cosmo import shapes` and package imports when run as a plain
# script (python scripts/bench_latency.py). The package root is scripts/.. .
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
    """Run fn() warmup times (discarded) then iters timed; return (median_ms, p90_ms).

    fn must perform its own block_until_ready / cuda.synchronize before returning
    so the wall-clock delta reflects completed work.
    """
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        s = _t()
        fn()
        ts.append((_t() - s) * 1e3)
    return _stats(ts)


def run_benchmark(buffer_lengths, iters=30, warmup=3, use_fake_cosmos=False, device_gpu=None):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from omegaconf import OmegaConf

    from mme_vla_suite.models.representation.cosmo_mem import CosmoBottleneckMemory
    from tests.cosmo import shapes as S

    CHUNK = S.COSMOS_TEMPORAL_FACTOR  # frames per consolidated chunk (8)

    # ---- Bottleneck config (mirrors models/config/robomme/cosmo-bottleneck-modul.yaml) ----
    cfg = OmegaConf.create({
        "budget": 256, "memory_token_dim": 1024,
        "cosmo_bottleneck_memory": {
            "num_latents": 256, "num_cross_blocks": 2, "num_heads": 8,
            "cosmos_latent_dim": 16, "siglip_dim": 2048, "proprio_in_dim": 32,
            "action_in_dim": 24, "lang_emb_dim": 2048,
            "action_tokenizer_hidden": 512, "proprio_tokenizer_hidden": 512}})
    bneck = CosmoBottleneckMemory(config=cfg, rngs=nnx.Rngs(0), dtype=jnp.bfloat16)
    bneck_fwd = nnx.jit(lambda m, a, b, c, d, e, f: m(a, b, c, d, e, f))

    # ---------------------------------------------------------------------------
    # Real (GPU) components: built once, reused across all L.
    # ---------------------------------------------------------------------------
    cosmo_fe = None
    siglip_encode = None          # jitted vision_encode: (k, v, 224,224,3) -> (k,v,p,d)
    baseline_sample = None        # baseline pi0.5 sample_actions (no memory)
    cosmo_model = None            # HistoryPi0 cosmo-modulation model
    cosmo_sample = None           # cosmo-modulation sample_actions
    base_obs_builder = None       # fn(batch) -> baseline observation
    cosmo_obs_builder = None      # fn(L) -> cosmo HistAugObservation for history-length L
    sample_key = None

    if not use_fake_cosmos:
        import torch
        from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend
        from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
        from mme_vla_suite.models.integration.history_observation import HistAugObservation
        import openpi.shared.nnx_utils as nnx_utils

        sample_key = jax.random.key(0)

        # 1) Cosmos frontend (frozen causal CV tokenizer) on GPU.
        cosmo_fe = CosmoFrontend(variant="CV8x8x8", device="cuda")

        # 2) Baseline pi0.5 (NO memory) and cosmo-modulation model, apples-to-apples.
        AH = 20  # action_horizon used by the mme_vla_suite train configs
        base_cfg = HistoryPi0Config(pi05=True, action_horizon=AH, use_history=False,
                                    history_config=None, discrete_state_input=False)
        base_model = base_cfg.create(jax.random.key(0))
        baseline_sample = nnx_utils.module_jit(base_model.sample_actions)

        cosmo_cfg = HistoryPi0Config(pi05=True, action_horizon=AH, use_history=True,
                                     history_config="cosmo-bottleneck-modul.yaml",
                                     discrete_state_input=False)
        cosmo_model = cosmo_cfg.create(jax.random.key(0))
        cosmo_sample = nnx_utils.module_jit(cosmo_model.sample_actions)
        siglip_encode = nnx_utils.module_jit(cosmo_model.vision_encode)

        # Resolve the *materialized* HistoryPi0Config so inputs_spec has the loaded
        # history DictConfig (create() replaces the string with the DictConfig).
        import dataclasses
        from mme_vla_suite.models.config.utils import get_history_config
        cosmo_cfg_mat = dataclasses.replace(
            cosmo_cfg, history_config=get_history_config("cosmo-bottleneck-modul.yaml"))

        base_spec, _ = base_cfg.inputs_spec(batch_size=1)

        def _materialize(spec):
            if spec is None:
                return None
            if spec.dtype == bool:
                return jnp.zeros(spec.shape, spec.dtype)
            if jnp.issubdtype(spec.dtype, jnp.integer):
                return jnp.zeros(spec.shape, spec.dtype)
            return jnp.asarray(np.random.randn(*spec.shape), spec.dtype)

        def _build_base_obs():
            base = jax.tree.map(_materialize, base_spec,
                                is_leaf=lambda x: x is None or hasattr(x, "dtype"))
            # baseline model is a HistoryPi0(use_history=False); its sample_actions
            # still calls HistAugObservation.preprocess -> to_base_obs, so wrap it.
            return HistAugObservation.from_base_obs(base)

        base_obs_builder = _build_base_obs

        # Build a cosmo HistAugObservation whose consolidated-latent length scales
        # with history length L (this is exactly what CosmoMemoryBuffer feeds:
        # each finalized chunk contributes COSMOS_SPATIAL_TOKENS latent tokens).
        cbm = cosmo_cfg_mat.history_config.cosmo_bottleneck_memory
        recent_len = (cosmo_cfg_mat.history_config.streaming_obs_horizon
                      * cosmo_cfg_mat.history_config.num_views)
        base_obs_spec_for_cosmo, _ = base_cfg.inputs_spec(batch_size=1)

        def _build_cosmo_obs(L):
            Lc = max(0, L // CHUNK)
            n_cosmo = max(Lc, 1) * S.COSMOS_SPATIAL_TOKENS
            base = jax.tree.map(_materialize, base_obs_spec_for_cosmo,
                                is_leaf=lambda x: x is None or hasattr(x, "dtype"))
            return HistAugObservation.from_base_obs(
                base,
                cosmo_latent_emb=jnp.asarray(np.random.randn(1, n_cosmo, int(cbm.cosmos_latent_dim)), jnp.float32),
                cosmo_recent_emb=jnp.asarray(np.random.randn(1, recent_len, int(cbm.siglip_dim)), jnp.float32),
                cosmo_mem_mask=jnp.ones((1, n_cosmo), bool),
                cosmo_action_emb=jnp.asarray(np.random.randn(1, recent_len, int(cbm.action_in_dim)), jnp.float32),
                cosmo_proprio_emb=jnp.asarray(np.random.randn(1, recent_len, int(cbm.proprio_in_dim)), jnp.float32),
                cosmo_lang_emb=jnp.asarray(np.random.randn(1, int(cbm.lang_emb_dim)), jnp.float32),
            )

        cosmo_obs_builder = _build_cosmo_obs
    else:
        # Light CPU stand-ins so the smoke test is fast and requires no GPU/weights.
        # A tiny jitted matmul stands in for pi05/modulation forwards; a tiny jitted
        # conv-like op stands in for SigLIP; numpy random stands in for Cosmos encode.
        _tiny = nnx.jit(lambda w, x: jnp.tanh(x @ w))
        _tiny_w = jnp.asarray(np.random.randn(64, 64), jnp.float32)
        _tiny_x_base = jnp.asarray(np.random.randn(1, 64), jnp.float32)
        _tiny_x_mod = jnp.asarray(np.random.randn(1, 64), jnp.float32)

    report = {}
    for L in buffer_lengths:
        Lc = max(0, L // CHUNK)                              # finalized consolidated chunks
        n_cosmo_tokens = max(Lc, 1) * S.COSMOS_SPATIAL_TOKENS  # latent tokens fed to bottleneck
        Lr = (L % CHUNK) or 1                                # recent frames (>=1)

        # ---- bottleneck (real, both modes) ----
        cosmo = jnp.asarray(np.random.randn(1, n_cosmo_tokens, 16), jnp.bfloat16)
        recent = jnp.asarray(np.random.randn(1, Lr * S.SIGLIP_TOKENS_PER_FRAME, 2048), jnp.bfloat16)
        act = jnp.asarray(np.random.randn(1, 16, 24), jnp.bfloat16)
        prop = jnp.asarray(np.random.randn(1, 16, 32), jnp.bfloat16)
        lang = jnp.asarray(np.random.randn(1, 2048), jnp.bfloat16)
        mask = jnp.ones((1, n_cosmo_tokens), bool)

        def _bneck():
            jax.block_until_ready(bneck_fwd(bneck, cosmo, recent, act, prop, mask, lang))

        med, p90 = _time_loop(_bneck, iters, warmup)
        row = {"bottleneck_ms": med, "bottleneck_ms_p90": p90}
        notes = {}

        if not use_fake_cosmos:
            import torch

            # ---- cosmos_encode: one chunk_frames-frame clip through Cosmos ----
            # frontend caches identical inputs, so feed a fresh random clip each call.
            def _cosmos():
                clip = np.random.randint(0, 256, (CHUNK, 224, 224, 3), np.uint8)
                cosmo_fe.encode(clip)
                torch.cuda.synchronize()

            c_med, c_p90 = _time_loop(_cosmos, iters, warmup)
            row["cosmos_encode_ms_perfire"] = c_med
            row["cosmos_encode_ms_perfire_p90"] = c_p90
            row["cosmos_encode_ms_amortized"] = c_med / CHUNK
            row["cosmos_encode_ms"] = c_med / CHUNK  # smoke-contract alias

            # ---- torch_to_jax: move one chunk's Cosmos latents into a jnp device array ----
            _lat_np = np.asarray(
                cosmo_fe.encode(np.random.randint(0, 256, (CHUNK, 224, 224, 3), np.uint8)),
                np.float32)

            def _t2j():
                x = jax.device_put(jnp.asarray(_lat_np))
                jax.block_until_ready(x)

            t_med, t_p90 = _time_loop(_t2j, iters, warmup)
            row["torch_to_jax_ms"] = t_med
            row["torch_to_jax_ms_p90"] = t_p90

            # ---- siglip_recent: jitted SigLIP vision_encode on Lr recent frames ----
            recent_frames = jnp.asarray(
                np.random.randn(Lr, 1, 224, 224, 3), jnp.float32)  # (k=frames, v=1, ...)

            def _siglip():
                jax.block_until_ready(siglip_encode(recent_frames))

            s_med, s_p90 = _time_loop(_siglip, iters, warmup)
            row["siglip_recent_ms"] = s_med
            row["siglip_recent_ms_p90"] = s_p90

            # ---- pi05_forward: BASELINE pi0.5 sample_actions (no memory) ----
            base_obs = base_obs_builder()

            def _pi05():
                jax.block_until_ready(baseline_sample(sample_key, base_obs))

            p_med, p_p90 = _time_loop(_pi05, iters, warmup)
            row["pi05_forward_ms"] = p_med
            row["pi05_forward_ms_p90"] = p_p90

            # ---- modulation: cosmo-modulation sample_actions MINUS baseline ----
            # This full forward includes embed_memory (the bottleneck) + the LLM
            # modulation seam. To isolate the *modulation seam* cost we subtract the
            # bottleneck (already measured) as well, so the reported components sum
            # without double counting. See notes.
            cosmo_obs = cosmo_obs_builder(L)

            def _cosmo_full():
                jax.block_until_ready(cosmo_sample(sample_key, cosmo_obs))

            cf_med, cf_p90 = _time_loop(_cosmo_full, iters, warmup)
            # modulation seam = full cosmo sample_actions - baseline - bottleneck
            mod = cf_med - p_med - med
            row["modulation_ms"] = max(mod, 0.0)
            row["modulation_ms_p90"] = max(cf_p90 - p_p90 - p90, 0.0)
            row["cosmo_sample_full_ms"] = cf_med  # full cosmo sample_actions (diagnostic)

            notes = {
                "pi05_forward_ms": "baseline HistoryPi0(pi05=True, use_history=False) "
                                   "jitted sample_actions on a fake batch=1 observation.",
                "modulation_ms": "cosmo-modulation sample_actions (embed_memory+LLM "
                                 "modulation) minus baseline sample_actions minus "
                                 "bottleneck_ms, to avoid double-counting bottleneck.",
                "cosmos_encode_ms": "amortized = perfire/chunk_frames; fires once per "
                                    f"{CHUNK} steps.",
                "torch_to_jax_ms": "per chunk-fire; amortized in total by /chunk_frames.",
                "siglip_recent_ms": "per-step (recent window re-encoded every step).",
                "per_step_vs_amortized": "total uses cosmos & torch_to_jax AMORTIZED "
                                         "(/chunk_frames); siglip, bottleneck, "
                                         "modulation, pi05 are per-step.",
            }

            total = (row["cosmos_encode_ms_amortized"] + row["siglip_recent_ms"]
                     + row["torch_to_jax_ms"] / CHUNK + row["bottleneck_ms"]
                     + row["modulation_ms"] + row["pi05_forward_ms"])
            row["total_optionD_ms"] = total
            row["overhead_ms"] = total - row["pi05_forward_ms"]

        else:
            # ---- Light CPU stand-ins (smoke) ----
            def _stand_cosmos():
                _ = np.random.randn(CHUNK, S.COSMOS_SPATIAL_TOKENS, 16).astype(np.float32)

            c_med, c_p90 = _time_loop(_stand_cosmos, iters, warmup)
            row["cosmos_encode_ms_perfire"] = c_med
            row["cosmos_encode_ms_perfire_p90"] = c_p90
            row["cosmos_encode_ms_amortized"] = c_med / CHUNK
            row["cosmos_encode_ms"] = c_med / CHUNK

            _lat_np = np.random.randn(1, S.COSMOS_SPATIAL_TOKENS, 16).astype(np.float32)

            def _t2j():
                x = jax.device_put(jnp.asarray(_lat_np))
                jax.block_until_ready(x)

            t_med, t_p90 = _time_loop(_t2j, iters, warmup)
            row["torch_to_jax_ms"] = t_med
            row["torch_to_jax_ms_p90"] = t_p90

            def _stand_siglip():
                jax.block_until_ready(_tiny(_tiny_w, _tiny_x_base))

            s_med, s_p90 = _time_loop(_stand_siglip, iters, warmup)
            row["siglip_recent_ms"] = s_med
            row["siglip_recent_ms_p90"] = s_p90

            def _stand_pi05():
                jax.block_until_ready(_tiny(_tiny_w, _tiny_x_base))

            p_med, p_p90 = _time_loop(_stand_pi05, iters, warmup)
            row["pi05_forward_ms"] = p_med
            row["pi05_forward_ms_p90"] = p_p90

            def _stand_mod():
                jax.block_until_ready(_tiny(_tiny_w, _tiny_x_mod))

            m_med, m_p90 = _time_loop(_stand_mod, iters, warmup)
            row["modulation_ms"] = m_med
            row["modulation_ms_p90"] = m_p90

            notes = {"mode": "use_fake_cosmos=True: CPU stand-ins for cosmos/siglip/"
                             "pi05/modulation; bottleneck is the REAL Perceiver. "
                             "Numbers are placeholders to exercise the contract."}

            total = (row["cosmos_encode_ms_amortized"] + row["siglip_recent_ms"]
                     + row["torch_to_jax_ms"] / CHUNK + row["bottleneck_ms"]
                     + row["modulation_ms"] + row["pi05_forward_ms"])
            row["total_optionD_ms"] = total
            row["overhead_ms"] = total - row["pi05_forward_ms"]

        row["_meta"] = {"L": L, "consolidated_chunks": Lc,
                        "n_cosmo_latent_tokens": n_cosmo_tokens,
                        "recent_frames": Lr, "chunk_frames": CHUNK}
        row["_notes"] = notes
        report[L] = row

    return report


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/cosmo_latency_report.json")
    ap.add_argument("--lengths", default="1,64,256,683,1024,2048")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fake", action="store_true", help="use CPU stand-ins")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    rep = run_benchmark(lengths, iters=args.iters, warmup=args.warmup,
                        use_fake_cosmos=args.fake)

    # Peak GPU memory of the Option-D path (bytes -> MiB), if on GPU.
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
