#!/usr/bin/env python3
"""Recompute align_prefix as a 3-view target: (T, 64, 2048) -> (T, 192, 2048).

Phase B / TASK B1. The stores' recent_siglip and latents_v* are ALREADY 3-view;
only align_prefix was 1-view (front view only). This script recomputes
align_prefix from the existing frames.npy (T, 3, 256, 256, 3) by running the
checkpoint's PaliGemma prefix on ALL 3 views and stacking view-major:

    align3[:, 0:64]   = view0 (front, == old 1-view align_prefix)
    align3[:, 64:128] = view1
    align3[:, 128:192]= view2

The decoder + loss read `align_num_patches` (config), so no code change is
needed there; only the config value 64 -> 192 (TASK B2).

Reuses build_wan_store._load_model / _load_prefix_fn (front-view prefix_fn
takes (t,H,W,3) uint8 -> (t,64,2048)). Runs per-view (3 calls) for correctness.

The old align_prefix.npy is backed up to align_prefix_1view.npy BEFORE the new
file is written (skipped if a backup already exists, so re-runs are idempotent).
"""
import argparse
import os
import sys

import numpy as np


def _episode_ids(store, episodes):
    feat = os.path.join(store, "features")
    ids = []
    for name in os.listdir(feat):
        if name.startswith("episode_"):
            try:
                ids.append(int(name.split("_", 1)[1]))
            except ValueError:
                pass
    ids.sort()
    if episodes:
        want = set(int(x) for x in episodes)
        ids = [i for i in ids if i in want]
    return ids


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, help="Store dir (has features/episode_N).")
    p.add_argument("--ckpt", required=True, help="Checkpoint dir (has params/).")
    p.add_argument("--config", default="pi05_baseline",
                   help="MME TrainConfig name (pi05_baseline == plain pi0.5) whose model matches --ckpt.")
    p.add_argument("--shard", type=int, default=0, help="This shard index [0, nshard).")
    p.add_argument("--nshard", type=int, default=1, help="Total number of shards.")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Optional explicit episode ids (else all).")
    p.add_argument("--dry-run-ep0", action="store_true",
                   help="Only process episode 0, do NOT overwrite; verify view0==old.")
    args = p.parse_args(argv)

    # Load a PLAIN pi0.5 model (openpi Pi0Config) directly from the checkpoint
    # params. This matches the original ds1 store build: build_wan_store's
    # _load_prefix_fn unpacks embed_prefix() as a 3-tuple, which only the plain
    # openpi Pi0 provides (the MME HistoryPi0.embed_prefix returns a 5-tuple and
    # type-hints HistAugObservation, so it is incompatible with _load_prefix_fn).
    # Both checkpoints (finetuned + pi05_base) are plain pi0.5 param trees, so
    # loading into Pi0Config(pi05=True) is byte-consistent with how the original
    # 1-view align_prefix (view0) was produced -> the view0==old check verifies it.
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0_config import Pi0Config
    from mme_vla_suite.dataset_builder.build_wan_store import _load_prefix_fn

    print(f"[rebuild3v] loading PLAIN pi0.5 model ckpt={args.ckpt}", flush=True)
    cfg = Pi0Config(pi05=True, action_horizon=20)
    model = cfg.load(
        _model.restore_params(os.path.join(args.ckpt, "params"), dtype=jnp.bfloat16)
    )
    prefix_fn = _load_prefix_fn(model)

    ids = _episode_ids(args.store, args.episodes)
    if args.dry_run_ep0:
        ids = [0]
    mine = [e for i, e in enumerate(ids) if (i % args.nshard) == args.shard]
    print(f"[rebuild3v] shard {args.shard}/{args.nshard}: {len(mine)} episodes: {mine[:5]}...", flush=True)

    for ep in mine:
        d = os.path.join(args.store, "features", f"episode_{ep}")
        fpath = os.path.join(d, "frames.npy")
        apath = os.path.join(d, "align_prefix.npy")
        bkp = os.path.join(d, "align_prefix_1view.npy")
        if not os.path.exists(fpath):
            print(f"[rebuild3v] ep{ep}: NO frames.npy, skip", flush=True)
            continue
        frames = np.load(fpath, mmap_mode="r")  # (T,3,H,W,3) uint8
        assert frames.ndim == 5 and frames.shape[1] == 3 and frames.shape[-1] == 3, \
            f"ep{ep} frames shape {frames.shape}"
        T = frames.shape[0]

        old = np.load(apath, mmap_mode="r") if os.path.exists(apath) else None
        if old is not None:
            assert old.shape[0] == T, f"ep{ep}: align T {old.shape[0]} != frames T {T}"
            if old.shape[1] == 192:
                print(f"[rebuild3v] ep{ep}: already 3-view {old.shape}, skip", flush=True)
                continue

        # per-view prefix -> (T,64,2048) each; view-major concat -> (T,192,2048)
        pv = []
        for v in range(3):
            fv = np.ascontiguousarray(frames[:, v])  # (T,H,W,3) uint8
            pv.append(prefix_fn(fv))                  # (T,64,2048) float32
        align3 = np.concatenate(pv, axis=1).astype(np.float32)  # (T,192,2048)
        assert align3.shape == (T, 192, 2048), f"ep{ep}: got {align3.shape}"

        if args.dry_run_ep0:
            if old is not None and old.shape[1] == 64:
                v0 = align3[:, :64]
                ok = np.allclose(np.asarray(old, np.float32), v0, rtol=1e-2, atol=1e-2)
                mad = float(np.max(np.abs(np.asarray(old, np.float32) - v0)))
                print(f"[rebuild3v] DRYRUN ep0: new {align3.shape}; view0 matches old "
                      f"(rtol1e-2/atol1e-2)={ok}; max_abs_diff={mad:.4g}", flush=True)
            else:
                print(f"[rebuild3v] DRYRUN ep0: new {align3.shape}; no 1-view old to compare", flush=True)
            return 0

        # back up old 1-view (idempotent) then write new
        if old is not None and not os.path.exists(bkp):
            os.rename(apath, bkp)
        tmp = apath + ".tmp.npy"
        np.save(tmp, align3)
        os.replace(tmp, apath)
        print(f"[rebuild3v] ep{ep}: wrote {align3.shape} (backup={os.path.exists(bkp)})", flush=True)

    print(f"[rebuild3v] shard {args.shard}/{args.nshard} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
