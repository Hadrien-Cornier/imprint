#!/usr/bin/env python3
"""Assemble washreturn_store_var8_base (Phase B / TASK B3.1).

var8_base = var8's latent/variant streams (Wan latents are checkpoint-agnostic)
            + ds1_base's checkpoint-specific streams (recent_siglip, align_prefix
            [NOW 3-view], lang_emb, action_proprio, reward, frames, chunk_index).

Every feature file is a symlink resolved to the REAL underlying file (realpath),
so downstream loaders that follow symlinks see real .npy/.json. Idempotent:
re-links each run.

MUST be run AFTER TASK B1 finishes for ds1_base, so align_prefix is (T,192,2048).
"""
import argparse
import json
import os

# streams sourced from var8 (variant / Wan-latent, checkpoint-agnostic)
VAR8_STREAMS = ("variant_index.json",)  # + latents_v*.npy discovered dynamically
# streams sourced from ds1_base (checkpoint-specific + shared frames/index)
BASE_STREAMS = (
    "recent_siglip.npy",
    "align_prefix.npy",
    "action_proprio.npy",
    "reward.npy",
    "frames.npy",
    "chunk_index.json",
    "lang_emb.npy",
)


def _link(src, dst):
    src = os.path.realpath(src)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--var8", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    var8_feat = os.path.join(args.var8, "features")
    base_feat = os.path.join(args.base, "features")
    out_feat = os.path.join(args.out, "features")
    os.makedirs(out_feat, exist_ok=True)

    eps = sorted(
        int(n.split("_", 1)[1]) for n in os.listdir(var8_feat)
        if n.startswith("episode_")
    )
    print(f"[assemble] {len(eps)} episodes")
    n_done = 0
    for ep in eps:
        v8d = os.path.join(var8_feat, f"episode_{ep}")
        bsd = os.path.join(base_feat, f"episode_{ep}")
        outd = os.path.join(out_feat, f"episode_{ep}")
        os.makedirs(outd, exist_ok=True)

        # var8 latents (v0..vN) + variant_index
        lat = sorted(f for f in os.listdir(v8d) if f.startswith("latents_v") and f.endswith(".npy"))
        assert lat, f"ep{ep}: no latents_v* in var8"
        for f in lat:
            _link(os.path.join(v8d, f), os.path.join(outd, f))
        for f in VAR8_STREAMS:
            s = os.path.join(v8d, f)
            if os.path.exists(s):
                _link(s, os.path.join(outd, f))

        # ds1_base checkpoint-specific streams
        for f in BASE_STREAMS:
            s = os.path.join(bsd, f)
            if os.path.exists(s):
                _link(s, os.path.join(outd, f))
            else:
                print(f"[assemble] WARN ep{ep}: base missing {f}")
        n_done += 1
    print(f"[assemble] linked {n_done} episode dirs into {out_feat}")

    # copy meta/samples.json + stats.json from var8 (variant sample list is what
    # the offline Stage-1 loader iterates; base uses the same variant sampling).
    out_meta = os.path.join(args.out, "meta")
    os.makedirs(out_meta, exist_ok=True)
    for f in ("samples.json", "stats.json"):
        s = os.path.join(args.var8, "meta", f)
        if os.path.exists(s):
            _link(s, os.path.join(out_meta, f))
    print("[assemble] DONE")


if __name__ == "__main__":
    main()
