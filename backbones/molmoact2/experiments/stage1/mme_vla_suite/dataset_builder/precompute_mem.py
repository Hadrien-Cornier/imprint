"""Precompute frozen-bottleneck memory tokens per (sample, variant).

Replicates the Stage-2 offline gather (dataset.py._gather_cosmo_bottleneck offline branch:
MAX_RECENT=2048, MAX_STEPS=800, LATENT_CAP=16384, pool_hw) for a SPECIFIC variant, runs the
frozen Stage-1 bottleneck, and saves (256, mem_dim) fp16 per (sample_id, variant).

Output: <out>/sample_{sid}.npz  {v{k}: (256,1024) fp16 for each causal variant}
Loading these at Stage-2 replaces ~18.5MB/sample of cosmo inputs with ~0.5MB tokens.
"""
import argparse, json, os
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples", type=int, default=0)   # 0 = all; >0 = trial subset
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    args = p.parse_args(argv)

    import jax, jax.numpy as jnp, flax.nnx as nnx
    import flax.traverse_util as trav
    from omegaconf import OmegaConf
    from mme_vla_suite.training.pretrain_loss import make_stage1_modules

    cfg = OmegaConf.load(args.config)
    cbm = cfg.cosmo_bottleneck_memory
    n_variants = int(getattr(cbm, "n_variants", 8))
    POOL_HW = int(getattr(cbm, "pool_hw", 0)) or None
    act_enc = str(getattr(cbm, "action_encoding", "raw_delta"))
    pro_enc = str(getattr(cbm, "proprio_encoding", "raw_delta"))
    lang_dim = int(getattr(cbm, "lang_emb_dim", 2048))
    MAX_RECENT, MAX_STEPS, LATENT_CAP = 2048, 800, 16384

    # --- load frozen bottleneck (overlay Stage-1 bottleneck/ weights) ---
    bneck = make_stage1_modules(cfg, seed=0)["bottleneck"]
    gd, state = nnx.split(bneck)
    flat = trav.flatten_dict(state.to_pure_dict())
    s1 = np.load(args.stage1_ckpt, allow_pickle=True)
    nset = 0
    for k in s1.files:
        if not k.startswith("bottleneck/"):
            continue
        tup = tuple(int(x) if x.isdigit() else x for x in k[len("bottleneck/"):].split("/"))
        for fk in list(flat.keys()):
            if tuple(str(z) for z in fk[:len(tup)]) == tuple(str(z) for z in tup):
                flat[fk] = np.asarray(s1[k], np.float32); nset += 1; break
    state.replace_by_pure_dict(trav.unflatten_dict(flat))
    bneck = nnx.merge(gd, state)
    print(f"[precompute] overlaid {nset} bottleneck params", flush=True)

    @nnx.jit
    def run(lat, rec, ac, pr, mm, lg, rm, am, pm):
        mem, _, _ = bneck(lat, rec, ac, pr, mm, lang_emb=lg, recent_mask=rm, action_mask=am, proprio_mask=pm)
        return mem

    def _fixed_pad(arr, L, D):
        out = np.zeros((L, D), np.float32); msk = np.zeros((L,), bool)
        a = arr[-L:] if arr.shape[0] > L else arr
        if a.shape[0]:
            out[:a.shape[0]] = a; msk[:a.shape[0]] = True
        return out, msk

    def gather_variant(epi_dir, t, v, vi):
        """Exact replica of the Stage-2 offline gather for a SPECIFIC variant v; returns padded
        inputs + masks, or None if v is not causal at t."""
        fr = vi[f"v{v}"]["frame_idx"]
        if not fr or fr[0] > t:
            return None
        fidx = np.array(fr, np.int64); fidx = fidx[fidx <= t]; n_jit = len(fidx)
        lat = np.load(os.path.join(epi_dir, f"latents_v{v}.npy"))
        nc = max(0, n_jit // 4); kept = min(lat.shape[0], nc); lat = lat[:kept]
        latent_dim = lat.shape[-1] if lat.ndim == 3 else 48
        if POOL_HW and lat.shape[0] > 0:
            ncur = lat.shape[0]; spatial = 64; Vv = lat.shape[1] // spatial
            side = 8; ph = int(POOL_HW); blk = side // ph
            gg = lat.reshape(ncur, Vv, side, side, latent_dim)
            gg = gg.reshape(ncur, Vv, ph, blk, ph, blk, latent_dim).mean(axis=(3, 5))
            lat = gg.reshape(ncur, Vv * ph * ph, latent_dim)
        clat = (np.ascontiguousarray(lat).reshape(-1, latent_dim).astype(np.float32)
                if lat.shape[0] > 0 else np.zeros((0, latent_dim), np.float32))
        recent_mm = np.load(os.path.join(epi_dir, "recent_siglip.npy"), mmap_mode="r")
        sdim = recent_mm.shape[-1]; vb = kept * 4
        ridx = fidx[vb:] if vb < n_jit else fidx[n_jit:]
        rec = (np.ascontiguousarray(recent_mm[ridx]).reshape(-1, sdim).astype(np.float32)
               if len(ridx) else np.zeros((0, sdim), np.float32))
        ap = np.load(os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True).item()
        va = np.ascontiguousarray(ap["action"][fidx]).astype(np.float32)
        vp = np.ascontiguousarray(ap["proprio"][fidx]).astype(np.float32)
        if act_enc == "raw": ae = va
        else:
            da = np.zeros_like(va); da[1:] = va[1:] - va[:-1]; ae = np.concatenate([va, da], -1)
        if pro_enc == "raw": pe = vp
        else:
            dp = np.zeros_like(vp); dp[1:] = vp[1:] - vp[:-1]; pe = np.concatenate([vp, dp], -1)
        rec_p, rec_m = _fixed_pad(rec, MAX_RECENT, sdim)
        act_p, act_m = _fixed_pad(ae, MAX_STEPS, ae.shape[-1] if ae.size else (12 if act_enc == "raw" else 24))
        pro_p, pro_m = _fixed_pad(pe, MAX_STEPS, pe.shape[-1] if pe.size else (16 if pro_enc == "raw" else 32))
        lat_p = np.zeros((LATENT_CAP, clat.shape[-1] if clat.size else 48), np.float32)
        lat_m = np.zeros((LATENT_CAP,), bool); lc = min(clat.shape[0], LATENT_CAP)
        if lc: lat_p[:lc] = clat[-LATENT_CAP:][:lc]; lat_m[:lc] = True
        lp = os.path.join(epi_dir, "lang_emb.npy")
        lang = (np.load(lp).astype(np.float32).reshape(-1) if os.path.exists(lp)
                else np.zeros((lang_dim,), np.float32))
        return lat_p, lat_m, rec_p, rec_m, act_p, act_m, pro_p, pro_m, lang

    samples = json.load(open(os.path.join(args.store, "meta", "samples.json")))
    if args.max_samples:
        samples = samples[:args.max_samples]
    os.makedirs(args.out, exist_ok=True)
    mine = [i for i in range(len(samples)) if i % args.nshard == args.shard]
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} samples", flush=True)
    done = 0
    for si in mine:
        epi, t = int(samples[si][0]), int(samples[si][1])
        outp = os.path.join(args.out, f"ep{epi}_t{t}.npz")
        if os.path.exists(outp):
            done += 1; continue
        epi_dir = os.path.join(args.store, "features", f"episode_{epi}")
        vi = json.load(open(os.path.join(epi_dir, "variant_index.json")))
        toks = {}
        for v in range(n_variants):
            g = gather_variant(epi_dir, t, v, vi)
            if g is None:
                continue
            lat_p, lat_m, rec_p, rec_m, act_p, act_m, pro_p, pro_m, lang = g
            def _b(x): return jnp.asarray(x)[None]
            # run(lat, rec, ac, pr, mm, lg, rm, am, pm) -- map correctly (data + masks)
            mem = run(_b(lat_p), _b(rec_p), _b(act_p), _b(pro_p), _b(lat_m), _b(lang), _b(rec_m), _b(act_m), _b(pro_m))
            toks[f"v{v}"] = np.asarray(mem[0], np.float16)
        if toks:
            np.savez(outp, **toks)
        done += 1
        if done % 200 == 0:
            print(f"[shard {args.shard}] {done}/{len(mine)}", flush=True)
    print(f"[shard {args.shard}] DONE {done} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
