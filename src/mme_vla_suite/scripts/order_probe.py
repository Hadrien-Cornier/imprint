"""Linear probe for cover-block color order in the Stage-1 memory bottleneck."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


def _pool_rect_latents(lat, cfg):
    if lat.shape[0] == 0:
        return lat.reshape(0, lat.shape[-1])
    cbm = cfg.cosmo_bottleneck_memory
    ph = int(getattr(cbm, "pool_h", 0) or 0)
    pw = int(getattr(cbm, "pool_w", 0) or 0)
    if not (ph and pw):
        return np.ascontiguousarray(lat).reshape(-1, lat.shape[-1]).astype(np.float32)
    gh = int(getattr(cbm, "wan_h", 192)) // 16
    gw = int(getattr(cbm, "wan_w", 256)) // 16
    spatial = gh * gw
    latent_dim = lat.shape[-1]
    pooled_spatial = ph * pw
    if lat.shape[1] % pooled_spatial == 0 and lat.shape[1] < spatial:
        return np.ascontiguousarray(lat).reshape(-1, latent_dim).astype(np.float32)
    views = lat.shape[1] // spatial
    bh, bw = gh // ph, gw // pw
    grid = lat.reshape(lat.shape[0], views, gh, gw, latent_dim)
    grid = grid.reshape(lat.shape[0], views, ph, bh, pw, bw, latent_dim).mean(axis=(3, 5))
    return np.ascontiguousarray(grid.reshape(lat.shape[0], views * ph * pw, latent_dim)).reshape(-1, latent_dim).astype(np.float32)


def _encode_action(action, mode):
    if mode == "raw":
        return action.astype(np.float32)
    delta = np.zeros_like(action)
    delta[1:] = action[1:] - action[:-1]
    return np.concatenate([action, delta], axis=-1).astype(np.float32)


def _gather_probe_sample(store, cfg, ep, t):
    epi_dir = os.path.join(store, "features", f"episode_{ep}")
    cbm = cfg.cosmo_bottleneck_memory
    chunk_frames = int(getattr(cbm, "temporal_factor", 4))

    lat = np.load(os.path.join(epi_dir, "latents.npy"), mmap_mode="r")
    n_chunks = max(0, (int(t) + 1) // chunk_frames)
    lat = _pool_rect_latents(np.asarray(lat[:n_chunks]), cfg)

    recent_mm = np.load(os.path.join(epi_dir, "recent_siglip.npy"), mmap_mode="r")
    boundary = n_chunks * chunk_frames
    rec = np.ascontiguousarray(recent_mm[boundary:int(t) + 1]).reshape(-1, recent_mm.shape[-1]).astype(np.float32)

    ap = np.load(os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True).item()
    action_src = ap.get("action_causal", ap["action"])
    action = np.ascontiguousarray(action_src[:int(t) + 1]).astype(np.float32)
    proprio = np.ascontiguousarray(ap["proprio"][:int(t) + 1]).astype(np.float32)
    act = _encode_action(action, str(getattr(cbm, "action_encoding", "raw_delta")))
    pro = _encode_action(proprio, str(getattr(cbm, "proprio_encoding", "raw_delta")))

    lang_p = os.path.join(epi_dir, "lang_emb.npy")
    lang_dim = int(getattr(cbm, "lang_emb_dim", 2048))
    lang = np.load(lang_p).astype(np.float32).reshape(-1) if os.path.exists(lang_p) else np.zeros((lang_dim,), np.float32)
    return {"lat": lat, "rec": rec, "act": act, "pro": pro, "lang": lang}


def _pad(arrs, fixed_len=None):
    max_len = max(1, max(a.shape[0] for a in arrs))
    out_len = int(fixed_len or max_len)
    dim = arrs[0].shape[-1]
    out = np.zeros((len(arrs), out_len, dim), np.float32)
    mask = np.zeros((len(arrs), out_len), bool)
    for i, arr in enumerate(arrs):
        arr = arr[-out_len:] if arr.shape[0] > out_len else arr
        if arr.shape[0]:
            out[i, : arr.shape[0]] = arr
            mask[i, : arr.shape[0]] = True
    return out, mask


def _make_folds(ep_to_class, folds, seed):
    by_class = defaultdict(list)
    rng = np.random.default_rng(seed)
    for ep, cls in ep_to_class.items():
        by_class[int(cls)].append(int(ep))
    split = [[] for _ in range(folds)]
    for eps in by_class.values():
        eps = list(eps)
        rng.shuffle(eps)
        for i, ep in enumerate(eps):
            split[i % folds].append(ep)
    return [set(x) for x in split]


def _softmax_fit_eval(x_train, y_train, x_eval, y_eval, num_classes, steps=300, lr=0.2, l2=1e-3):
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sd
    x_eval = (x_eval - mu) / sd
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), np.float32)], axis=1)
    x_eval = np.concatenate([x_eval, np.ones((x_eval.shape[0], 1), np.float32)], axis=1)
    w = np.zeros((x_train.shape[1], num_classes), np.float32)
    y_one = np.eye(num_classes, dtype=np.float32)[y_train]
    for _ in range(int(steps)):
        logits = x_train @ w
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        grad = x_train.T @ (prob - y_one) / x_train.shape[0]
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    logits = x_eval @ w
    pred = logits.argmax(axis=1)
    return pred, logits


def run_order_probe(
    bottleneck,
    cfg,
    store,
    labels_path,
    step,
    out_dir,
    *,
    folds=5,
    steps_per_episode=8,
    batch_size=32,
    seed=0,
):
    with open(labels_path) as f:
        label_payload = json.load(f)
    permutations = label_payload["permutations"]
    ep_to_class = {int(ep): int(v["class_id"]) for ep, v in label_payload["labels"].items()}
    episodes = sorted(ep_to_class)

    sample_meta = []
    gathered = []
    for ep in episodes:
        ci = json.load(open(os.path.join(store, "features", f"episode_{ep}", "chunk_index.json")))
        n = int(ci["num_steps"])
        ts = np.unique(np.linspace(0, n - 1, int(steps_per_episode), dtype=np.int64))
        for t in ts:
            sample_meta.append((ep, int(t), ep_to_class[ep]))
            gathered.append(_gather_probe_sample(store, cfg, ep, int(t)))

    feats = []
    for start in range(0, len(gathered), int(batch_size)):
        batch = gathered[start:start + int(batch_size)]
        lat, lat_m = _pad([g["lat"] for g in batch], fixed_len=2560)
        rec, rec_m = _pad([g["rec"] for g in batch], fixed_len=2048)
        act, act_m = _pad([g["act"] for g in batch], fixed_len=800)
        pro, pro_m = _pad([g["pro"] for g in batch], fixed_len=800)
        lang = np.stack([g["lang"] for g in batch], axis=0).astype(np.float32)
        mem, _, _ = bottleneck(
            jnp.asarray(lat), jnp.asarray(rec), jnp.asarray(act), jnp.asarray(pro),
            jnp.asarray(lat_m), lang_emb=jnp.asarray(lang),
            recent_mask=jnp.asarray(rec_m), action_mask=jnp.asarray(act_m), proprio_mask=jnp.asarray(pro_m),
        )
        mem = np.asarray(jax.device_get(mem), np.float32)
        feats.append(np.concatenate([mem.mean(axis=1), mem.max(axis=1)], axis=-1))
    x = np.concatenate(feats, axis=0)
    y = np.asarray([m[2] for m in sample_meta], np.int64)
    ep_arr = np.asarray([m[0] for m in sample_meta], np.int64)

    fold_sets = _make_folds(ep_to_class, int(folds), int(seed))
    all_pred = np.full_like(y, -1)
    all_logits = np.zeros((len(y), len(permutations)), np.float32)
    for fold_eps in fold_sets:
        eval_mask = np.asarray([ep in fold_eps for ep in ep_arr])
        train_mask = ~eval_mask
        pred, logits = _softmax_fit_eval(
            x[train_mask], y[train_mask], x[eval_mask], y[eval_mask], len(permutations)
        )
        all_pred[eval_mask] = pred
        all_logits[eval_mask] = logits

    sample_acc = float(np.mean(all_pred == y))
    per_class_acc = {}
    for cls, name in enumerate(permutations):
        m = y == cls
        if np.any(m):
            per_class_acc[name] = float(np.mean(all_pred[m] == y[m]))
    balanced_acc = float(np.mean(list(per_class_acc.values()))) if per_class_acc else 0.0

    ep_preds = {}
    for ep in episodes:
        m = ep_arr == ep
        ep_preds[ep] = int(all_logits[m].mean(axis=0).argmax())
    episode_acc = float(np.mean([ep_preds[ep] == ep_to_class[ep] for ep in episodes]))

    confusion = np.zeros((len(permutations), len(permutations)), np.int64)
    for truth, pred in zip(y, all_pred):
        confusion[int(truth), int(pred)] += 1

    metrics = {
        "order_probe/sample_acc": sample_acc,
        "order_probe/episode_acc": episode_acc,
        "order_probe/balanced_acc_observed": balanced_acc,
        "order_probe/num_samples": int(len(y)),
        "order_probe/num_episodes": int(len(episodes)),
    }
    for name, acc in per_class_acc.items():
        metrics[f"order_probe/class_acc/{name}"] = acc

    os.makedirs(out_dir, exist_ok=True)
    result = {
        "step": int(step),
        "permutations": permutations,
        "metrics": metrics,
        "per_class_acc": per_class_acc,
        "confusion": confusion.tolist(),
        "episode_predictions": {
            str(ep): {
                "true": permutations[ep_to_class[ep]],
                "pred": permutations[ep_preds[ep]],
            }
            for ep in episodes
        },
    }
    out_path = os.path.join(out_dir, f"order_probe_step{int(step)}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return metrics, out_path
