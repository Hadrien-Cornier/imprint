"""Augment an EXISTING wan store with a per-episode pooled language embedding.

FIX(bug4): Stage-1 must train the bottleneck's language-FiLM path with the SAME
pooled PaliGemma text embedding that Stage-2 / inference feed -- otherwise FiLM is
untrained at Stage-2 start. Rather than rebuild the 125GB var8 store, we only ADD
one small file per episode:

    features/episode_<e>/lang_emb.npy   (lang_emb_dim,) float32

The embedding is the masked mean-pool of PaliGemma.llm(tokenize(prompt), method="embed")
over the episode's GENERIC instruction (from meta/episodes.jsonl "tasks"[0]) -- NOT the
left/right-container reward phrasing, so no answer leaks into memory conditioning.

Single source of truth: mme_vla_suite.shared.data_utils.pool_prompt_text_emb, the SAME
function policy.py._pool_prompt_text_emb calls at inference. This keeps the four consumers
(store-build / Stage-1 / Stage-2 / inference) byte-identical up to bf16.

MUST use the SAME checkpoint (finetuned pi0.5) the store's align_prefix was built from,
so the pooled text hidden states share the align target's PaliGemma weights.

Usage:
    OPENPI_DATA_HOME=/path/to/openpi-cache CUDA_VISIBLE_DEVICES=0 \
      python -m mme_vla_suite.dataset_builder.augment_store_lang \
        --store /path/to/washreturn_store_var8 \
        --lerobot /path/to/washreturn_lerobot \
        --ckpt <pi0.5 finetuned ckpt dir> --config <openpi config name>
"""
import argparse
import json
import os

import numpy as np


def _episode_task_map(lerobot_dir):
    """episode_index -> generic instruction string (tasks[0]) from meta/episodes.jsonl."""
    m = {}
    with open(os.path.join(lerobot_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            tasks = d.get("tasks") or [d.get("task")]
            m[int(d["episode_index"])] = tasks[0]
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="Add per-episode lang_emb.npy to a wan store.")
    p.add_argument("--store", required=True, help="wan store dir (has features/episode_*).")
    p.add_argument("--lerobot", required=True, help="source LeRobot dir (meta/episodes.jsonl).")
    p.add_argument("--ckpt", required=True, help="pi0.5 checkpoint dir used to build align_prefix.")
    p.add_argument("--config", required=True, help="openpi config name for that checkpoint.")
    p.add_argument("--overwrite", action="store_true", help="Rewrite lang_emb.npy if present.")
    args = p.parse_args(argv)

    # Reuse the store builder's model loader so the PaliGemma weights match align_prefix EXACTLY.
    from mme_vla_suite.dataset_builder.build_wan_store import _load_model
    from mme_vla_suite.shared.data_utils import pool_prompt_text_emb

    model = _load_model(args.ckpt, args.config)

    task_map = _episode_task_map(args.lerobot)
    # cache: only a few unique instruction strings -> encode each once.
    cache = {}
    feat = os.path.join(args.store, "features")
    eps = sorted(int(x.split("_")[1]) for x in os.listdir(feat) if x.startswith("episode_"))
    print(f"augmenting {len(eps)} episodes; {len(set(task_map.values()))} unique instruction(s)")

    for e in eps:
        out_path = os.path.join(feat, f"episode_{e}", "lang_emb.npy")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        prompt = task_map[e]
        if prompt not in cache:
            emb = np.asarray(pool_prompt_text_emb(model, prompt), np.float32).reshape(-1)
            cache[prompt] = emb
            print(f"  encoded [{emb.shape[0]}] :: {prompt!r}")
        np.save(out_path, cache[prompt])
    print(f"done -> {len(eps)} lang_emb.npy written under {feat}")


if __name__ == "__main__":
    main()
