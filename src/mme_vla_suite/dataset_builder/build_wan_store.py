"""Offline Wan-latent memory store builder (sub-project A). Per-episode packing.

Per episode <store>/features/episode_<N>/:
  latents.npy       (num_chunks, V*spatial, 48)   finalized Wan chunks, step order
  chunk_index.json  {num_steps, chunk_frames, downsample, num_views, step_to_chunk, num_chunks}
  recent_siglip.npy (steps, V, 64, 2048)          recent-INPUT stream (SigLIP-64)
  align_prefix.npy  (steps, 64, 2048)             align TARGET (PaliGemma-prefix-64, front view)
  action_proprio.npy dict{action,proprio,da,dp}
  (reward.npy is a SEPARATE second-pass augment -- NOT built here)
meta/samples.json enumerates (episode, root_step_t).

Feature extractors (SigLIP, PaliGemma-prefix) are parameterized by a pi0.5 checkpoint path;
they are STUBBED to raise a clear NotImplementedError if the checkpoint is absent, so the
build orchestration + Wan-latent path are testable now and the feature extraction wires in
once the pi05_unified_human300_100k checkpoint exists.

INVARIANT (see tests/cosmo/test_wan_store.py::test_store_prefix_equals_online_buffer):
the offline store, for a sample rooted at step t, equals the online CosmoMemoryBuffer at
step t. The store encodes each episode ONCE (finalized_encode_multiview over the downsampled
episode) and a training sample slices the prefix of finalized chunks <= t. The online buffer
builds incrementally via _encode_chunk (== encode_multiview per standalone chunk). Because the
Wan2.2 VAE is strictly causal and every finalized chunk is stable under future appends,
encode-once-and-slice reproduces the incremental path exactly.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable

import numpy as np

_MISSING_CKPT_MSG = (
    "prefix/siglip extraction needs pi05 checkpoint -- wire once "
    "pi05_unified_human300_100k exists"
)


# --------------------------------------------------------------------------- #
# Frame helpers                                                               #
# --------------------------------------------------------------------------- #
def _downsample_frames(frames_v: np.ndarray, downsample: int) -> np.ndarray:
    """(T, V, H, W, 3) -> (ceil(T/downsample), V, H, W, 3), taking every downsample-th frame.

    Mirrors the online buffer's ``step % downsample == 0`` selection (step 0 kept).
    """
    frames_v = np.asarray(frames_v)
    assert frames_v.ndim == 5 and frames_v.shape[-1] == 3, (
        f"expected (T,V,H,W,3), got {frames_v.shape}"
    )
    return np.ascontiguousarray(frames_v[::downsample])


def _with_delta(arr: np.ndarray) -> np.ndarray:
    """Return the per-step first difference (delta[0]=0), same shape as arr."""
    arr = np.asarray(arr, np.float32)
    delta = np.zeros_like(arr)
    if arr.shape[0] > 1:
        delta[1:] = arr[1:] - arr[:-1]
    return delta


# --------------------------------------------------------------------------- #
# Per-episode packing                                                         #
# --------------------------------------------------------------------------- #
def build_episode(
    frames_v: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    wan_frontend,
    siglip_fn: Callable[[np.ndarray], np.ndarray],
    prefix_fn: Callable[[np.ndarray], np.ndarray],
    downsample: int,
    chunk_frames: int,
    out_dir: str,
    episode_id: int = 0,
    no_latents: bool = False,
    reward_curves: dict | None = None,
    dump_frames_res: int = 0,
    align_prompt: str | None = None,
    align_all_views: bool = False,
) -> list[tuple[int, int]]:
    """Encode ONE episode into the per-episode store layout and return its samples.

    Args:
      frames_v: (T, V, H, W, 3) uint8 raw camera frames (front view is view 0).
      states:   (T, state_dim) float32 proprio.
      actions:  (T, action_dim) float32 executed actions.
      wan_frontend: object with .finalized_encode_multiview and .temporal_factor.
      siglip_fn: callable (t, V, H, W, 3) -> (t, V, 64, 2048) recent-input SigLIP stream.
      prefix_fn: callable (t, H, W, 3) -> (t, 64, 2048) or
        (t, V, H, W, 3), prompt -> (t, V*64, 2048) PaliGemma-prefix align target.
      downsample: keep every downsample-th frame (matches online buffer).
      chunk_frames: frames per consolidated Wan chunk (== wan_frontend.temporal_factor).
      out_dir: <store>/features/episode_<N>/ directory (created if absent).
      episode_id: id used in the returned (episode, root_t) sample tuples.

    Returns:
      list of (episode_id, root_step_t) for every downsampled step.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames_v = np.asarray(frames_v)
    states = np.asarray(states, np.float32)
    actions = np.asarray(actions, np.float32)
    assert frames_v.shape[0] == states.shape[0] == actions.shape[0], (
        "frames/states/actions must share the time axis"
    )

    # 1) Downsample the whole episode ONCE.
    ds = _downsample_frames(frames_v, downsample)   # (t, V, H, W, 3)
    n_steps = ds.shape[0]
    V = ds.shape[1]
    ds_states = states[::downsample]
    ds_actions = actions[::downsample]
    # Causal action stream for memory. Online eval keeps frame r when
    # r % downsample == 0, but the action available before predicting at r is
    # the previous raw control action from r-1, not the current target action
    # at r. Store both streams: "action" remains the reconstruction/policy
    # target, while "action_causal" is the memory input stream.
    kept_raw_idx = np.arange(0, actions.shape[0], downsample, dtype=np.int64)
    ds_actions_causal = np.zeros_like(ds_actions)
    valid_prev = kept_raw_idx > 0
    if np.any(valid_prev):
        ds_actions_causal[valid_prev] = actions[kept_raw_idx[valid_prev] - 1]

    # 1b) Optionally dump raw uint8 frames for ONLINE Wan latents at train time.
    #     Stored at dump_frames_res so the training-loop WanFrontend can compute
    #     latents from JITTERED frame subsets (any stride/start) without decoding
    #     video in the data loader. Saved per-view; (t, V, R, R, 3) uint8.
    if dump_frames_res and dump_frames_res > 0:
        import cv2
        R = int(dump_frames_res)
        t_, v_ = ds.shape[0], ds.shape[1]
        fr = np.empty((t_, v_, R, R, 3), np.uint8)
        for i in range(t_):
            for j in range(v_):
                fr[i, j] = cv2.resize(ds[i, j], (R, R), interpolation=cv2.INTER_AREA)
        np.save(os.path.join(out_dir, "frames.npy"), fr)

    # 2) Wan finalized chunks: encode-once over the downsampled episode.
    #    Skipped when no_latents=True -- the Wan tokenizer is deterministic and
    #    cheap enough to recompute ONLINE from the same downsampled frames, so we
    #    only precompute the expensive/target streams here. chunk_index is still
    #    written (num_chunks derived analytically) so the dataloader/buffer can
    #    reconstruct step->chunk mapping identically.
    if no_latents:
        # Wan drops the terminal (unfinalized) chunk: finalized chunks = n//cf if
        # n%cf==0 else n//cf ... but finalized_encode_multiview yields exactly the
        # complete chunks, i.e. floor(n/cf) with the terminal partial excluded.
        num_chunks = n_steps // chunk_frames
        latents = None
    else:
        latents = wan_frontend.finalized_encode_multiview(ds)   # (num_chunks, V*spatial, 48)
        latents = np.asarray(latents, np.float32)
        num_chunks = int(latents.shape[0])

    # step_to_chunk: downsampled step i belongs to chunk (i // chunk_frames);
    # -1 if that chunk is not (yet) finalized (i.e. still terminal/partial).
    step_to_chunk = [
        (i // chunk_frames) if (i // chunk_frames) < num_chunks else -1
        for i in range(n_steps)
    ]
    chunk_index = {
        "num_steps": n_steps,
        "chunk_frames": int(chunk_frames),
        "downsample": int(downsample),
        "num_views": int(V),
        "step_to_chunk": step_to_chunk,
        "num_chunks": num_chunks,
        "latents_online": bool(no_latents),
    }

    # 3) Recent-input SigLIP stream, per downsampled frame/view.
    recent_siglip = np.asarray(siglip_fn(ds), np.float32)   # (t, V, 64, 2048)

    # 4) Align target. Default keeps the historical front-view image-only target
    #    for callers that have not opted into the RoboDojo 3-view language target.
    if align_all_views:
        align_prefix = np.asarray(
            prefix_fn(np.ascontiguousarray(ds), prompt=align_prompt),
            np.float32,
        )  # (t, V*64, 2048)
    else:
        align_prefix = np.asarray(
            prefix_fn(np.ascontiguousarray(ds[:, 0]), prompt=align_prompt),
            np.float32,
        )  # (t, 64, 2048)

    # 5) Action/proprio + first differences.
    action_proprio = {
        "action": ds_actions.astype(np.float32),
        "action_causal": ds_actions_causal.astype(np.float32),
        "proprio": ds_states.astype(np.float32),
        "da": _with_delta(ds_actions),
        "da_causal": _with_delta(ds_actions_causal),
        "dp": _with_delta(ds_states),
    }

    # 6) Reward target: per-mode unified reward, downsampled to the memory frames.
    #    reward_curves[mode] is a FULL-LENGTH (T,) array over raw frames; we take
    #    the SAME [::downsample] frames the memory sees so it is per-step aligned.
    if reward_curves is not None:
        reward = {}
        for mode, curve in reward_curves.items():
            c = np.asarray(curve, np.float32)
            assert c.shape[0] == states.shape[0], (
                f"reward '{mode}' len {c.shape[0]} != episode T {states.shape[0]}"
            )
            reward[mode] = c[::downsample].astype(np.float32)   # (t,)
        np.save(os.path.join(out_dir, "reward.npy"), reward, allow_pickle=True)

    # 7) Persist (mmap-friendly .npy per stream). latents skipped when online.
    if latents is not None:
        np.save(os.path.join(out_dir, "latents.npy"), latents)
    np.save(os.path.join(out_dir, "recent_siglip.npy"), recent_siglip)
    np.save(os.path.join(out_dir, "align_prefix.npy"), align_prefix)
    np.save(os.path.join(out_dir, "action_proprio.npy"), action_proprio, allow_pickle=True)
    with open(os.path.join(out_dir, "chunk_index.json"), "w") as f:
        json.dump(chunk_index, f)

    return [(int(episode_id), t) for t in range(n_steps)]


# --------------------------------------------------------------------------- #
# Checkpoint-parameterized feature extractors (wired to pi0.5 99999)          #
# --------------------------------------------------------------------------- #
#
# WIRING (cosmo-option-d):
#   siglip_fn: EXACT MemoryBuffer.add_buffer recipe -- SigLipTokenizer (pi05
#     siglip_params.pkl) on frames normalized to [-1,1], resize_with_pad(224),
#     giving (t,v,256,2048); pooled 16x16->8x8=64 via pool_tokens_to_size.
#     This is the PRE-LLM SigLIP feature (recent-INPUT stream).
#   prefix_fn: FRONT-VIEW ONLY. Loads the pi0.5 model from the checkpoint,
#     runs embed_prefix on an IMAGE-ONLY observation (no language prompt, to
#     keep the align target task-agnostic), then runs the full gemma_2b
#     (PaliGemma.llm) prefix forward pass exactly as pi0.sample_actions does
#     (make_attn_mask + cumsum positions) and reads prefix_out -- the
#     LAST-LAYER hidden states of the 256 image tokens (16x16 @224px, 2048-d)
#     AFTER the PaliGemma LLM backbone. Pooled 16x16->8x8=64. This is the
#     VLM-perception (align TARGET) feature and is DISTINCT from SigLIP.


def _load_model(ckpt: str, config_name: str):
    """Load the pi0.5 model ONCE from the checkpoint (shared by both extractors).

    The model's OWN vision encoder (model.vision_encode -> PaliGemma.img) is what
    the ONLINE CosmoMemoryBuffer uses at inference (policy.py passes
    module_jit(model.vision_encode) as siglip_fn). So the offline store MUST use
    the SAME finetuned encoder -- NOT the standalone base siglip_params.pkl -- or
    the offline==online core invariant breaks.
    """
    if ckpt is None or not os.path.exists(ckpt):
        raise NotImplementedError(_MISSING_CKPT_MSG)
    import jax.numpy as jnp
    from openpi.training import config as _openpi_config
    from openpi.models import model as _model
    train_config = _openpi_config.get_config(config_name)
    model = train_config.model.load(
        _model.restore_params(os.path.join(ckpt, "params"), dtype=jnp.bfloat16)
    )
    return model


def _load_siglip_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """(t, V, H, W, 3) uint8 -> (t, V, 64, 2048) recent-input SigLIP-64.

    Uses the CHECKPOINT's finetuned vision encoder (model.vision_encode ==
    PaliGemma.img), matching the online buffer's siglip_fn exactly (same [-1,1]
    normalization + resize_with_pad(224) as CosmoMemoryBuffer._encode_recent),
    then pools 256->64 to match the store schema / dataloader (recent_siglip is
    (steps,V,64,2048)). NOTE: the online deploy siglip_fn must pool to 64 too.
    """
    import jax
    import jax.numpy as jnp
    import einops
    from openpi.shared import image_tools
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size

    # Use the checkpoint's finetuned SigLIP encoder. HistoryPi0 exposes
    # .vision_encode; plain Pi0 does not, but PaliGemma.img is the same encoder
    # (Pi0.embed_prefix calls self.PaliGemma.img). Wrap whichever exists so the
    # (k,v,224,224,3)->(k,v,256,2048) contract matches the online buffer.
    if hasattr(model, "vision_encode"):
        _enc = model.vision_encode
    else:
        def _enc(imgs):  # (k,v,224,224,3) -> (k,v,256,2048)
            k, v = imgs.shape[0], imgs.shape[1]
            flat = einops.rearrange(imgs, "k v h w c -> (k v) h w c")
            tok, _ = model.PaliGemma.img(flat, train=False)
            return einops.rearrange(tok, "(k v) p d -> k v p d", k=k, v=v)
    vision_enc = jax.jit(_enc)

    def siglip_fn(frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        assert frames.ndim == 5 and frames.shape[-1] == 3, (
            f"expected (t,V,h,w,3), got {frames.shape}"
        )
        t, v = frames.shape[0], frames.shape[1]
        image_jnp = jnp.array(frames.astype(np.float32) / 255.0 * 2.0 - 1.0)
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = vision_enc(image_jnp)               # (t, v, 256, 2048) finetuned SigLIP
        pooled = pool_tokens_to_size(output_emb, 64)     # (t, v, 64, 2048)
        return np.asarray(jax.device_get(pooled), np.float32)

    return siglip_fn


def _load_prefix_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """PaliGemma-prefix align targets from the loaded pi0.5 checkpoint.

    Historical callers pass a front-view array `(t,H,W,3)` and get the old
    image-only `(t,64,2048)` target. RoboDojo passes all views `(t,V,H,W,3)`
    plus the episode prompt; this runs pi0.5's native prefix path with all three
    images and language, then supervises the language-conditioned image-token
    hidden states for all views, pooled to `(t,V*64,2048)`.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import image_tools
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size

    @jax.jit
    def _prefix_hidden(img):  # img: (b, 224, 224, 3) float32 in [-1,1]
        b = img.shape[0]
        obs = _model.Observation(
            images={"base_0_rgb": img},
            image_masks={"base_0_rgb": jnp.ones((b,), dtype=jnp.bool_)},
            state=jnp.zeros((b, model.action_dim), jnp.float32),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attn, positions=positions
        )
        # image-only prefix => prefix_out is (b, 256, 2048) image-token hidden states.
        pooled = pool_tokens_to_size(prefix_out.astype(jnp.float32), 64)  # (b, 64, 2048)
        return pooled

    @jax.jit
    def _prefix_hidden_3view_lang(base, right, wrist, tokenized_prompt, tokenized_prompt_mask):
        b = base.shape[0]
        obs = _model.Observation(
            # Keep token blocks in RoboDojo store order: image, right_image, wrist_image.
            images={
                "base_0_rgb": base,
                "right_wrist_0_rgb": right,
                "left_wrist_0_rgb": wrist,
            },
            image_masks={
                "base_0_rgb": jnp.ones((b,), dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones((b,), dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones((b,), dtype=jnp.bool_),
            },
            state=jnp.zeros((b, model.action_dim), jnp.float32),
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attn, positions=positions
        )
        # embed_prefix concatenates 256 tokens per image before text. Select only
        # image-token outputs, now conditioned by the full image+language prefix.
        image_out = prefix_out[:, : 3 * 256].astype(jnp.float32)
        image_out = image_out.reshape(b, 3, 256, image_out.shape[-1])
        pooled = pool_tokens_to_size(image_out, 64)  # (b, 3, 64, 2048)
        return pooled.reshape(b, 3 * 64, pooled.shape[-1])

    max_token_len = int(getattr(model, "max_token_len", 64))
    tokenizer = PaligemmaTokenizer(max_len=max_token_len)

    def prefix_fn(frames: np.ndarray, prompt: str | None = None) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim == 4:
            assert frames.shape[-1] == 3, f"expected (t,h,w,3), got {frames.shape}"
            img = jnp.array(frames.astype(np.float32) / 255.0 * 2.0 - 1.0)
            img = image_tools.resize_with_pad(img, 224, 224)     # (t,224,224,3)
            pooled = _prefix_hidden(img)                         # (t,64,2048)
            return np.asarray(jax.device_get(pooled), np.float32)

        assert frames.ndim == 5 and frames.shape[1] == 3 and frames.shape[-1] == 3, (
            f"expected (t,3,h,w,3) for 3-view align target, got {frames.shape}"
        )
        if prompt is None:
            raise ValueError("prompt is required for 3-view language-conditioned align target")
        img = jnp.array(frames.astype(np.float32) / 255.0 * 2.0 - 1.0)
        img = image_tools.resize_with_pad(img.reshape((-1,) + frames.shape[2:]), 224, 224)
        img = img.reshape((frames.shape[0], frames.shape[1], 224, 224, 3))
        toks, tok_mask = tokenizer.tokenize(str(prompt))
        toks = jnp.asarray(np.broadcast_to(toks, (frames.shape[0], toks.shape[0])), jnp.int32)
        tok_mask = jnp.asarray(np.broadcast_to(tok_mask, (frames.shape[0], tok_mask.shape[0])), jnp.bool_)
        pooled = _prefix_hidden_3view_lang(img[:, 0], img[:, 1], img[:, 2], toks, tok_mask)
        return np.asarray(jax.device_get(pooled), np.float32)

    return prefix_fn


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _iter_episodes(lerobot_dir: str, num_views: int, episodes):
    """Yield (episode_id, frames_v (T,V,H,W,3) uint8, states (T,16), actions (T,12)).

    Uses GrootOpenpiSingleDataset (byte-identical preprocessing to the training
    path). Views (in order): agentview_left (view 0 = FRONT), agentview_right,
    eye_in_hand. `episodes` is an int N (first N trajectories) or a list of ids.
    """
    from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiSingleDataset

    ds = GrootOpenpiSingleDataset(
        {"path": lerobot_dir, "filter_key": None}, action_horizon=1
    )
    traj_ids = list(ds.trajectory_ids)
    traj_lens = list(ds.trajectory_lengths)
    # global step offset for each trajectory (trajectories are contiguous in all_steps).
    offsets = np.concatenate([[0], np.cumsum(traj_lens)])

    if episodes is None:
        wanted = list(range(len(traj_ids)))
    elif isinstance(episodes, int):
        wanted = list(range(min(episodes, len(traj_ids))))
    else:
        wanted = list(episodes)

    view_order = ["observation/image", "observation/right_image", "observation/wrist_image"][:num_views]

    for ti in wanted:
        start = int(offsets[ti])
        length = int(traj_lens[ti])
        frames_steps, state_steps, action_steps = [], [], []
        for base in range(length):
            item = ds[start + base]
            views = [np.asarray(item[k], np.uint8) for k in view_order]
            frames_steps.append(np.stack(views, axis=0))           # (V,H,W,3)
            state_steps.append(np.asarray(item["observation/state"], np.float32))  # (16,)
            action_steps.append(np.asarray(item["actions"], np.float32)[0])        # (12,)
        frames_v = np.stack(frames_steps, axis=0)                  # (T,V,H,W,3)
        states = np.stack(state_steps, axis=0)                     # (T,16)
        actions = np.stack(action_steps, axis=0)                   # (T,12)
        yield int(traj_ids[ti]), frames_v, states, actions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline Wan-latent memory store builder (per-episode packing)."
    )
    parser.add_argument("--lerobot-dir", required=True, help="LeRobot dataset root.")
    parser.add_argument("--out", required=True, help="Output store directory.")
    parser.add_argument(
        "--ckpt", default=None,
        help="pi0.5 checkpoint path for SigLIP/PaliGemma-prefix extractors. "
             "Required for real feature extraction.",
    )
    parser.add_argument(
        "--config-name", default="pi05_washreturn_toy",
        help="openpi TrainConfig name whose model architecture matches --ckpt.",
    )
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument(
        "--episodes", default=None,
        help="Int N (first N episodes) OR comma-separated ids (default: all).",
    )
    parser.add_argument(
        "--no-latents", action="store_true",
        help="Skip Wan tokenizer latents (recomputed online); store only "
             "recent-SigLIP, align-prefix, action/proprio (+reward).",
    )
    parser.add_argument(
        "--reward-json", default=None,
        help="Path to combined reward json {ep: {mode: [per-frame...], n, split}}; "
             "writes reward.npy (per-mode, downsampled) per episode.",
    )
    parser.add_argument(
        "--dump-frames-res", type=int, default=0,
        help="If >0, also save raw uint8 frames.npy (t,V,R,R,3) at this resolution "
             "for ONLINE Wan latents at train time (enables stride/start jitter).",
    )
    args = parser.parse_args(argv)

    os.makedirs(os.path.join(args.out, "features"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)

    # Feature extractors are checkpoint-gated; this raises a clear
    # NotImplementedError until the checkpoint exists. Load the model ONCE and
    # share it -- both extractors use the checkpoint's finetuned encoder so the
    # store matches the online buffer (which uses model.vision_encode).
    model = _load_model(args.ckpt, args.config_name)
    siglip_fn = _load_siglip_fn(model)
    prefix_fn = _load_prefix_fn(model)

    # Wan frontend only needed when we actually precompute latents.
    if args.no_latents:
        wan = None
    else:
        from mme_vla_suite.models.representation.wan_frontend import WanFrontend
        wan = WanFrontend(device="cuda")

    # Optional reward labels (per-episode, per-mode, full-length curves).
    reward_by_ep = None
    if args.reward_json:
        with open(args.reward_json) as f:
            _rw = json.load(f)
        reward_by_ep = {}
        for ep_str, rec in _rw.items():
            reward_by_ep[int(ep_str)] = {
                m: rec[m] for m in ("incremental", "forward", "backward") if m in rec
            }

    if args.episodes is None:
        episodes = None
    elif "," in str(args.episodes):
        episodes = [int(e) for e in str(args.episodes).split(",")]
    else:
        episodes = int(args.episodes)  # first N trajectories
    all_samples: list[tuple[int, int]] = []
    for episode_id, frames_v, states, actions in _iter_episodes(
        args.lerobot_dir, args.num_views, episodes
    ):
        out_dir = os.path.join(args.out, "features", f"episode_{episode_id}")
        rc = reward_by_ep.get(int(episode_id)) if reward_by_ep is not None else None
        all_samples.extend(
            build_episode(
                frames_v=frames_v, states=states, actions=actions,
                wan_frontend=wan, siglip_fn=siglip_fn, prefix_fn=prefix_fn,
                downsample=args.downsample, chunk_frames=args.chunk_frames,
                out_dir=out_dir, episode_id=episode_id,
                no_latents=args.no_latents, reward_curves=rc,
                dump_frames_res=args.dump_frames_res,
            )
        )

    with open(os.path.join(args.out, "meta", "samples.json"), "w") as f:
        json.dump([list(s) for s in all_samples], f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
