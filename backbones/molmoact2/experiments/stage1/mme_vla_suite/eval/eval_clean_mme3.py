"""MEMORY pi0.5 (cosmo-bottleneck) eval on the CLEAN ShahRutav ReMemBench env.

Built from eval_clean_nomem.py (the CORRECT 3-camera live-env body: agentview_left/right
+ eye_in_hand rendered@256, [::-1] flip). Adds the mme memory-server protocol:
MMEClient(reset/add_buffer) + _push_memory() that feeds the SAME 3 training views, the
16-dim openpi-order state, and the prev executed action (openpi order, no reorder --
RoboMMEOutputs emits [eef,rot,grip,base,ctrl] = the order GrootOpenpi stored the memory
action stream). num_views=3 memory. Server = serve_policy_mme.py (MME_VLA_Policy, live
frozen bottleneck). Checkpoint holds BOTH the trained VLA/seam AND the frozen Stage-1
bottleneck (mem_encoder) in one params dir -> single --ckpt-dir, no separate mem ckpt.

(orig nomem header follows)
NO-MEMORY pi0.5 eval on the CLEAN ShahRutav ReMemBench env.

Derived DIRECTLY from eval_clean_mme2.py (the memory client that is the user's
baseline). The ONLY logic removed vs eval_clean_mme2.py is the mme memory-server
protocol: the MMEClient subclass, client.reset(), client.add_buffer(...) and the
_push_memory() helper. The env construction, ep_meta loading + per-episode
cycling (_load_all_ep_metas / _reset_env), adaptive/per-episode horizon,
_check_success, seed-offset sharding, resample_objects, video save and stats.json
are IDENTICAL to eval_clean_mme2.py so results stay head-to-head comparable.

Server: a STANDARD scripts/serve_policy.py policy:checkpoint
  --policy.config=pi05_pretrain_human300_unified --policy.dir=<99999>
(NOT serve_policy_mme). The checkpoint has NO memory buffer, so we speak the plain
openpi WebsocketClientPolicy .infer(element) protocol.

ELEMENT / camera contract (verified against the checkpoint's data config):
  The unified checkpoint is a pi05 model whose RobocasaInputs REQUIRES three images:
  observation/image (base = agentview_left), observation/wrist_image (eye_in_hand),
  observation/right_image (agentview_right). The unified LeRobot training data for the
  memory tasks was re-rendered with agentview_left + agentview_right + eye_in_hand
  (verified: left != right in the training videos). The ReMemBench demo hdf5 only
  records agentview_center + eye_in_hand, but the mujoco model DOES contain
  robot0_agentview_left / robot0_agentview_right cameras, so we render them directly
  from the sim (env.sim.render(camera_name=...)) and apply the SAME [::-1] vertical
  flip the clean client's _img() uses (verified: the lerobot conversion stored frames
  as raw_obs[::-1]). wrist stays eye_in_hand from the obs dict, matching _img().

Run from a cwd outside the RMBench source tree (namespace-package shadow)
with MUJOCO_GL=egl. make_env returns (env, kwargs) -> unwrap [0].
"""
import collections
import copy
import dataclasses
import glob
import json
import logging
import math
import os
import pathlib
from datetime import datetime

import h5py
import imageio
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _websocket_client_policy

from robocasa.utils.mem_utils import make_env, EnvArgs

DATA_ROOT = os.environ.get("RMBENCH_DATA_ROOT", "artifacts/datasets/rmbench")


class MMEClient(_websocket_client_policy.WebsocketClientPolicy):
    """WebsocketClientPolicy + the memory-protocol messages the mme server handles
    (reset clears the server buffer; add_buffer pushes a frame+state+action)."""

    def reset(self, prompt: str | None = None) -> None:
        self._ws.send(self._packer.pack({"reset": True, "prompt": prompt}))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"server error on reset:\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def add_buffer(self, mem_obs: dict) -> None:
        msg = dict(mem_obs)
        msg["add_buffer"] = True
        self._ws.send(self._packer.pack(msg))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"server error on add_buffer:\n{resp}")
        return msgpack_numpy.unpackb(resp)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    # MEMORY-input resolution (Wan latents + recent-SigLIP were built at 128px in the store).
    # Decoupled from resize_size (the 224 policy/VLA element). The memory buffer is fed at this res.
    mem_size: int = 128
    replan_steps: int = 5
    num_trials: int = 50
    # Hard upper cap on per-episode steps (author-specified 2048). The actual
    # per-task horizon is min(ceil(p90(demo_len)*1.5), max_horizon).
    max_horizon: int = 2048
    # Optional override: if >0, use this fixed horizon for every task (skip adaptive).
    fixed_horizon: int = 0
    # Re-sample object set/counts each reset (cycle demo ep_metas) -- SAME as the
    # memory client so both evals see the SAME demo IDs per episode.
    resample_objects: bool = True
    # size each episode's horizon to its cycled demo's own action_len (ceil*1.5, cap).
    per_episode_horizon: bool = True
    env_names: str = ""
    log_dir: str = None
    seed: int = 7
    save_video: bool = False


def _datasets_for(env_name):
    """All hdf5 files for a task (multiple timestamp dirs => more demos to cycle).
    MUST match the pi0.5 client's pool so both evals cycle the SAME demo IDs."""
    pats = sorted(glob.glob(f"{DATA_ROOT}/{env_name}/*/demo_im128_notp.hdf5"))
    if not pats:
        pats = sorted(glob.glob(f"{DATA_ROOT}/{env_name}/*/demo.hdf5"))
    assert pats, f"no dataset hdf5 for {env_name} under {DATA_ROOT}"
    return pats


def _dataset_for(env_name):
    return _datasets_for(env_name)[0]


def _adaptive_horizon(ds_path, max_horizon):
    """ceil(p90(demo_action_len) * 1.5) capped at max_horizon (HALO-style)."""
    try:
        with h5py.File(ds_path, "r") as f:
            data = f["data"]
            lens = [data[k]["actions"].shape[0] for k in data.keys()]
        if not lens:
            return max_horizon
        p90 = float(np.percentile(np.asarray(lens), 90))
        return int(min(math.ceil(p90 * 1.5), max_horizon))
    except Exception:
        return max_horizon


def _load_all_ep_metas(ds_paths):
    """Every demo's (ep_meta, action_len) across ALL of a task's hdf5 files, demo-sorted.
    IDENTICAL to the pi0.5 client (same file order, same sort, same tuple) so both evals
    cycle the SAME demo IDs and can size per-episode horizon to the same demo length."""
    if isinstance(ds_paths, str):
        ds_paths = [ds_paths]
    out = []
    for ds in ds_paths:
        try:
            with h5py.File(ds, "r") as f:
                data = f["data"]
                keys = sorted(data.keys(), key=lambda x: int(x.split("_")[-1]))
                for k in keys:
                    ep = data[k].attrs["ep_meta"]
                    ep = json.loads(ep) if isinstance(ep, (str, bytes)) else dict(ep)
                    out.append((ep, int(data[k]["actions"].shape[0])))
        except Exception as e:
            logging.warning("could not load ep_metas from %s: %r", ds, e)
    return out or None


def _build_state(obs):
    """16-dim EEF-first state matching the checkpoint's norm_stats (verified)."""
    return np.concatenate([
        np.asarray(obs["robot0_base_to_eef_pos"]).reshape(-1),   # 3
        np.asarray(obs["robot0_base_to_eef_quat"]).reshape(-1),  # 4
        np.asarray(obs["robot0_base_pos"]).reshape(-1),          # 3
        np.asarray(obs["robot0_base_quat"]).reshape(-1),         # 4
        np.asarray(obs["robot0_gripper_qpos"]).reshape(-1),      # 2
    ], axis=0).astype(np.float64)


def _img(obs, key, size):
    # [::-1] vertical flip REQUIRED: pixel comparison against the ACTUAL unified/memory_lerobot
    # training frames (extracted via GrootOpenpiSingleDataset) shows the training videos are
    # stored FLIPPED relative to the clean env's raw obs/render. Flipped MAE=10-16 vs raw MAE=44-82;
    # top/bot brightness confirms flip matches training orientation. Applies to obs-dict cams too.
    im = np.asarray(obs[key])[::-1]
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(im, size, size))


def _img_render(env, camera_name, size, render_res=256):
    """Render a mujoco camera the ReMemBench demo did NOT record (agentview_left/right, and
    eye_in_hand for parity), flip [::-1] to match training orientation, and reproduce the
    TRAINING supersample path: render at 256 (the offline re-render resolution the model
    trained on, per convert_memory_to_lerobot.py IMG=(256,256,3)) THEN resize_with_pad to the
    policy input size (224). Rendering natively at 224 (old path) produced coarser aliasing
    -> ~MAE-10 high-freq edge gap vs training that tripped the nav/manip mode gate."""
    im = np.asarray(env.sim.render(height=render_res, width=render_res, camera_name=camera_name))[::-1]
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(im, size, size))


def _action(a):
    return np.asarray(a, dtype=np.float64)[:12]  # robosuite-native order, pass-through


def _mem_views(env, mem_size, render_res=256):
    """The 3 TRAINING memory views in the store's view_order:
    [agentview_left, agentview_right, eye_in_hand], rendered@256 (native) + [::-1] flip, then
    resize_with_pad to mem_size. The MEMORY streams (Wan latents + recent-SigLIP) were built at
    128px (store recent_siglip=64 tok/view, latents=3*64=192 -> 128->8x8=64 Wan tokens), so the
    memory buffer MUST be fed 128px -- NOT the policy's 224. The WanStreamer encodes at input res
    (no internal resize), so feeding 224 would give 196 tokens != the trained 64 (wrong + slow)."""
    cams = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    views = []
    for c in cams:
        im = np.asarray(env.sim.render(height=render_res, width=render_res, camera_name=c))[::-1]
        views.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(im, mem_size, mem_size)))
    return np.ascontiguousarray(np.stack(views, axis=0))  # (3, mem_size, mem_size, 3)


def _push_memory(client, env, obs, mem_size, prev_action=None):
    """Feed the current 3-view frame + 16-dim openpi state + prev executed action into the
    server memory buffer. num_views=3 (store view_order [left,right,eye_in_hand]). Frames at
    mem_size=128 (memory resolution), decoupled from the policy's 224 VLA element.

    Action: RoboMMEOutputs emits openpi order [eef,rot,grip,base,ctrl] = the order GrootOpenpi
    stored the memory action stream, so the executed env action is sent AS-IS (no reorder).
    First frame -> zeros (no prior action). Buffer applies its own raw/delta encoding."""
    state = _build_state(obs)
    mem_img = _mem_views(env, mem_size)                # (3, mem_size, mem_size, 3)
    images = mem_img[None, ...]                        # (1, 3, mem_size, mem_size, 3)
    action_dim = 12
    pa = (np.zeros((1, action_dim), np.float32) if prev_action is None
          else np.asarray(prev_action, np.float32).reshape(1, -1)[:, :action_dim])
    client.add_buffer({"images": images, "state": state[None, ...], "actions": pa})
    return state


def eval_env(env_name, args):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"{args.log_dir}/evals_clean/{env_name}/{now}"
    # skip if already done
    base = f"{args.log_dir}/evals_clean/{env_name}"
    if os.path.isdir(base):
        for d in os.listdir(base):
            if os.path.exists(os.path.join(base, d, "stats.json")):
                print(f"{env_name} stats exist, skipping", flush=True)
                return
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    client = MMEClient(args.host, args.port)
    ds_paths = _datasets_for(env_name)
    ds = ds_paths[0]
    ea = EnvArgs(robots="PandaOmron", controller="OSC_POSE", use_camera_obs=True, control_freq=20)
    ret = make_env(ds, ea)
    env = ret[0] if isinstance(ret, tuple) else ret  # make_env returns (env, kwargs)

    task_horizon = args.fixed_horizon if args.fixed_horizon > 0 else _adaptive_horizon(ds, args.max_horizon)

    # OBJECT VARIETY (Rutav's protocol): make_env pins demo-0's ep_meta and replays it every
    # reset, freezing the scene. HALO's casa_rollouts.py loads EVERY demo's ep_meta and cycles
    # ep_metas[(i+seed) % N] per episode -> per-demo object instance/placement/target-count
    # variety. IDENTICAL pool + sort + seed convention as the pi0.5 client, so both evals see
    # the SAME demo IDs per episode (fair head-to-head).
    all_ep_metas = _load_all_ep_metas(ds_paths) if args.resample_objects else None
    if all_ep_metas:
        logging.info("[%s] cycling %d demo ep_metas (across %d files) for object variety",
                     env_name, len(all_ep_metas), len(ds_paths))
    # Per-episode horizon: size each episode to its cycled demo's own action_len (ceil*1.5,
    # cap), so short targets (fewer bread/bowl) get a smaller budget. Matches pi0.5 client.
    per_ep_h = args.per_episode_horizon and bool(all_ep_metas) and args.fixed_horizon <= 0

    def _reset_env(ep_idx):
        alen = 0
        if all_ep_metas and hasattr(env, "set_ep_meta"):
            try:
                ep_meta, alen = all_ep_metas[ep_idx % len(all_ep_metas)]
                env.set_ep_meta(copy.deepcopy(ep_meta))
            except Exception as e:
                logging.warning("set_ep_meta(demo %d) failed: %r", ep_idx, e)
        return env.reset(), alen

    logging.info("[%s] mode=mem task_horizon=%d per_episode_horizon=%s resample=%s", env_name,
                 task_horizon, per_ep_h, args.resample_objects)

    total, succ = 0, 0
    for ep in tqdm.tqdm(range(args.num_trials), desc=f"{env_name}[mem]"):
        # offset by seed so different shards cover different demos (sharded eval)
        obs, demo_len = _reset_env(ep + args.seed)
        horizon = int(min(math.ceil(demo_len * 1.5), args.max_horizon)) if (per_ep_h and demo_len > 0) else task_horizon
        # lang can change per episode once objects re-sample; refetch each reset.
        try:
            lang = env.get_ep_meta().get("lang") or ""
        except Exception:
            lang = ""
        # TRAIN PARITY (casing differs by consumer, verified byte-exact vs store/train tokens):
        #  * VLA prompt: build_robomme_dataset lowercases (task_goal.lower()) -> VLA tokens trained
        #    on LOWERCASE ("wash the fruit..."). So the element prompt must be lowercased.
        #  * MEMORY lang_emb: augment_store_lang built the store lang_emb from episodes.jsonl
        #    tasks[0] RAW (capitalized "Wash the fruit..."). So memory reset must get the RAW cap
        #    string. Feeding lowercase to memory gives max_abs_diff 1.19 vs store (WRONG).
        lang_mem = lang                # capitalized -> matches store lang_emb byte-exact
        lang_vla = lang.lower()        # lowercase   -> matches VLA training tokens
        plan = collections.deque()
        frames = []
        done = False
        steps = 0

        client.reset(prompt=(lang_mem or None))   # memory-FiLM lang = RAW capitalized (matches store lang_emb)
        _prev_a = None   # prev executed action (openpi order); None -> zeros on frame 0

        def _record(o):
            if args.save_video:
                img = np.asarray(env.sim.render(height=256, width=256,
                                                camera_name="robot0_agentview_center"))[::-1]
                frames.append(np.ascontiguousarray(img).astype(np.uint8))

        # ---------- learned policy (memory buffer fed every step) ----------
        while steps < horizon and not done:
            state = _push_memory(client, env, obs, args.mem_size, prev_action=_prev_a)
            if args.save_video and steps % 2 == 0:
                _record(obs)

            if not plan:
                # BASE = agentview_left, RIGHT = agentview_right (the model's training views,
                # per groot loader: observation/image<-video.robot0_agentview_left,
                # observation/right_image<-video.robot0_agentview_right). The ReMemBench demos
                # only record center+wrist, so left/right are sim-rendered. ALL cams flipped
                # [::-1] to match training orientation (verified by pixel-compare vs actual
                # unified/memory_lerobot training frames). wrist = eye_in_hand (also flipped).
                # All 3 views: render@256 -> resize_with_pad 224, flipped -- matching the
                # offline 256 3-cam re-render the model trained on. Wrist also rendered @256
                # (env exposes robot0_eye_in_hand cam) for parity vs the recorded 128px obs.
                element = {
                    "observation/image": _img_render(env, "robot0_agentview_left", args.resize_size),
                    "observation/wrist_image": _img_render(env, "robot0_eye_in_hand", args.resize_size),
                    "observation/right_image": _img_render(env, "robot0_agentview_right", args.resize_size),
                    "observation/state": state,
                    "prompt": lang_vla or "",
                }
                chunk = client.infer(element)["actions"]
                plan.extend(chunk[: args.replan_steps])

            a = np.asarray(plan.popleft())
            _prev_a = np.asarray(a, dtype=np.float32)[:12]   # executed action -> memory next step (openpi order)
            obs, r, d, info = env.step(_action(a))
            steps += 1
            if env._check_success():
                done = True
                break

        if done:
            succ += 1
        total += 1
        if args.save_video and frames:
            sfx = "success" if done else "failure"
            imageio.mimwrite(os.path.join(log_path, f"rollout_{ep}_{sfx}.mp4"), frames, fps=20)
        logging.info("[%s] ep %d/%d success=%s steps=%d SR=%.1f%%", env_name, total,
                     args.num_trials, done, steps, 100.0 * succ / total)

    sr = succ / total if total else 0.0
    logging.info("[%s] FINAL success_rate=%.4f (%d/%d) mode=mem", env_name, sr, succ, total)
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        json.dump({"env_name": env_name, "num_episodes": total,
                   "num_successes": succ, "success_rate": sr,
                   "explore_mode": "mem",
                   "per_episode_horizon": per_ep_h, "task_horizon": task_horizon,
                   "max_horizon": args.max_horizon,
                   "resample_objects": args.resample_objects,
                   "num_demo_ep_metas": len(all_ep_metas) if all_ep_metas else 0,
                   "object_variety": "cycle_demo_ep_metas" if args.resample_objects else "frozen_demo0",
                   "split": "remembench_native_memory"}, f, indent=4)
    try:
        env.close()
    except Exception:
        pass


def main(args: Args):
    np.random.seed(args.seed)
    envs = [e.strip() for e in args.env_names.split(",") if e.strip()]
    assert envs and args.log_dir
    logging.info("clean-env MEMORY eval on port %d: %s", args.port, envs)
    for e in envs:
        eval_env(e, args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
