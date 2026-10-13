# Evaluating Stage-2 Memory VLA (washreturn) — Instructions

## Overview
Two-process eval on the CLEAN ShahRutav ReMemBench env:
- **Server** (`scripts/serve_policy_mme.py`, openpi venv) loads ONE checkpoint that contains the
  trained VLA + FiLM seam + FROZEN Stage-1 bottleneck, and serves reset/add_buffer/infer.
- **Client** (`eval/eval_clean_mme3.py`, remembench_clean venv) drives the env, feeds memory,
  calls infer, records videos, writes stats.json.

Memory model = `mme_wan_stage2_finetuned` (ft) or `mme_wan_stage2_base` (base).
No-memory baseline = openpi core `scripts/train.py` config `pi05_washreturn_toy` (different client:
`eval/eval_clean_nomem.py`, no memory protocol).

## Ground truth (VERIFIED byte-exact vs training store)
- **3 cameras** (washreturn toy trained on these; env only records center+wrist so left/right are
  `env.sim.render(camera_name=...)`, ALL flipped `[::-1]`):
    - `observation/image`      = robot0_agentview_left   (VLA base_0_rgb)
    - `observation/wrist_image`= robot0_eye_in_hand       (VLA left_wrist_0_rgb)
    - `observation/right_image`= robot0_agentview_right   (VLA right_wrist_0_rgb)
- **Resolutions**: env renders 256 -> VLA element resize_with_pad **224**; MEMORY buffer fed **128**
  (Wan latents + recent-SigLIP were built at 128; 8x8=64 tok/view, pooled pool_hw=2 -> 4/view).
- **State**: 16-dim openpi order [eef_pos3, eef_quat4, base_pos3, base_quat4, gripper2].
- **Action** to memory: prev EXECUTED action, openpi order [eef,rot,grip,base,ctrl]=12d, NO reorder.
- **Language (casing differs by consumer!)**:
    - VLA prompt = LOWERCASE ("wash the fruit and return it to the container.") -- build_robomme_dataset .lower()
    - Memory lang_emb = RAW CAPITALIZED ("Wash...") -- augment_store_lang used episodes.jsonl tasks[0]
    - eval_clean_mme3.py sets lang_vla=lang.lower() (element prompt) and lang_mem=lang (client.reset prompt).
- **Buffer parity**: downsample=2, pool_hw=2, fixed-cap pad (LATENT_CAP16384/MAX_RECENT2048/MAX_STEPS800)+masks.
  cosmo_buffer applies these; matches training gather byte-exact (action 0.0, latent shape/count match).

## Required code fixes (all committed here except openpi core)
- policy.py: raw_action_dim (no //2 for raw), _prepare_history fixed-cap pad + masks, pool_hw pass, reset(prompt).
- cosmo_buffer.py: pool_hw + _pool_latents (8x8->2x2), action/proprio append gated by downsample.
- policy_config.py + serving/websocket_policy_server.py: reset forwards per-episode prompt; int-key coercion (redundant).
- eval/eval_clean_mme3.py: 3-cam _img_render element, 128px _mem_views, split-casing lang.
- **openpi CORE (NOT git-tracked)**: apply patches/openpi_model_intkey_fix.md to src/openpi/models/model.py
  BEFORE serving any memory checkpoint (else checkpoint load fails: sharding/pytree key mismatch).

## How to run (per checkpoint, per task)
Server uses the NON-precompute yaml (cosmo-wan-bottleneck-modul-s2nojitter.yaml) -> LIVE bottleneck
(eval episodes are novel; there is NO precompute cache at eval). fsdp not needed for serving.

```bash
# env for the checkpoint's variant:
#   ft*  -> STORE=washreturn_store_var8       PI05=pi05_washreturn_toy_10k/9999/params  CONFIG=mme_wan_stage2_finetuned
#   base*-> STORE=washreturn_store_var8_base  PI05=pi05_base/params                     CONFIG=mme_wan_stage2_base
# STAGE1_CKPT = /path/to/checkpoints/stage1_<var>/stage1_step5000.npz

# 1) SERVER (openpi venv), one GPU, a FRESH PORT (never reuse a port across failed launches -> zombie hang):
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OPENPI_DATA_HOME=/path/to/openpi-cache HF_HUB_OFFLINE=1 \
  STAGE1_CKPT=$S1 STAGE2_STORE=$STORE PI05_PARAMS=$PI \
  python scripts/serve_policy_mme.py --ckpt-dir=$CKPT_STEP_DIR --config=$CONFIG \
    --history-config=cosmo-wan-bottleneck-modul-s2nojitter.yaml --port=$PORT
# WAIT for "server listening on 0.0.0.0:$PORT" AND verify GPU mem > 0 (a real server allocates ~160GB;
# a zombie shows "connection open" but 0 MiB and never serves infer -> client hangs in _ws.recv).

# 2) CLIENT (remembench_clean venv), run from /tmp (namespace shadow), same GPU:
cd /tmp; MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU \
  <clean_venv>/bin/python <repo>/eval/eval_clean_mme3.py \
    --host 127.0.0.1 --port $PORT --num-trials 10 --max-horizon 1000 --fixed-horizon 1000 \
    --env-names MemWashAndReturnLeft --log-dir <OUT> --save-video
```
Tasks (3): MemWashAndReturnLeft, MemWashAndReturnRight, MemWashAndReturnSameLocation.
Steady-state ~0.15s/step (memory add_buffer ~1ms via O(T) Wan streamer; infer ~83ms). ~2.5min/10-trial-ep-set.

## Output
stats.json per (ckpt,task): success_rate, num_successes/episodes. Videos rollout_<ep>_<success|failure>.mp4.
Upload to your configured W&B project via the OpenPI environment.
