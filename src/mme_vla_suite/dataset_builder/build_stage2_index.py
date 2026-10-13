"""Build the Stage-2 pkl sample index (data/*.pkl + meta/stats.json) for the washreturn
stores. Stage-2's SampleDataset reads data/{idx}.pkl; RoboMMEDataset._gather_cosmo_bottleneck
then pulls checkpoint-specific features by (epis_idx, step_idx). The index is
checkpoint-INDEPENDENT (state/action/image/prompt identical for ft & base), so build ONCE and
symlink into both stores.

pkl frame_dict carries all 3 camera views at native 256px (VLA transforms resize to 224):
  image=view0 (agentview_left/front), wrist_image=view2 (eye_in_hand),
  right_image=view1 (agentview_right).
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd


def _episode_prompt_map(lerobot_dir):
    m = {}
    with open(os.path.join(lerobot_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            m[int(d["episode_index"])] = d["tasks"][0]  # generic instruction (no L/R leak)
    return m


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot", required=True)
    p.add_argument("--frames-store", required=True,
                   help="store whose features/episode_*/frames.npy provides the 3-view images.")
    p.add_argument("--out", required=True)
    p.add_argument("--action-horizon", type=int, default=20)
    args = p.parse_args(argv)

    data_dir = os.path.join(args.out, "data")
    meta_dir = os.path.join(args.out, "meta")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    prompts = _episode_prompt_map(args.lerobot)
    H = int(args.action_horizon)
    sample_id = 0
    parquet_dir = os.path.join(args.lerobot, "data", "chunk-000")
    eps = sorted(int(f.split("episode_")[1].split(".")[0])
                 for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    for ep in eps:
        df = pd.read_parquet(os.path.join(parquet_dir, f"episode_{ep:06d}.parquet"))
        state = np.stack(df["observation.state"].values).astype(np.float32)   # (T,16) RAW GROOT
        # FIX(state-order): parquet observation.state is GROOT/modality order
        # [base_pos3, base_rot4, eef_pos3, eef_rot4, gripper2]. The pi0.5 policy + norm_stats
        # expect the unified openpi order [eef_pos3, eef_rot4, base_pos3, base_rot4, gripper2]
        # (matches GrootOpenpiSingleDataset state compose + eval _build_state). Reading raw
        # parquet put base values into the eef norm slots -> Normalize blew up (std ~2.4e5).
        state = np.concatenate(
            [state[:, 7:10], state[:, 10:14], state[:, 0:3], state[:, 3:7], state[:, 14:16]], axis=-1
        ).astype(np.float32)  # -> [eef_pos3, eef_rot4, base_pos3, base_rot4, gripper2]
        action = np.stack(df["action"].values).astype(np.float32)             # (T,12) RAW GROOT
        # FIX(action-order): parquet action is GROOT order [base4, control1, eef_pos3, eef_rot3, gripper1].
        # The pi0.5 policy + norm_stats expect the unified openpi order [eef_pos3, eef_rot3, gripper1, base4, control1]
        # (matches openpi/training/old_groot_lerobot_dataset.py reorder + RoboMMEOutputs). Reading the parquet
        # action directly (no reorder) put real eef/rot/gripper into the zero-std base/control norm slots ->
        # Normalize divided by ~1e-6 -> action targets exploded (~2.8e4 std, loss ~6e10). Reorder here.
        action = np.concatenate(
            [action[:, 5:8], action[:, 8:11], action[:, 11:12], action[:, 0:4], action[:, 4:5]], axis=-1
        ).astype(np.float32)  # -> [eef_pos3, eef_rot3, gripper1, base4, control1]
        T = len(df)
        frames = np.load(os.path.join(args.frames_store, "features", f"episode_{ep}", "frames.npy"),
                         mmap_mode="r")                                        # (T,3,256,256,3) uint8
        assert frames.shape[0] == T, f"ep{ep}: frames {frames.shape[0]} != parquet {T}"
        prompt = prompts[ep].lower()
        for t in range(T):
            # action chunk [t, t+H); pad the tail with the LAST action (repeat) if near episode end.
            chunk = action[t:t + H]
            if chunk.shape[0] < H:
                pad = np.repeat(action[-1:], H - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            frame_dict = {
                "image": np.ascontiguousarray(frames[t, 0]),        # front  -> base_0_rgb
                "wrist_image": np.ascontiguousarray(frames[t, 2]),  # wrist  -> left_wrist_0_rgb
                "right_image": np.ascontiguousarray(frames[t, 1]),  # right  -> right_wrist_0_rgb
                "state": state[t],
                "actions": chunk,
                "is_demo": np.array([False], dtype=np.bool_),
                "exec_start_idx": np.array([0], dtype=np.int32),
                "step_idx": np.array([t], dtype=np.int32),
                "epis_idx": np.array([ep], dtype=np.int32),
                "prompt": prompt,
                "simple_subgoal": "", "grounded_subgoal": "",
                "simple_subgoal_online": "", "grounded_subgoal_online": "",
            }
            with open(os.path.join(data_dir, f"{sample_id}.pkl"), "wb") as f:
                pickle.dump(frame_dict, f)
            sample_id += 1
        print(f"ep{ep}: {T} samples (total {sample_id})", flush=True)

    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump({"total_samples": sample_id, "execution_samples": sample_id}, f)
    print(f"DONE: {sample_id} pkl samples -> {data_dir}")


if __name__ == "__main__":
    main()
