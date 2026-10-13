"""Action streams aligned to a selected latent variant's local timebase."""

from pathlib import Path

import numpy as np


def load_variant_causal_actions(epi_dir, variant, frame_idx, raw_actions):
    """Return previous sampled raw actions for a variant's prefix.

    Newer stores materialize this stream as ``action_causal_v{variant}.npy``.
    Older var8 stores derive the same result from the selected raw-action
    sequence, which avoids indexing a canonical causal stream at skipped frames.
    """
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    n_frames = len(frame_idx)
    path = Path(epi_dir) / f"action_causal_v{int(variant)}.npy"
    if path.is_file():
        actions = np.load(path, allow_pickle=False)
        if len(actions) < n_frames:
            raise ValueError(
                f"{path} is shorter than selected variant prefix: "
                f"{len(actions)} < {n_frames}"
            )
        return np.ascontiguousarray(actions[:n_frames]).astype(np.float32)

    sampled_raw = np.ascontiguousarray(raw_actions[frame_idx]).astype(np.float32)
    actions = np.zeros_like(sampled_raw)
    actions[1:] = sampled_raw[:-1]
    return actions
