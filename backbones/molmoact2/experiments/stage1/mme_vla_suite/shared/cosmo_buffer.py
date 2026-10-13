"""Online dual-stream memory buffer for Option D: consolidated Cosmos latents + recent SigLIP tokens.

DROP-TERMINAL: only finalized (non-terminal) Cosmos chunks are consolidated; the newest
complete-but-provisional chunk stays in the recent-SigLIP window until a newer chunk exists.

A chunk of chunk_frames downsampled frames becomes FINALIZED as soon as at least one
frame of the NEXT chunk has arrived (that makes the earlier chunk non-terminal, hence
bit-stable). Concretely, after appending a frame, while len(_recent_frames) > chunk_frames
we pop the oldest chunk_frames frames and append the finalized latent for that chunk to
_consolidated. CAUSAL: the finalized latent is the c-th chunk of a causal encode over the
GROWING PREFIX of every downsampled frame seen so far (matching the offline encode-once store
builder), NOT the popped chunk encoded standalone -- standalone re-encoding discards the
strictly-causal temporal context and gives the wrong latent for every non-leading chunk.
The recent-SigLIP window therefore spans every frame
since the last finalized boundary (up to 2*chunk_frames-1 frames: the most recent complete
provisional chunk plus the current partial chunk).

Action encoding modes (config-selectable via action_encoding):
  * "raw_delta" (DEFAULT, backward compatible): cosmo_action_emb = concat(action, Δaction),
    width = 2*action_dim (=24 for the 12-dim robocasa action). Δ[i]=action[i]-action[i-1].
  * "geometric" (Cosmos-3 style): reframe the 12-dim robocasa action
    [eef_pos(3), eef_rot_rotvec(3), gripper(1), base_motion(4), control_mode(1)] into a single
    15-dim geometric vector [eef_pos_delta(3), eef_rot_6d(6), gripper_abs(1), base(4), mode(1)].
    The 3-dim eef_rot rotation-vector DELTA is converted to a continuous 6D rotation
    representation (Zhou et al. 2019 = first two columns of the rotation matrix). Grasp and
    control_mode pass through as absolute/current values. Unlike raw_delta, geometric mode does
    NOT append a separate Δ concat: the relative pose already encodes motion (Cosmos-3 scheme),
    so the width is 15, not 30.
"""
import numpy as np
import einops
import jax
from openpi.shared import image_tools


def rotvec_to_6d(rotvec):
    """Convert an axis-angle rotation vector (3,) to a continuous 6D rotation (Zhou 2019).

    rotvec -> R (3x3) via Rodrigues exp-map -> first two columns of R, laid out
    column-major as [col0(3), col1(3)] = R[:, :2].reshape(-1, order="F").
    Identity (theta~0) returns [1,0,0, 0,1,0].
    """
    rotvec = np.asarray(rotvec, np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        R = np.eye(3, dtype=np.float64)
    else:
        k = rotvec / theta
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]], dtype=np.float64)
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R[:, :2].reshape(-1, order="F").astype(np.float32)


class CosmoMemoryBuffer:
    def __init__(self, cosmo, siglip_fn, chunk_frames=8, downsample=2, num_views=1,
                 state_dim=16, action_dim=12, latent_dim=16, max_steps=2048,
                 action_encoding="raw_delta", proprio_encoding="raw_delta", pool_hw=None):
        self.cosmo = cosmo
        self.siglip_fn = siglip_fn
        self.chunk_frames = chunk_frames
        self.downsample = downsample
        self.num_views = num_views
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        if action_encoding not in ("raw_delta", "geometric", "raw"):
            raise ValueError(f"unknown action_encoding: {action_encoding!r}")
        self.action_encoding = action_encoding
        if proprio_encoding not in ("raw_delta", "raw"):
            raise ValueError(f"unknown proprio_encoding: {proprio_encoding!r}")
        self.proprio_encoding = proprio_encoding
        # FIX(eval parity): pool consolidated Wan latents 8x8->pool_hw x pool_hw per view,
        # EXACTLY as training dataset._gather_cosmo_bottleneck does (pool_hw=2 -> 4 tok/view).
        self.pool_hw = int(pool_hw) if pool_hw else None
        self.clear()

    def clear(self):
        self._consolidated = []      # finalized chunk latents, each (spatial_tokens, latent_dim)
        self._recent_frames = []     # downsampled frames since last FINALIZED boundary
        self._all_ds_frames = []     # ALL downsampled frames (kept so finalized chunks can be
                                     # re-encoded over the GROWING PREFIX -- causal consolidation)
        self._states = []            # every step state
        self._actions = []           # every step action
        self._lang_emb = None
        self._seen = 0
        # O(T) streaming cache: if the frontend can make a streamer (Wan), use it to encode
        # ONLY the new chunk each finalization (verified byte-exact vs full re-encode) instead
        # of the O(T^2) full-prefix path. `_stream_all` holds ALL chunks the streamer has
        # emitted (incl. the currently-terminal one); `_consolidated` exposes drop-terminal
        # (== offline store's finalized_encode == encode[:-1]).
        self._streamer = self.cosmo.make_streamer(self.num_views) if hasattr(self.cosmo, "make_streamer") else None
        self._stream_all = []        # every chunk the streamer emitted (terminal included)
        self._pushed_ds = 0          # how many _all_ds_frames already pushed to the streamer

    def set_language(self, lang_emb):
        self._lang_emb = None if lang_emb is None else np.asarray(lang_emb, np.float32)

    def add_buffer(self, images, states, actions, step_idx_list):
        # images: (t,v,h,w,3) uint8 ; keep num_views views ; downsample by step index
        for i, step in enumerate(step_idx_list):
            # FIX(eval parity): training indexes action/proprio at the DOWNSAMPLED frame_idx
            # (every -th step), NOT every step. Gate states/actions by the SAME
            # downsample as frames so all three streams share the frame_idx cadence.
            if step % self.downsample != 0:
                continue
            self._states.append(np.asarray(states[i], np.float32))
            self._actions.append(np.asarray(actions[i], np.float32))
            self._all_ds_frames.append(np.asarray(images[i, :self.num_views]))  # (V,h,w,3)
        # CAUSAL CONSOLIDATION driven off the frontend's OWN finalized-chunk count (single
        # source of truth -> no cadence/finalized_encode off-by-one). Both Wan and Cosmos are
        # strictly causal, so chunk c's latent is the c-th chunk of a causal encode over the
        # growing prefix -- byte-for-byte the offline store builder's encode-once path. We
        # commit any chunks that finalized_encode now reports beyond what we've committed.
        # Gate on floor(frames/cf) > committed (necessary for a new chunk under BOTH the real
        # frontend's warmup+terminal drop accounting and the plain fake accounting) so we only
        # re-encode when a new chunk could exist. NOTE O(T^2): re-encodes the whole prefix;
        # a stateful causal cache (tracked separately) restores O(T).
        n_ds = len(self._all_ds_frames)
        if self._streamer is not None:
            # O(T) path: push only the NEW downsampled frames; streamer emits new chunks.
            if n_ds > self._pushed_ds:
                new = np.stack(self._all_ds_frames[self._pushed_ds:], 0)   # (k, V, h, w, 3)
                self._pushed_ds = n_ds
                self._stream_all.extend(self._streamer.push(new))
            # drop-terminal: commit all but the currently-terminal chunk (== offline finalized_encode).
            n_finalized = max(0, len(self._stream_all) - 1)
            for c in range(len(self._consolidated), n_finalized):
                self._consolidated.append(np.asarray(self._stream_all[c], np.float32))
        elif n_ds // self.chunk_frames > len(self._consolidated):
            # O(T^2) fallback (Cosmos / test fakes without a streamer): full-prefix re-encode.
            finalized = self._finalized_encode_all()          # (num_finalized, V*spatial, latent)
            for c in range(len(self._consolidated), int(finalized.shape[0])):
                self._consolidated.append(np.asarray(finalized[c], np.float32))
        # recent window = downsampled frames after the last finalized chunk boundary
        self._recent_frames = self._all_ds_frames[len(self._consolidated) * self.chunk_frames:]
        self._seen = len(self._states)

    def _finalized_encode_all(self):
        """Causally encode the FULL growing prefix of downsampled frames seen so far and
        return per-chunk latents: (num_chunks, V*spatial, latent). Chunk c is bit-stable.

        For real frontends (WanFrontend/CosmoFrontend) this calls finalized_encode_multiview /
        finalized_encode -- literally the SAME code the OFFLINE store builder uses -- so the
        online consolidated chunks equal the offline encode-once chunks byte-for-byte. The
        buffer only ever indexes chunk c once it has been popped (drop-terminal), and the pop
        cadence matches finalized_encode's chunk count, so index c is always in range.

        Because it re-encodes the whole prefix on every finalization, consolidating a T-frame
        episode is O(T^2) VAE calls; acceptable for correctness (see buffer docstring / report).
        num_views==1 with a frontend exposing only encode() falls back to the single-view path."""
        frames = np.stack(self._all_ds_frames, 0)             # (t, V, h, w, 3)
        V = frames.shape[1]
        # Prefer the frontend's own causal drop-terminal methods when present so the online
        # path is literally the same code as the offline store builder.
        if V > 1 and hasattr(self.cosmo, "finalized_encode_multiview"):
            return self.cosmo.finalized_encode_multiview(frames)
        if hasattr(self.cosmo, "finalized_encode"):
            return self.cosmo.finalized_encode(frames[:, 0])
        # Backward-compat fallback for encoders exposing only encode()/encode_multiview()
        # (e.g. test fakes). encode() returns ALL complete chunks with no drop-terminal, so we
        # drop the terminal (provisional) chunk here to match the real frontends' finalized_encode
        # accounting (finalized count = complete_chunks - 1). This keeps the consolidation cadence
        # identical whether driven by a real frontend or a fake, eliminating the off-by-one that
        # caused the IndexError regression.
        raw = (self.cosmo.encode_multiview(frames) if (V > 1 and hasattr(self.cosmo, "encode_multiview"))
               else self.cosmo.encode(frames[:, 0]))
        return raw[:-1] if raw.shape[0] > 0 else raw

    def _encode_recent(self):
        if not self._recent_frames:
            return np.zeros((0, 2048), np.float32)
        # _recent_frames entries are (V,h,w,3)
        x = np.stack(self._recent_frames, 0)                  # (t, V, h, w, 3)
        V = x.shape[1]
        xj = (x.astype(np.float32) / 255.0 * 2.0 - 1.0)
        xj = einops.rearrange(xj, "t v h w c -> (t v) h w c")
        xj = image_tools.resize_with_pad(xj, 224, 224)
        xj = einops.rearrange(xj, "(t v) h w c -> t v h w c", v=V)
        emb = np.asarray(jax.device_get(self.siglip_fn(xj)))  # (t, V, siglip_tokens, dim)
        return emb.reshape(-1, emb.shape[-1])                 # (t*V*siglip_tokens, dim)

    @staticmethod
    def _with_delta(arr):
        if arr.shape[0] == 0:
            return np.zeros((0, arr.shape[1] * 2), np.float32)
        delta = np.zeros_like(arr)
        delta[1:] = arr[1:] - arr[:-1]
        return np.concatenate([arr, delta], axis=1).astype(np.float32)

    @staticmethod
    def _geometric_actions(arr):
        """Reframe robocasa 12-dim actions into Cosmos-3-style 15-dim geometric vectors.

        Input layout per step: [eef_pos(3), eef_rot_rotvec(3), gripper(1), base(4), mode(1)].
        Output layout per step: [eef_pos_delta(3), eef_rot_6d(6), gripper_abs(1), base(4), mode(1)].
        No separate Δ concat: the relative pose already encodes motion (Cosmos-3 scheme).
        """
        if arr.shape[0] == 0:
            return np.zeros((0, 15), np.float32)
        out = np.zeros((arr.shape[0], 15), np.float32)
        for i in range(arr.shape[0]):
            a = arr[i]
            out[i, 0:3] = a[0:3]                       # eef_pos delta passthrough
            out[i, 3:9] = rotvec_to_6d(a[3:6])         # rotvec -> 6D rotation
            out[i, 9] = a[6]                           # gripper absolute
            out[i, 10:14] = a[7:11]                    # base_motion passthrough
            out[i, 14] = a[11]                         # control_mode passthrough
        return out

    def _pool_latents(self, chunks):
        """Pool each finalized chunk's Wan latents 8x8 -> pool_hw x pool_hw per view, EXACTLY as
        dataset._gather_cosmo_bottleneck (pool_hw=2 -> 4 tok/view). chunks: list of (V*64, d).
        Returns (num_chunks*V*pool_hw^2, d) flattened, matching the training latent stream."""
        if not chunks or not self.pool_hw:
            return (np.concatenate(chunks, 0) if chunks
                    else np.zeros((0, self.latent_dim), np.float32))
        lat = np.stack(chunks, 0)                       # (nc, V*64, d)
        nc = lat.shape[0]; d = lat.shape[-1]
        spatial = 64; Vv = lat.shape[1] // spatial
        side = 8; ph = int(self.pool_hw); blk = side // ph
        assert lat.shape[1] % 64 == 0, f"unexpected spatial layout {lat.shape}"
        assert 8 % ph == 0, f"pool_hw {ph} must divide 8"
        gg = lat.reshape(nc, Vv, side, side, d)
        gg = gg.reshape(nc, Vv, ph, blk, ph, blk, d).mean(axis=(3, 5))
        return gg.reshape(nc * Vv * ph * ph, d).astype(np.float32)

    def get_history_feats(self, step_idx):
        cosmo = self._pool_latents(self._consolidated)
        recent = self._encode_recent()
        states = np.stack(self._states, 0) if self._states else np.zeros((0, self.state_dim), np.float32)
        actions = np.stack(self._actions, 0) if self._actions else np.zeros((0, self.action_dim), np.float32)
        if self.action_encoding == "geometric":
            action_emb = self._geometric_actions(actions)
        elif self.action_encoding == "raw":
            action_emb = actions.astype(np.float32)          # action only, no delta (12d)
        else:
            action_emb = self._with_delta(actions)           # raw_delta (24d)
        return {
            "cosmo_latent_emb": cosmo.astype(np.float32),
            "cosmo_recent_emb": recent.astype(np.float32),
            "cosmo_mem_mask": np.ones((cosmo.shape[0],), bool),
            "cosmo_proprio_emb": (states.astype(np.float32) if self.proprio_encoding == "raw"
                                  else self._with_delta(states)),
            "cosmo_action_emb": action_emb,
            "cosmo_lang_emb": self._lang_emb,
        }
