"""RoboDojo (bimanual joint-space) input/output transforms for the ROBOMME memory VLA.

Mirrors mme_vla_suite.policies.robomme_policy (robocasa 12-dim) but retargets to
RoboDojo's schema (from HF meta/info.json):
  - action:            [14]  left_joint_0..6 + right_joint_0..6   (bimanual, 7 DOF/arm)
  - observation.state: [14]  same joint layout
  - cameras (all 3x480x640): cam_high, cam_left_wrist, cam_right_wrist, cam_wrist

Camera mapping into pi0.5's 3 image slots:
  cam_high        -> base_0_rgb        (overhead/base view)
  cam_left_wrist  -> left_wrist_0_rgb
  cam_right_wrist -> right_wrist_0_rgb
  (cam_wrist is the single-arm fallback view; UNUSED here — we keep the dual-wrist views,
   which is a strictly richer 3-view than robocasa's zero-padded right slot.)

Action/state are joint-space and NOT delta — pi0.5 handles absolute joint targets via
its own norm stats, so (as with robocasa's already-relative actions) we apply NO
DeltaActions transform here. The model action space is a fixed 32-dim pad
(PadStatesAndActions(32) in ModelTransformFactory); native 14 -> padded to 32, and
RoboDojoOutputs slices back to the first 14 dims at inference.
"""
import dataclasses

import einops
import numpy as np

import openpi.models.model as _model
import openpi.transforms as transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        assert np.abs(image).max() <= 1.0, "Image is not normalized"
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (c,h,w) -> (h,w,c)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# RoboDojo bimanual joint action/state dimension (7 per arm x 2 arms).
ROBODOJO_ACTION_DIM = 14


@dataclasses.dataclass(frozen=True)
class RoboDojoInputs(transforms.DataTransformFn):
    """Convert RoboDojo elements into the model input dict. Used for train AND inference.

    Identical memory-stream passthrough (static_/recur_/cosmo_/aux_) to RoboMMEInputs so the
    cosmo-bottleneck and E2E+aux paths plug in unchanged; only the image keys, state, and action
    dim differ from robocasa.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])            # cam_high
        left_wrist = _parse_image(data["observation/wrist_image"])      # cam_left_wrist
        right_wrist = _parse_image(data["observation/right_image"])     # cam_right_wrist

        inputs = {
            "state": data["observation/state"],  # (14,), padded to 32 by ModelTransformFactory
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            # perceptual memory
            "static_image_emb": data.get("static_image_emb", None),
            "static_pos_emb": data.get("static_pos_emb", None),
            "static_state_emb": data.get("static_state_emb", None),
            "static_mask": data.get("static_mask", None),
            "static_time": data.get("static_time", None),
            # recurrent memory
            "recur_image_emb": data.get("recur_image_emb", None),
            "recur_pos_emb": data.get("recur_pos_emb", None),
            "recur_state_emb": data.get("recur_state_emb", None),
            "recur_mask": data.get("recur_mask", None),
            # cosmo_bottleneck memory (offline latents_v* path)
            "cosmo_latent_emb": data.get("cosmo_latent_emb", None),
            "cosmo_mem_mask": data.get("cosmo_mem_mask", None),
            "cosmo_recent_emb": data.get("cosmo_recent_emb", None),
            "cosmo_recent_mask": data.get("cosmo_recent_mask", None),
            "cosmo_action_emb": data.get("cosmo_action_emb", None),
            "cosmo_action_mask": data.get("cosmo_action_mask", None),
            "cosmo_proprio_emb": data.get("cosmo_proprio_emb", None),
            "cosmo_proprio_mask": data.get("cosmo_proprio_mask", None),
            "cosmo_lang_emb": data.get("cosmo_lang_emb", None),
            "cosmo_mem_tokens": data.get("cosmo_mem_tokens", None),
            # E2E+aux reconstruction targets
            "aux_query_steps": data.get("aux_query_steps", None),
            "aux_query_mask": data.get("aux_query_mask", None),
            "aux_align_target": data.get("aux_align_target", None),
            "aux_action_target": data.get("aux_action_target", None),
            "aux_proprio_target": data.get("aux_proprio_target", None),
            "aux_reward_target": data.get("aux_reward_target", None),
            # symbolic memory
            "simple_subgoal": data.get("simple_subgoal", None),
            "grounded_subgoal": data.get("grounded_subgoal", None),
            "sample_loss_weight": data.get("sample_loss_weight", None),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]  # (h, 14), padded to 32 downstream
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboDojoOutputs(transforms.DataTransformFn):
    """Slice the 32-dim model action back to RoboDojo's native 14-dim bimanual joint action."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROBODOJO_ACTION_DIM])}
