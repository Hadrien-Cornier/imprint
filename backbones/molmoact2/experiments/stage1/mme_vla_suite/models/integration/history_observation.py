from flax import struct

from openpi.models.model import ArrayT
from openpi.models.model import Observation as _Observation
from openpi.models.model import preprocess_observation as _preprocess_observation
import openpi.shared.array_typing as at


#  b: batch size
#  t: time step
#  v: view
#  h: height
#  w: width
#  c: channel
#  l: seq len


@at.typecheck
@struct.dataclass
class HistAugObservation(_Observation):
    # for perceptual memory
    static_image_emb: at.Float[at.Array, "b l1 d1"] | None = None
    static_mask: at.Bool[at.Array, "b l1"] | None = None
    static_pos_emb: at.Float[at.Array, "b l1 d2"] | None = None
    static_state_emb: at.Float[at.Array, "b l1 d3"] | None = None
    static_time: at.Float[at.Array, "b l1 1"] | None = None  # per-frame elapsed frac (time add-on)

    # for recurrent memory
    recur_image_emb: at.Float[at.Array, "b t v p d1"] | None = None  # left padded
    recur_mask: at.Bool[at.Array, "b t"] | None = None  # left padded
    recur_pos_emb: at.Float[at.Array, "b t v p d2"] | None = None  # left padded
    recur_state_emb: at.Float[at.Array, "b t d3"] | None = None  # left padded

    # for cosmo_bottleneck memory (Option D)
    cosmo_latent_emb: at.Float[at.Array, "b lc dc"] | None = None      # consolidated Cosmos latents
    cosmo_recent_emb: at.Float[at.Array, "b lr dr"] | None = None      # recent per-frame SigLIP tokens
    cosmo_mem_mask: at.Bool[at.Array, "b lc"] | None = None            # validity for consolidated latents
    cosmo_recent_mask: at.Bool[at.Array, "b lr"] | None = None         # validity for recent siglip
    cosmo_action_mask: at.Bool[at.Array, "b la"] | None = None         # validity for action stream
    cosmo_proprio_mask: at.Bool[at.Array, "b lp"] | None = None        # validity for proprio stream
    cosmo_action_emb: at.Float[at.Array, "b la da"] | None = None      # executed action + delta_a (pre-MLP)
    cosmo_proprio_emb: at.Float[at.Array, "b lp dp"] | None = None     # proprio state + delta_p (pre-MLP)
    cosmo_lang_emb: at.Float[at.Array, "b dl"] | None = None           # pooled PaliGemma text emb (FiLM)
    cosmo_mem_tokens: at.Float[at.Array, "b m dt"] | None = None        # PRECOMPUTED frozen-bottleneck tokens (skip mem_encoder)

    # for symbolic memory
    symbolic_tokenized_prompt: at.Int[at.Array, "b l2"] | None = None
    symbolic_tokenized_prompt_mask: at.Bool[at.Array, "b l2"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "HistAugObservation":
        parent_obs = super().from_dict(data)
        return cls(
            # Base observation fields
            images=parent_obs.images,
            image_masks=parent_obs.image_masks,
            state=parent_obs.state,
            tokenized_prompt=parent_obs.tokenized_prompt,
            tokenized_prompt_mask=parent_obs.tokenized_prompt_mask,
            token_ar_mask=parent_obs.token_ar_mask,
            token_loss_mask=parent_obs.token_loss_mask,
            # MMEVLA fields
            static_image_emb=data.get("static_image_emb", None),
            static_mask=data.get("static_mask", None),
            static_pos_emb=data.get("static_pos_emb", None),
            static_state_emb=data.get("static_state_emb", None),
            static_time=data.get("static_time", None),
            recur_image_emb=data.get("recur_image_emb", None),
            recur_mask=data.get("recur_mask", None),
            recur_pos_emb=data.get("recur_pos_emb", None),
            recur_state_emb=data.get("recur_state_emb", None),
            cosmo_latent_emb=data.get("cosmo_latent_emb", None),
            cosmo_recent_emb=data.get("cosmo_recent_emb", None),
            cosmo_mem_mask=data.get("cosmo_mem_mask", None),
            cosmo_recent_mask=data.get("cosmo_recent_mask", None),
            cosmo_action_mask=data.get("cosmo_action_mask", None),
            cosmo_proprio_mask=data.get("cosmo_proprio_mask", None),
            cosmo_action_emb=data.get("cosmo_action_emb", None),
            cosmo_proprio_emb=data.get("cosmo_proprio_emb", None),
            cosmo_lang_emb=data.get("cosmo_lang_emb", None),
            cosmo_mem_tokens=data.get("cosmo_mem_tokens", None),
            symbolic_tokenized_prompt=data.get("symbolic_tokenized_prompt", None),
            symbolic_tokenized_prompt_mask=data.get(
                "symbolic_tokenized_prompt_mask", None
            ),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = super().to_dict()
        result["static_image_emb"] = self.static_image_emb
        result["static_mask"] = self.static_mask
        result["static_pos_emb"] = self.static_pos_emb
        result["static_state_emb"] = self.static_state_emb
        result["static_time"] = self.static_time
        result["recur_image_emb"] = self.recur_image_emb
        result["recur_mask"] = self.recur_mask
        result["recur_pos_emb"] = self.recur_pos_emb
        result["recur_state_emb"] = self.recur_state_emb
        result["cosmo_latent_emb"] = self.cosmo_latent_emb
        result["cosmo_recent_emb"] = self.cosmo_recent_emb
        result["cosmo_mem_mask"] = self.cosmo_mem_mask
        result["cosmo_recent_mask"] = self.cosmo_recent_mask
        result["cosmo_action_mask"] = self.cosmo_action_mask
        result["cosmo_proprio_mask"] = self.cosmo_proprio_mask
        result["cosmo_action_emb"] = self.cosmo_action_emb
        result["cosmo_proprio_emb"] = self.cosmo_proprio_emb
        result["cosmo_lang_emb"] = self.cosmo_lang_emb
        result["cosmo_mem_tokens"] = self.cosmo_mem_tokens
        result["symbolic_tokenized_prompt"] = self.symbolic_tokenized_prompt  #  subgoal
        result["symbolic_tokenized_prompt_mask"] = self.symbolic_tokenized_prompt_mask
        return result

    def to_base_obs(self) -> _Observation:
        return _Observation(
            images=self.images,
            image_masks=self.image_masks,
            state=self.state,
            tokenized_prompt=self.tokenized_prompt,
            tokenized_prompt_mask=self.tokenized_prompt_mask,
            token_ar_mask=self.token_ar_mask,
            token_loss_mask=self.token_loss_mask,
        )

    @classmethod
    def from_base_obs(
        cls,
        base_obs: _Observation,
        static_image_emb: at.Float[ArrayT, "*b l d1"] | None = None,
        static_mask: at.Bool[ArrayT, "*b l"] | None = None,
        static_pos_emb: at.Float[ArrayT, "*b l d2"] | None = None,
        static_state_emb: at.Float[ArrayT, "*b l d3"] | None = None,
        static_time: at.Float[ArrayT, "*b l 1"] | None = None,
        recur_image_emb: at.Float[ArrayT, "*b t v p d1"] | None = None,
        recur_mask: at.Bool[ArrayT, "*b t"] | None = None,
        recur_pos_emb: at.Float[ArrayT, "*b t v p d2"] | None = None,
        recur_state_emb: at.Float[ArrayT, "*b t d3"] | None = None,
        cosmo_latent_emb: at.Float[ArrayT, "*b lc dc"] | None = None,
        cosmo_recent_emb: at.Float[ArrayT, "*b lr dr"] | None = None,
        cosmo_mem_mask: at.Bool[ArrayT, "*b lc"] | None = None,
        cosmo_recent_mask: at.Bool[ArrayT, "*b lr"] | None = None,
        cosmo_action_mask: at.Bool[ArrayT, "*b la"] | None = None,
        cosmo_proprio_mask: at.Bool[ArrayT, "*b lp"] | None = None,
        cosmo_action_emb: at.Float[ArrayT, "*b la da"] | None = None,
        cosmo_proprio_emb: at.Float[ArrayT, "*b lp dp"] | None = None,
        cosmo_lang_emb: at.Float[ArrayT, "*b dl"] | None = None,
        cosmo_mem_tokens: at.Float[ArrayT, "*b m dt"] | None = None,
        symbolic_tokenized_prompt: at.Int[ArrayT, "*b l d5"] | None = None,
        symbolic_tokenized_prompt_mask: at.Bool[ArrayT, "*b l d6"] | None = None,
    ) -> "HistAugObservation":
        return HistAugObservation(
            images=base_obs.images,
            image_masks=base_obs.image_masks,
            state=base_obs.state,
            tokenized_prompt=base_obs.tokenized_prompt,
            tokenized_prompt_mask=base_obs.tokenized_prompt_mask,
            token_ar_mask=base_obs.token_ar_mask,
            token_loss_mask=base_obs.token_loss_mask,
            static_image_emb=static_image_emb,
            static_mask=static_mask,
            static_pos_emb=static_pos_emb,
            static_state_emb=static_state_emb,
            static_time=static_time,
            recur_image_emb=recur_image_emb,
            recur_mask=recur_mask,
            recur_pos_emb=recur_pos_emb,
            recur_state_emb=recur_state_emb,
            cosmo_latent_emb=cosmo_latent_emb,
            cosmo_recent_emb=cosmo_recent_emb,
            cosmo_mem_mask=cosmo_mem_mask,
            cosmo_recent_mask=cosmo_recent_mask,
            cosmo_action_mask=cosmo_action_mask,
            cosmo_proprio_mask=cosmo_proprio_mask,
            cosmo_action_emb=cosmo_action_emb,
            cosmo_proprio_emb=cosmo_proprio_emb,
            cosmo_lang_emb=cosmo_lang_emb,
            cosmo_mem_tokens=cosmo_mem_tokens,
            symbolic_tokenized_prompt=symbolic_tokenized_prompt,
            symbolic_tokenized_prompt_mask=symbolic_tokenized_prompt_mask,
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: HistAugObservation,
    *args,
    **kwargs,
) -> HistAugObservation:
    base_obs: _Observation = _preprocess_observation(
        rng,
        observation.to_base_obs(),
        *args,
        **kwargs,
    )
    return HistAugObservation.from_base_obs(
        base_obs,
        static_image_emb=observation.static_image_emb,
        static_mask=observation.static_mask,
        static_pos_emb=observation.static_pos_emb,
        static_state_emb=observation.static_state_emb,
        static_time=observation.static_time,
        recur_image_emb=observation.recur_image_emb,
        recur_mask=observation.recur_mask,
        recur_pos_emb=observation.recur_pos_emb,
        recur_state_emb=observation.recur_state_emb,
        cosmo_latent_emb=observation.cosmo_latent_emb,
        cosmo_recent_emb=observation.cosmo_recent_emb,
        cosmo_mem_mask=observation.cosmo_mem_mask,
        cosmo_recent_mask=observation.cosmo_recent_mask,
        cosmo_action_mask=observation.cosmo_action_mask,
        cosmo_proprio_mask=observation.cosmo_proprio_mask,
        cosmo_action_emb=observation.cosmo_action_emb,
        cosmo_proprio_emb=observation.cosmo_proprio_emb,
        cosmo_lang_emb=observation.cosmo_lang_emb,
        symbolic_tokenized_prompt=observation.symbolic_tokenized_prompt,
        symbolic_tokenized_prompt_mask=observation.symbolic_tokenized_prompt_mask,
    )
