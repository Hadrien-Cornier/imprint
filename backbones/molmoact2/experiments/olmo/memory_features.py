"""Feature-name contracts for external Stage-1 memory producers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryFeatureKeys:
    """Names used by a dataset row before memory values enter MolmoAct2."""

    cosmo_latents: str = "memory_cosmo_latents"
    recent_embeddings: str = "memory_recent_embeddings"
    action_embeddings: str = "memory_action_embeddings"
    proprio_embeddings: str = "memory_proprio_embeddings"
    cosmo_mask: str = "memory_cosmo_mask"
    recent_mask: str = "memory_recent_mask"
    action_mask: str = "memory_action_mask"
    proprio_mask: str = "memory_proprio_mask"
    language_embedding: str = "memory_language_embedding"
    tokens: str = "memory_tokens"
    attention_mask: str = "memory_attention_mask"

    @classmethod
    def from_prefix(cls, prefix: str) -> "MemoryFeatureKeys":
        prefix = str(prefix)
        return cls(
            cosmo_latents=f"{prefix}cosmo_latents",
            recent_embeddings=f"{prefix}recent_embeddings",
            action_embeddings=f"{prefix}action_embeddings",
            proprio_embeddings=f"{prefix}proprio_embeddings",
            cosmo_mask=f"{prefix}cosmo_mask",
            recent_mask=f"{prefix}recent_mask",
            action_mask=f"{prefix}action_mask",
            proprio_mask=f"{prefix}proprio_mask",
            language_embedding=f"{prefix}language_embedding",
            tokens=f"{prefix}tokens",
            attention_mask=f"{prefix}attention_mask",
        )

    @classmethod
    def from_schema(cls, schema: str, prefix: str = "memory_") -> "MemoryFeatureKeys":
        """Return the external feature names for a supported Stage-1 producer."""

        normalized_schema = str(schema).strip().lower()
        if normalized_schema == "canonical":
            return cls.from_prefix(prefix)
        if normalized_schema == "pi05_dinov3":
            return cls(
                cosmo_latents="cosmo_latent_emb",
                recent_embeddings="cosmo_recent_emb",
                action_embeddings="cosmo_action_emb",
                proprio_embeddings="cosmo_proprio_emb",
                cosmo_mask="cosmo_mem_mask",
                recent_mask="cosmo_recent_mask",
                action_mask="cosmo_action_mask",
                proprio_mask="cosmo_proprio_mask",
                language_embedding="cosmo_lang_emb",
                tokens="memory_tokens",
                attention_mask="memory_attention_mask",
            )
        raise ValueError(
            "Unsupported memory feature schema "
            f"{schema!r}; expected 'pi05_dinov3' or 'canonical'."
        )
