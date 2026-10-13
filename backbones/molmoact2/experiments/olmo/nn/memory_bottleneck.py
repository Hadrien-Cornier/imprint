"""Stage-1 memory bottleneck inputs shared by MolmoAct2 Stage-2 paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from olmo.config import BaseConfig


@dataclass
class MemoryBottleneckConfig(BaseConfig):
    """Configuration for the Pi0.5-compatible Stage-1 memory encoder."""

    enabled: bool = False
    mode: str = "frozen"
    num_latents: int = 256
    memory_dim: int = 1024
    cosmos_latent_dim: int = 48
    recent_dim: int = 768
    action_dim: int = 16
    proprio_dim: int = 16
    language_dim: int = 2048
    tokenizer_hidden_dim: int = 512
    num_cross_blocks: int = 2
    num_heads: int = 8
    stage1_checkpoint: str | None = None

    def validate(self) -> None:
        if self.mode not in {"frozen", "finetune", "precomputed"}:
            raise ValueError("memory mode must be frozen, finetune, or precomputed")
        if self.enabled and self.num_latents < 1:
            raise ValueError("num_latents must be positive")
        if self.memory_dim < 1:
            raise ValueError("memory_dim must be positive")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.memory_dim % self.num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")


@dataclass
class MemoryInputs:
    """Raw Stage-1 feature streams consumed by the memory bottleneck."""

    cosmo_latents: Tensor
    recent_embeddings: Tensor
    action_embeddings: Tensor
    proprio_embeddings: Tensor
    cosmo_mask: Tensor
    recent_mask: Tensor
    action_mask: Tensor
    proprio_mask: Tensor
    language_embedding: Tensor
    config: MemoryBottleneckConfig

    @classmethod
    def from_tensors(
        cls,
        *,
        cosmo_latents: Tensor,
        recent_embeddings: Tensor,
        action_embeddings: Tensor,
        proprio_embeddings: Tensor,
        cosmo_mask: Tensor,
        recent_mask: Tensor,
        action_mask: Tensor,
        proprio_mask: Tensor,
        language_embedding: Tensor,
        config: MemoryBottleneckConfig,
    ) -> "MemoryInputs":
        return cls(
            cosmo_latents=cosmo_latents,
            recent_embeddings=recent_embeddings,
            action_embeddings=action_embeddings,
            proprio_embeddings=proprio_embeddings,
            cosmo_mask=cosmo_mask,
            recent_mask=recent_mask,
            action_mask=action_mask,
            proprio_mask=proprio_mask,
            language_embedding=language_embedding,
            config=config,
        )

    @classmethod
    def from_model_kwargs(
        cls,
        *,
        config: MemoryBottleneckConfig,
        memory_cosmo_latents: Tensor,
        memory_recent_embeddings: Tensor,
        memory_action_embeddings: Tensor,
        memory_proprio_embeddings: Tensor,
        memory_cosmo_mask: Tensor,
        memory_recent_mask: Tensor,
        memory_action_mask: Tensor,
        memory_proprio_mask: Tensor,
        memory_language_embedding: Tensor,
    ) -> "MemoryInputs":
        return cls.from_tensors(
            cosmo_latents=memory_cosmo_latents,
            recent_embeddings=memory_recent_embeddings,
            action_embeddings=memory_action_embeddings,
            proprio_embeddings=memory_proprio_embeddings,
            cosmo_mask=memory_cosmo_mask,
            recent_mask=memory_recent_mask,
            action_mask=memory_action_mask,
            proprio_mask=memory_proprio_mask,
            language_embedding=memory_language_embedding,
            config=config,
        )

    def validate(self) -> None:
        self.config.validate()
        tensors = (
            self.cosmo_latents,
            self.recent_embeddings,
            self.action_embeddings,
            self.proprio_embeddings,
            self.cosmo_mask,
            self.recent_mask,
            self.action_mask,
            self.proprio_mask,
            self.language_embedding,
        )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError("all memory inputs must be on the same device")
        streams = (
            ("cosmo_latents", self.cosmo_latents, self.config.cosmos_latent_dim),
            ("recent_embeddings", self.recent_embeddings, self.config.recent_dim),
            ("action_embeddings", self.action_embeddings, self.config.action_dim),
            ("proprio_embeddings", self.proprio_embeddings, self.config.proprio_dim),
        )
        batch_size: int | None = None
        for name, value, width in streams:
            if value.ndim != 3:
                raise ValueError(f"{name} must have rank 3, got shape {tuple(value.shape)}")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if value.shape[-1] != width:
                raise ValueError(f"{name} must have width {width}, got {value.shape[-1]}")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(f"{name} batch size does not match memory streams")

        assert batch_size is not None
        masks = (
            ("cosmo_mask", self.cosmo_mask, self.cosmo_latents.shape[1]),
            ("recent_mask", self.recent_mask, self.recent_embeddings.shape[1]),
            ("action_mask", self.action_mask, self.action_embeddings.shape[1]),
            ("proprio_mask", self.proprio_mask, self.proprio_embeddings.shape[1]),
        )
        for name, mask, sequence_length in masks:
            if mask.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
            if tuple(mask.shape) != (batch_size, sequence_length):
                raise ValueError(
                    f"{name} must have shape {(batch_size, sequence_length)}, got {tuple(mask.shape)}"
                )

        if tuple(self.language_embedding.shape) != (batch_size, self.config.language_dim):
            raise ValueError(
                "language_embedding must have shape "
                f"{(batch_size, self.config.language_dim)}, got {tuple(self.language_embedding.shape)}"
            )
        if not self.language_embedding.is_floating_point():
            raise ValueError("language_embedding must be floating point")
        valid_tokens = (
            self.cosmo_mask.sum(dim=1)
            + self.recent_mask.sum(dim=1)
            + self.action_mask.sum(dim=1)
            + self.proprio_mask.sum(dim=1)
        )
        if torch.any(valid_tokens == 0):
            raise ValueError("each batch row must contain at least one valid memory token")


class MlpTokenizer(nn.Module):
    """Stage-1 action/proprio tokenizer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: Tensor) -> Tensor:
        return self.l2(F.gelu(self.l1(value), approximate="tanh"))


class MemoryMultiHeadAttention(nn.Module):
    """Bias-enabled attention matching Flax NNX ``MultiHeadAttention`` semantics."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("memory attention dimension must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        queries: Tensor,
        keys_values: Tensor,
        *,
        valid_keys_values: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_length, _ = queries.shape
        key_value_length = keys_values.shape[1]
        q = self.q_proj(queries).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(keys_values).view(
            batch_size, key_value_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(keys_values).view(
            batch_size, key_value_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_mask = None
        if valid_keys_values is not None:
            invalid = ~valid_keys_values.to(device=queries.device, dtype=torch.bool)
            attn_mask = invalid[:, None, None, :].to(dtype=queries.dtype)
            attn_mask = attn_mask * torch.finfo(queries.dtype).min
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, self.dim)
        return self.out_proj(attended)


class MemoryPerceiverBlock(nn.Module):
    """One cross-attention, self-attention, and tanh residual Stage-1 block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MemoryMultiHeadAttention(dim, num_heads)
        self.self_attn = MemoryMultiHeadAttention(dim, num_heads)
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.norm_s = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Linear(dim, dim)

    def forward(
        self,
        queries: Tensor,
        keys_values: Tensor,
        valid_keys_values: Tensor,
    ) -> Tensor:
        queries = queries + self.attn(
            self.norm_q(queries),
            self.norm_kv(keys_values),
            valid_keys_values=valid_keys_values,
        )
        normalized_queries = self.norm_s(queries)
        queries = queries + self.self_attn(normalized_queries, normalized_queries)
        return queries + torch.tanh(self.ffn(queries))


class MemoryBottleneckEncoder(nn.Module):
    """PyTorch port of Pi0.5's learned Stage-1 Cosmo Perceiver bottleneck."""

    def __init__(self, config: MemoryBottleneckConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.memory_dim
        self.proj_cosmo = nn.Linear(config.cosmos_latent_dim, dim)
        self.proj_recent = nn.Linear(config.recent_dim, dim)
        self.tok_action = MlpTokenizer(config.action_dim, config.tokenizer_hidden_dim, dim)
        self.tok_proprio = MlpTokenizer(config.proprio_dim, config.tokenizer_hidden_dim, dim)
        self.lang_film = nn.Linear(config.language_dim, dim * 2)
        self.latents = nn.Parameter(torch.empty(config.num_latents, dim))
        self.blocks = nn.ModuleList(
            MemoryPerceiverBlock(dim, config.num_heads)
            for _ in range(config.num_cross_blocks)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.latents, mean=0.0, std=0.02)

    def forward(self, memory: MemoryInputs) -> tuple[Tensor, Tensor]:
        memory.validate()
        dtype = self.latents.dtype
        streams = torch.cat(
            (
                self.proj_cosmo(memory.cosmo_latents.to(dtype=dtype)),
                self.proj_recent(memory.recent_embeddings.to(dtype=dtype)),
                self.tok_action(memory.action_embeddings.to(dtype=dtype)),
                self.tok_proprio(memory.proprio_embeddings.to(dtype=dtype)),
            ),
            dim=1,
        )
        stream_mask = torch.cat(
            (
                memory.cosmo_mask,
                memory.recent_mask,
                memory.action_mask,
                memory.proprio_mask,
            ),
            dim=1,
        )
        queries = self.latents.unsqueeze(0).expand(streams.shape[0], -1, -1)
        gamma, beta = self.lang_film(memory.language_embedding.to(dtype=dtype)).chunk(2, dim=-1)
        queries = queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, streams, stream_mask)
        return queries, torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
