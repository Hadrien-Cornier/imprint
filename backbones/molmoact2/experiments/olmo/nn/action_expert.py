import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard

from olmo.config import BaseConfig, D
from olmo.nn.flash_attention_api import dispatch_flash_attn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _tensor_all_finite(tensor: Optional[torch.Tensor]) -> Optional[bool]:
    if tensor is None:
        return None
    return bool(torch.isfinite(tensor).all().item())


def _tensor_max_abs(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None:
        return None
    safe = torch.nan_to_num(tensor.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(safe.abs().max().item())


def _nonfinite_batch_rows(tensor: Optional[torch.Tensor]) -> Optional[list[int]]:
    if tensor is None or tensor.ndim == 0:
        return None
    bad = ~torch.isfinite(tensor)
    if bad.ndim == 1:
        rows = bad
    else:
        rows = bad.reshape(bad.shape[0], -1).any(dim=1)
    return rows.nonzero(as_tuple=False).flatten().tolist()


class ActionExpertRMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


class MemoryFiLMRMSNorm(nn.Module):
    """Memory-conditioned RMSNorm with an identity-preserving, trainable FiLM delta."""

    def __init__(
        self,
        hidden_size: int,
        *,
        init_std: float = 0.02,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.to_scale_shift = nn.Linear(hidden_size, hidden_size * 2)
        self.init_std = init_std
        self.identity_init = identity_init
        self.register_buffer(
            "reference_weight",
            torch.empty_like(self.to_scale_shift.weight, requires_grad=False),
        )
        self.register_buffer(
            "reference_bias",
            torch.empty_like(self.to_scale_shift.bias, requires_grad=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        nn.init.normal_(self.to_scale_shift.weight, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.to_scale_shift.bias)
        with torch.no_grad():
            self.reference_weight.copy_(self.to_scale_shift.weight)
            self.reference_bias.copy_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, memory_read: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(memory_read)
        if not self.identity_init:
            scale, shift = scale_shift.chunk(2, dim=-1)
            return self.norm(x) * (1 + scale) + shift
        reference = F.linear(
            memory_read.detach(),
            self.reference_weight,
            self.reference_bias,
        )
        scale, shift = (scale_shift - reference).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class ActionExpertRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, half_dim, device=q.device, dtype=torch.float32) / max(half_dim, 1))
        )
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=q.dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=q.dtype).view(1, 1, seq_len, half_dim)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)

    def apply_positions(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to [batch, heads, sequence, head_dim] at explicit positions."""
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / max(half_dim, 1))
        )
        freqs = torch.outer(positions.to(device=x.device, dtype=torch.float32), inv_freq)
        cos = freqs.cos().to(dtype=x.dtype).view(1, 1, -1, half_dim)
        sin = freqs.sin().to(dtype=x.dtype).view(1, 1, -1, half_dim)
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ActionExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.qk_norm = qk_norm
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        if attn_mask is None and q.is_cuda:
            try:
                return dispatch_flash_attn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    causal=is_causal,
                )
            except RuntimeError:
                pass

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self._apply_qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.contiguous()
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.reshape(bsz, seq_len, self.hidden_size)
        out = self.out_proj(out)
        return self.out_drop(out)


class ActionExpertCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _as_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[2] == self.num_heads:
                return x
            if x.shape[1] == self.num_heads:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"Unexpected cross-attention KV shape {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D cross-attention KV, got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        if attn_mask is None and q.is_cuda:
            try:
                return dispatch_flash_attn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    causal=False,
                )
            except RuntimeError:
                pass

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv: Optional[torch.Tensor] = None,
        kv_k: Optional[torch.Tensor] = None,
        kv_v: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (kv_k is None) != (kv_v is None):
            raise ValueError("kv_k and kv_v must both be provided or both be None.")
        if kv is not None and kv_k is not None:
            raise ValueError("Provide either kv or kv_k/kv_v, not both.")
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        if kv_k is not None and kv_v is not None:
            k = self._as_heads(kv_k)
            v = self._as_heads(kv_v)
        else:
            if kv is None:
                raise ValueError("cross-attention requires kv or kv_k/kv_v.")
            src_len = kv.shape[1]
            kv_proj = self.kv_proj(kv).view(bsz, src_len, 2, self.num_heads, self.head_dim)
            k = kv_proj[:, :, 0]
            v = kv_proj[:, :, 1]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self._apply_qk_norm(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        out = self.out_proj(out)
        return self.out_drop(out)


class Pi05MemoryCrossAttention(nn.Module):
    """Pi0.5-style memory-only grouped-query cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by memory_num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("memory_num_heads must be divisible by memory_num_kv_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.input_norm = Pi05MemoryInputNorm(hidden_size)
        self.q_norm = None
        self.k_norm = None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, 2 * num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        src_len = kv.shape[1]
        q = self.q_proj(self.input_norm(x)).view(bsz, tgt_len, self.num_heads, self.head_dim)
        kv_proj = self.kv_proj(self.input_norm(kv)).view(
            bsz,
            src_len,
            2,
            self.num_kv_heads,
            self.head_dim,
        )
        k = kv_proj[:, :, 0].transpose(1, 2)
        v = kv_proj[:, :, 1].transpose(1, 2)
        q = q.transpose(1, 2)
        q = self.rope.apply_positions(
            q,
            torch.arange(src_len, src_len + tgt_len, device=q.device),
        )
        k = self.rope.apply_positions(k, torch.arange(src_len, device=k.device))
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().reshape(bsz, tgt_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class Pi05MemoryInputNorm(nn.Module):
    """Shared Pi0.5 memory-input RMSNorm with a zero-initialized residual scale."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * (1 + self.scale)


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        mlp_ratio: float,
        multiple_of: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        ffn_multiple_of: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope: bool = True,
        memory_enabled: bool = False,
        memory_seam_mode: Literal["native", "pi05"] = "native",
        memory_num_heads: int = 4,
        memory_num_kv_heads: int = 1,
        memory_film_init_std: float = 0.002,
        memory_film_identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_rope=rope,
        )
        self.cross_attn = ActionExpertCrossAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.mlp = ActionExpertMLP(
            hidden_size,
            mlp_ratio=mlp_ratio,
            multiple_of=ffn_multiple_of,
            dropout=dropout,
        )
        self.modulation = ActionExpertModulation(hidden_size, 9)
        self.memory_enabled = memory_enabled
        self.memory_seam_mode = memory_seam_mode
        if memory_enabled:
            if memory_seam_mode == "pi05":
                self.memory_query_norm = None
                self.memory_attn: Optional[nn.Module] = Pi05MemoryCrossAttention(
                    hidden_size,
                    memory_num_heads,
                    memory_num_kv_heads,
                    attn_dropout=attn_dropout,
                    proj_dropout=dropout,
                )
                self.memory_film: Optional[MemoryFiLMRMSNorm] = MemoryFiLMRMSNorm(
                    hidden_size,
                    init_std=memory_film_init_std,
                    identity_init=False,
                )
            else:
                self.memory_query_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
                self.memory_attn = ActionExpertCrossAttention(
                    hidden_size,
                    num_heads,
                    attn_dropout=attn_dropout,
                    proj_dropout=dropout,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                )
                self.memory_film = MemoryFiLMRMSNorm(
                    hidden_size,
                    init_std=memory_film_init_std,
                    identity_init=memory_film_identity_init,
                )
        else:
            self.memory_query_norm = None
            self.memory_attn = None
            self.memory_film = None

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_context: Optional[torch.Tensor] = None,
        cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(conditioning).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            is_causal=is_causal,
        )

        attn_kwargs = {}
        if cross_kv is not None:
            attn_kwargs["kv_k"] = cross_kv[0]
            attn_kwargs["kv_v"] = cross_kv[1]
        else:
            if cross_context is None:
                raise ValueError("cross-attention requires cross_context or cross_kv.")
            attn_kwargs["kv"] = cross_context
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            attn_mask=attn_mask,
            **attn_kwargs,
        )

        ffn_input = self.ff_norm(x)
        if memory_tokens is not None:
            if self.memory_attn is None or self.memory_film is None:
                raise RuntimeError("Memory tokens were provided to an action expert without memory conditioning.")
            memory_read = self.memory_attn(
                x if self.memory_query_norm is None else self.memory_query_norm(x),
                kv=memory_tokens,
                attn_mask=memory_attention_mask,
            )
            memory_film_output = self.memory_film(x, memory_read)
            ffn_input = (
                self.ff_norm(memory_film_output)
                if self.memory_seam_mode == "pi05"
                else memory_film_output
            )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(ffn_input, shift_mlp, scale_mlp)
        )
        return x


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class ActionExpertConfig(BaseConfig):
    max_horizon: int = 30
    max_action_dim: int = 32
    hidden_size: int = 1024
    num_layers: int = 32
    num_heads: int = 16
    mlp_ratio: float = 8.0 / 3.0
    ffn_multiple_of: int = 256
    timestep_embed_dim: int = 256
    dropout: float = 0.0
    attn_dropout: float = 0.0
    context_layer_norm: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    rope_on_cross_attention: bool = False
    causal_attn: bool = False
    memory_enabled: bool = False
    # OmegaConf cannot materialize typing.Literal in nested model configs.
    # _validate_memory_seam() below enforces the same two allowed values.
    memory_seam_mode: str = "native"
    memory_num_heads: int = 4
    memory_num_kv_heads: int = 1
    memory_film_init_std: float = 0.002
    memory_film_identity_init: bool = True
    compile: Optional[str] = "blocks"

    def _validate_memory_seam(self) -> None:
        if self.memory_seam_mode not in {"native", "pi05"}:
            raise ValueError(
                "memory_seam_mode must be one of {'native', 'pi05'}, "
                f"got {self.memory_seam_mode!r}."
            )
        if self.memory_seam_mode != "pi05":
            return
        if self.memory_num_heads <= 0:
            raise ValueError("memory_num_heads must be positive for the Pi0.5 memory seam.")
        if self.memory_num_kv_heads <= 0:
            raise ValueError("memory_num_kv_heads must be positive for the Pi0.5 memory seam.")
        if self.hidden_size % self.memory_num_heads != 0:
            raise ValueError("hidden_size must be divisible by memory_num_heads for the Pi0.5 memory seam.")
        if self.memory_num_heads % self.memory_num_kv_heads != 0:
            raise ValueError(
                "memory_num_heads must be divisible by memory_num_kv_heads "
                "for the Pi0.5 memory seam."
            )

    def build(
        self,
        llm_dim: int,
        llm_kv_dim: Optional[int] = None,
        llm_num_kv_heads: Optional[int] = None,
        llm_num_layers: Optional[int] = None,
        device=None,
    ) -> "ActionExpert":
        self._validate_memory_seam()
        return ActionExpert(
            self,
            llm_dim=llm_dim,
            llm_kv_dim=llm_kv_dim,
            llm_num_kv_heads=llm_num_kv_heads,
            llm_num_layers=llm_num_layers,
            device=device,
        )

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "implementation" in config:
            implementation = str(config["implementation"])
            if implementation not in {"new", "modern"}:
                raise ValueError(
                    "Only the modern MolmoAct2 action expert is supported. "
                    f"Found legacy implementation={implementation!r}."
                )
            del config["implementation"]
        if "action_dim" in config:
            legacy_dim = int(config["action_dim"])
            if "max_action_dim" in config and int(config["max_action_dim"]) != legacy_dim:
                raise ValueError(
                    "Found conflicting action expert action dimensions in config: "
                    f"action_dim={legacy_dim} vs max_action_dim={int(config['max_action_dim'])}."
                )
            config["max_action_dim"] = legacy_dim
            del config["action_dim"]
        if "context_connection" in config:
            del config["context_connection"]
        return config


class ActionExpert(nn.Module):
    def __init__(
        self,
        config: ActionExpertConfig,
        llm_dim: int,
        llm_kv_dim: Optional[int] = None,
        llm_num_kv_heads: Optional[int] = None,
        llm_num_layers: Optional[int] = None,
        device=None,
    ):
        super().__init__()
        config._validate_memory_seam()
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_dim = llm_dim
        self.llm_kv_dim = llm_dim if llm_kv_dim is None else llm_kv_dim
        self.llm_num_kv_heads = llm_num_kv_heads
        self.llm_num_layers = llm_num_layers
        self.use_kv_flat_condition = True
        self.use_kv_condition = True
        self.action_head_dim = config.hidden_size // config.num_heads
        self.llm_kv_head_dim = None

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
        )

        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size, device=device)
        self.memory_adapter: Optional[nn.Linear] = None
        self.state_encoder = nn.Linear(config.hidden_size, config.hidden_size, device=device)
        self.state_norm = ActionExpertRMSNorm(config.hidden_size, eps=1e-6, device=device)

        self.context_k_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)
        self.context_v_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)

        self.context_norm = (
            ActionExpertRMSNorm(config.hidden_size, eps=1e-6)
            if config.context_layer_norm
            else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    ffn_multiple_of=config.ffn_multiple_of,
                    attn_dropout=config.attn_dropout,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                    rope=config.rope,
                    memory_enabled=config.memory_enabled,
                    memory_seam_mode=config.memory_seam_mode,
                    memory_num_heads=config.memory_num_heads,
                    memory_num_kv_heads=config.memory_num_kv_heads,
                    memory_film_init_std=config.memory_film_init_std,
                    memory_film_identity_init=config.memory_film_identity_init,
                )
                for _ in range(config.num_layers)
            ]
        )
        if self.use_kv_condition:
            for block in self.blocks:
                block.cross_attn.kv_proj.weight.requires_grad = False
                if block.cross_attn.kv_proj.bias is not None:
                    block.cross_attn.kv_proj.bias.requires_grad = False

        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.max_action_dim)
        self.reset_parameters()

    def _reshape_hidden_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.config.num_heads, self.action_head_dim)

    def _flatten_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.hidden_size)

    def reset_parameters(self):
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        _init_linear(self.state_encoder)
        self.state_norm.reset_parameters()
        self.context_k_proj.reset_parameters()
        self.context_v_proj.reset_parameters()
        if isinstance(self.context_norm, ActionExpertRMSNorm):
            self.context_norm.reset_parameters()

        residual_scale = (2 * max(self.config.num_layers, 1)) ** -0.5
        for block in self.blocks:
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.kv_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)
            block.self_norm.reset_parameters()
            block.cross_norm.reset_parameters()
            block.ff_norm.reset_parameters()
            if block.self_attn.q_norm is not None:
                block.self_attn.q_norm.reset_parameters()
            if block.self_attn.k_norm is not None:
                block.self_attn.k_norm.reset_parameters()
            if block.cross_attn.q_norm is not None:
                block.cross_attn.q_norm.reset_parameters()
            if block.cross_attn.k_norm is not None:
                block.cross_attn.k_norm.reset_parameters()
            if block.memory_enabled:
                assert block.memory_attn is not None
                assert block.memory_film is not None
                if block.memory_seam_mode == "pi05":
                    for projection in (
                        block.memory_attn.q_proj,
                        block.memory_attn.kv_proj,
                        block.memory_attn.out_proj,
                    ):
                        nn.init.normal_(projection.weight, mean=0.0, std=projection.in_features**-0.5)
                        nn.init.zeros_(projection.bias)
                else:
                    _init_linear(block.memory_attn.q_proj)
                    _init_linear(block.memory_attn.kv_proj)
                    _init_linear(block.memory_attn.out_proj, scale=residual_scale)
                if block.memory_query_norm is not None:
                    block.memory_query_norm.reset_parameters()
                block.memory_film.reset_parameters()
                if isinstance(block.memory_attn, Pi05MemoryCrossAttention):
                    nn.init.zeros_(block.memory_attn.input_norm.scale)
                if block.memory_attn.q_norm is not None:
                    block.memory_attn.q_norm.reset_parameters()
                if block.memory_attn.k_norm is not None:
                    block.memory_attn.k_norm.reset_parameters()

        self.final_layer.norm.reset_parameters()
        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    def prepare_state_dict_for_loading(self, state_dict: dict[str, torch.Tensor], prefix: str = "") -> int:
        """Add cleaned-code compatibility keys that older MolmoAct2 checkpoints do not contain."""
        added = 0

        def sample_tensor() -> torch.Tensor | None:
            for key in (
                f"{prefix}action_embed.weight",
                f"{prefix}final_layer.linear.weight",
                f"{prefix}context_k_proj.weight",
            ):
                value = state_dict.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            return None

        sample = sample_tensor()
        dtype = sample.dtype if sample is not None else torch.float32
        device = sample.device if sample is not None else torch.device("cpu")
        hidden_size = int(self.config.hidden_size)

        state_encoder_weight_key = f"{prefix}state_encoder.weight"
        if state_encoder_weight_key not in state_dict:
            state_dict[state_encoder_weight_key] = torch.eye(hidden_size, dtype=dtype, device=device)
            added += 1

        state_encoder_bias_key = f"{prefix}state_encoder.bias"
        if state_encoder_bias_key not in state_dict:
            state_dict[state_encoder_bias_key] = torch.zeros(hidden_size, dtype=dtype, device=device)
            added += 1

        # Under the single supported kv_cache conditioning path, projected K/V tensors
        # are passed directly into cross-attention. The kv_proj module is frozen and
        # inactive, but strict checkpoint loading still expects its parameters.
        for layer_idx in range(len(self.blocks)):
            q_proj_key = f"{prefix}blocks.{layer_idx}.cross_attn.q_proj.weight"
            q_proj = state_dict.get(q_proj_key)
            layer_hidden_size = int(q_proj.shape[0]) if isinstance(q_proj, torch.Tensor) else hidden_size
            layer_dtype = q_proj.dtype if isinstance(q_proj, torch.Tensor) else dtype
            layer_device = q_proj.device if isinstance(q_proj, torch.Tensor) else device

            kv_weight_key = f"{prefix}blocks.{layer_idx}.cross_attn.kv_proj.weight"
            if kv_weight_key not in state_dict:
                state_dict[kv_weight_key] = torch.zeros(
                    (layer_hidden_size * 2, layer_hidden_size),
                    dtype=layer_dtype,
                    device=layer_device,
                )
                added += 1

            kv_bias_key = f"{prefix}blocks.{layer_idx}.cross_attn.kv_proj.bias"
            if kv_bias_key not in state_dict:
                state_dict[kv_bias_key] = torch.zeros(
                    layer_hidden_size * 2,
                    dtype=layer_dtype,
                    device=layer_device,
                )
                added += 1

        expected_memory_state = self.state_dict()
        if self.config.memory_seam_mode == "pi05":
            for layer_idx in range(len(self.blocks)):
                seam_prefix = f"{prefix}blocks.{layer_idx}.memory_"
                expected_kv = expected_memory_state[
                    f"blocks.{layer_idx}.memory_attn.kv_proj.weight"
                ]
                loaded_kv = state_dict.get(f"{seam_prefix}attn.kv_proj.weight")
                if isinstance(loaded_kv, torch.Tensor) and loaded_kv.shape != expected_kv.shape:
                    for state_key in list(state_dict):
                        if state_key.startswith(seam_prefix):
                            del state_dict[state_key]
                            added += 1
            for state_key in list(state_dict):
                if not state_key.startswith(prefix):
                    continue
                name = state_key.removeprefix(prefix)
                if ".memory_" not in name and not name.startswith("memory_adapter."):
                    continue
                expected_value = expected_memory_state.get(name)
                actual_value = state_dict[state_key]
                if (
                    expected_value is None
                    or not isinstance(actual_value, torch.Tensor)
                    or actual_value.shape != expected_value.shape
                ):
                    del state_dict[state_key]
                    added += 1

        for name, value in expected_memory_state.items():
            if ".memory_" not in name and not name.startswith("memory_adapter."):
                continue
            memory_key = f"{prefix}{name}"
            if memory_key not in state_dict:
                if name == "memory_adapter.weight":
                    adapter_weight = torch.empty_like(value)
                    nn.init.xavier_uniform_(adapter_weight)
                    state_dict[memory_key] = adapter_weight
                elif name == "memory_adapter.bias":
                    state_dict[memory_key] = torch.zeros_like(value)
                elif name.endswith("memory_film.to_scale_shift.weight"):
                    film_weight = torch.empty_like(value)
                    nn.init.normal_(
                        film_weight,
                        mean=0.0,
                        std=(
                            self.config.memory_film_init_std
                            if self.config.memory_seam_mode == "pi05"
                            else 0.02
                        ),
                    )
                    state_dict[memory_key] = film_weight
                elif name.endswith("memory_film.to_scale_shift.bias"):
                    state_dict[memory_key] = torch.zeros_like(value)
                elif name.endswith("memory_film.reference_weight"):
                    source_key = memory_key.replace(
                        "memory_film.reference_weight",
                        "memory_film.to_scale_shift.weight",
                    )
                    source_weight = state_dict.get(source_key)
                    if source_weight is None:
                        source_weight = torch.empty_like(value)
                        nn.init.normal_(
                            source_weight,
                            mean=0.0,
                            std=(
                                self.config.memory_film_init_std
                                if self.config.memory_seam_mode == "pi05"
                                else 0.02
                            ),
                        )
                        state_dict[source_key] = source_weight
                    state_dict[memory_key] = source_weight.detach().clone()
                elif name.endswith("memory_film.reference_bias"):
                    source_key = memory_key.replace(
                        "memory_film.reference_bias",
                        "memory_film.to_scale_shift.bias",
                    )
                    source_bias = state_dict.get(source_key)
                    if source_bias is None:
                        source_bias = torch.zeros_like(value)
                        state_dict[source_key] = source_bias
                    state_dict[memory_key] = source_bias.detach().clone()
                elif name.endswith("memory_attn.out_proj.weight"):
                    out_proj_weight = torch.empty_like(value)
                    if self.config.memory_seam_mode == "pi05":
                        nn.init.normal_(out_proj_weight, mean=0.0, std=value.shape[1] ** -0.5)
                    else:
                        nn.init.xavier_uniform_(out_proj_weight)
                        out_proj_weight.mul_((2 * max(self.config.num_layers, 1)) ** -0.5)
                    state_dict[memory_key] = out_proj_weight
                elif name.endswith(
                    (
                        "memory_attn.q_proj.weight",
                        "memory_attn.kv_proj.weight",
                    )
                ):
                    attention_weight = torch.empty_like(value)
                    if self.config.memory_seam_mode == "pi05":
                        nn.init.normal_(attention_weight, mean=0.0, std=value.shape[1] ** -0.5)
                    else:
                        nn.init.xavier_uniform_(attention_weight)
                    state_dict[memory_key] = attention_weight
                elif name.endswith(
                    (
                        "memory_attn.q_proj.bias",
                        "memory_attn.kv_proj.bias",
                        "memory_attn.out_proj.bias",
                    )
                ):
                    state_dict[memory_key] = torch.zeros_like(value)
                elif name.endswith(
                    (
                        "memory_query_norm.weight",
                        "memory_attn.q_norm.weight",
                        "memory_attn.k_norm.weight",
                    )
                ):
                    state_dict[memory_key] = torch.ones_like(value)
                else:
                    state_dict[memory_key] = torch.zeros_like(value)
                added += 1

        return added

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(timesteps)

    def get_continuous_state_conditioning_modules(self) -> Sequence[nn.Module]:
        return [self.state_encoder, self.state_norm]

    def freeze_continuous_state_conditioning(self) -> None:
        for module in self.get_continuous_state_conditioning_modules():
            module.requires_grad_(False)

    def _encode_states(
        self,
        states: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if states is None:
            return None
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if states.shape[-1] != self.hidden_size:
            feat_dim = states.shape[-1]
            if feat_dim < self.hidden_size:
                states = F.pad(states, (0, self.hidden_size - feat_dim))
            else:
                states = states[..., : self.hidden_size]
        encoded = self.state_encoder(states)
        return self.state_norm(encoded)

    def _project_kv_tensor(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        flat = self.context_norm(proj(x))
        return self._reshape_hidden_to_heads(flat)

    def _prepare_kv_context(
        self,
        encoder_kv_states: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        encoded_states: Optional[torch.Tensor],
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if self.context_k_proj is None or self.context_v_proj is None:
            raise RuntimeError("KV conditioning requested but KV projection modules are not initialized.")
        kv_contexts = []
        state_heads = self._reshape_hidden_to_heads(encoded_states) if encoded_states is not None else None

        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(
                "MolmoAct2 action expert expects one KV state per action expert block "
                f"(got {len(encoder_kv_states)}, expected {len(self.blocks)})."
            )

        for layer_idx, (k_in, v_in) in enumerate(encoder_kv_states):
            if not torch.isfinite(k_in).all() or not torch.isfinite(v_in).all():
                raise RuntimeError(
                    "Non-finite encoder KV states before action expert projection: "
                    f"layer_idx={layer_idx}, "
                    f"k_finite={_tensor_all_finite(k_in)}, "
                    f"v_finite={_tensor_all_finite(v_in)}, "
                    f"k_bad_batch_rows={_nonfinite_batch_rows(k_in)}, "
                    f"v_bad_batch_rows={_nonfinite_batch_rows(v_in)}, "
                    f"k_shape={tuple(k_in.shape)}, "
                    f"v_shape={tuple(v_in.shape)}."
                )
            k_ctx = self._project_kv_tensor(k_in, self.context_k_proj)
            v_ctx = self._project_kv_tensor(v_in, self.context_v_proj)
            if not torch.isfinite(k_ctx).all() or not torch.isfinite(v_ctx).all():
                raise RuntimeError(
                    "Non-finite projected KV context in action expert: "
                    f"layer_idx={layer_idx}, "
                    f"k_in_finite={_tensor_all_finite(k_in)}, "
                    f"v_in_finite={_tensor_all_finite(v_in)}, "
                    f"k_ctx_finite={_tensor_all_finite(k_ctx)}, "
                    f"v_ctx_finite={_tensor_all_finite(v_ctx)}, "
                    f"k_in_max_abs={_tensor_max_abs(k_in)}, "
                    f"v_in_max_abs={_tensor_max_abs(v_in)}, "
                    f"k_ctx_max_abs={_tensor_max_abs(k_ctx)}, "
                    f"v_ctx_max_abs={_tensor_max_abs(v_ctx)}."
                )
            if state_heads is not None:
                k_ctx = torch.cat([k_ctx, state_heads], dim=1)
                v_ctx = torch.cat([v_ctx, state_heads], dim=1)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    def _build_cross_attention_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        encoded_states: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        state_seq_len = 0 if encoded_states is None else encoded_states.shape[1]
        if encoder_attention_mask is None and state_seq_len == 0:
            return None
        if encoder_attention_mask is None:
            return None
        if encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
        else:
            mask = encoder_attention_mask.to(dtype=dtype)
        if state_seq_len > 0:
            ones = torch.ones(
                batch_size,
                1,
                1,
                state_seq_len,
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([mask, ones], dim=-1)
        return (1.0 - mask) * torch.finfo(dtype).min

    def _build_self_attention_mask(
        self,
        action_attention_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        mask = None
        if action_attention_mask is not None:
            valid = action_attention_mask.to(device=device, dtype=torch.bool)
            if valid.ndim != 2 or valid.shape[1] != seq_len:
                raise ValueError(
                    f"Expected action_attention_mask shape (batch, {seq_len}), got {tuple(valid.shape)}."
                )
            key_mask = (~valid)[:, None, None, :].to(dtype=dtype)
            mask = key_mask * torch.finfo(dtype).min
        if self.config.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
            causal = causal.unsqueeze(0).unsqueeze(0).to(dtype=dtype) * torch.finfo(dtype).min
            mask = causal if mask is None else (mask + causal)
        return mask

    def _build_memory_attention_mask(
        self,
        memory_attention_mask: torch.Tensor,
        *,
        batch_size: int,
        memory_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        valid = memory_attention_mask.to(device=device, dtype=torch.bool)
        if tuple(valid.shape) != (batch_size, memory_length):
            raise ValueError(
                "Expected memory_attention_mask shape "
                f"{(batch_size, memory_length)}, got {tuple(valid.shape)}."
            )
        if (~valid).all(dim=-1).any():
            raise ValueError("Each batch row must contain at least one valid memory token.")
        return (~valid)[:, None, None, :].to(dtype=dtype) * torch.finfo(dtype).min

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        state_embeddings: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_kv_states is None:
            raise ValueError("Provide encoder_kv_states for MolmoAct2 action expert conditioning.")
        if len(encoder_kv_states) == 0:
            raise ValueError("Expected at least one encoder kv state for action expert conditioning")
        bsz, seq_len, _ = actions.shape
        if seq_len > self.config.max_horizon:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds configured max_horizon={self.config.max_horizon}"
            )
        if (memory_tokens is None) != (memory_attention_mask is None):
            raise ValueError("Provide memory_tokens and memory_attention_mask together.")
        if memory_tokens is not None and not self.config.memory_enabled:
            raise ValueError("Memory tokens require ActionExpertConfig.memory_enabled=True.")
        if memory_tokens is not None:
            if memory_tokens.ndim != 3:
                raise ValueError(
                    f"Expected rank-3 memory_tokens, got shape {tuple(memory_tokens.shape)}."
                )
            expected_memory_dim = (
                self.hidden_size
                if self.memory_adapter is None
                else self.memory_adapter.in_features
            )
            if memory_tokens.shape[0] != bsz or memory_tokens.shape[-1] != expected_memory_dim:
                raise ValueError(
                    "Expected memory_tokens shape "
                    f"({bsz}, memory_length, {expected_memory_dim}), got {tuple(memory_tokens.shape)}."
                )

        timestep_embed = self.time_embed(timesteps)
        if not torch.isfinite(timestep_embed).all():
            raise RuntimeError(
                "Non-finite timestep embedding in action expert: "
                f"timesteps_finite={_tensor_all_finite(timesteps)}, "
                f"timestep_embed_finite={_tensor_all_finite(timestep_embed)}, "
                f"timesteps_max_abs={_tensor_max_abs(timesteps)}, "
                f"timestep_embed_max_abs={_tensor_max_abs(timestep_embed)}."
            )
        encoded_states = self._encode_states(state_embeddings)
        if encoded_states is not None and not torch.isfinite(encoded_states).all():
            raise RuntimeError(
                "Non-finite encoded state conditioning in action expert: "
                f"state_embeddings_finite={_tensor_all_finite(state_embeddings)}, "
                f"encoded_states_finite={_tensor_all_finite(encoded_states)}, "
                f"state_embeddings_max_abs={_tensor_max_abs(state_embeddings)}, "
                f"encoded_states_max_abs={_tensor_max_abs(encoded_states)}."
            )
        x = self.action_embed(actions)
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "Non-finite action embedding in action expert: "
                f"actions_finite={_tensor_all_finite(actions)}, "
                f"action_embed_finite={_tensor_all_finite(x)}, "
                f"actions_max_abs={_tensor_max_abs(actions)}, "
                f"action_embed_max_abs={_tensor_max_abs(x)}."
            )
        if memory_tokens is not None:
            memory_tokens = memory_tokens.to(device=x.device, dtype=x.dtype)
            if self.memory_adapter is not None:
                memory_tokens = self.memory_adapter(memory_tokens)
        conditioning = timestep_embed
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * valid_action
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    "Non-finite masked action embedding in action expert: "
                    f"action_attention_mask_finite={_tensor_all_finite(action_attention_mask)}, "
                    f"masked_action_embed_finite={_tensor_all_finite(x)}, "
                    f"masked_action_embed_max_abs={_tensor_max_abs(x)}."
                )

        kv_contexts = self._prepare_kv_context(
            encoder_kv_states,
            encoded_states,
        )

        cross_mask = self._build_cross_attention_mask(
            encoder_attention_mask,
            encoded_states,
            bsz,
            x.dtype,
        )
        self_mask = self._build_self_attention_mask(
            action_attention_mask,
            seq_len,
            x.device,
            x.dtype,
        )
        memory_mask = (
            self._build_memory_attention_mask(
                memory_attention_mask,
                batch_size=bsz,
                memory_length=memory_tokens.shape[1],
                dtype=x.dtype,
                device=x.device,
            )
            if memory_tokens is not None and memory_attention_mask is not None
            else None
        )

        for block_idx, (block, kv_context) in enumerate(zip(self.blocks, kv_contexts)):
            x = block(
                x,
                conditioning,
                cross_kv=kv_context,
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=self.config.causal_attn,
                memory_tokens=memory_tokens,
                memory_attention_mask=memory_mask,
            )
            if valid_action is not None:
                x = x * valid_action
            if not torch.isfinite(x).all():
                k_ctx, v_ctx = kv_context
                raise RuntimeError(
                    "Non-finite hidden state inside KV-conditioned action expert block: "
                    f"block_idx={block_idx}, "
                    f"x_finite={_tensor_all_finite(x)}, "
                    f"k_ctx_finite={_tensor_all_finite(k_ctx)}, "
                    f"v_ctx_finite={_tensor_all_finite(v_ctx)}, "
                    f"conditioning_finite={_tensor_all_finite(conditioning)}, "
                    f"k_ctx_max_abs={_tensor_max_abs(k_ctx)}, "
                    f"v_ctx_max_abs={_tensor_max_abs(v_ctx)}, "
                    f"conditioning_max_abs={_tensor_max_abs(conditioning)}, "
                    f"x_max_abs={_tensor_max_abs(x)}, "
                    f"self_mask_present={self_mask is not None}, "
                    f"cross_mask_present={cross_mask is not None}."
                )

        out = self.final_layer(x, conditioning)
        if valid_action is not None:
            out = out * valid_action
        if not torch.isfinite(out).all():
            raise RuntimeError(
                "Non-finite final action expert output before return: "
                f"out_finite={_tensor_all_finite(out)}, "
                f"x_finite={_tensor_all_finite(x)}, "
                f"conditioning_finite={_tensor_all_finite(conditioning)}, "
                f"out_max_abs={_tensor_max_abs(out)}, "
                f"x_max_abs={_tensor_max_abs(x)}, "
                f"conditioning_max_abs={_tensor_max_abs(conditioning)}."
            )
        return out

    def reset_with_pretrained_weights(self):
        return

    def apply_activation_checkpointing(self):
        self.blocks = nn.ModuleList([checkpoint_wrapper(block) for block in self.blocks])

    def apply_compile(self, **compile_kwargs):
        if self.config.compile == "blocks":
            for block in self.blocks:
                block.compile(**compile_kwargs)
        elif self.config.compile is not None:
            raise NotImplementedError(self.config.compile)

    def apply_fsdp2(self, **fully_shard_kwargs):
        fully_shard(self, **fully_shard_kwargs)
