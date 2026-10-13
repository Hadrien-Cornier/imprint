"""Hybrid Gated DeltaNet memory encoder.

The long-term stream is a chronological sequence of finalized Wan chunks and
their aligned causal action/proprio samples. A recurrent delta-rule state
compresses that sequence. Language-conditioned learned queries read the state,
then ordinary cross-attention fuses the exact recent visual/action/proprio
window. The output contract is identical to ``CosmoBottleneckMemory``:
``(batch, num_latents, memory_token_dim)``.

The module supports two execution modes:

* ``__call__`` scans a complete padded history for Stage-1 and Stage-2 training.
* ``update_incremental`` and ``read_incremental`` maintain a persistent state
  for online inference without rescanning finalized history.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.representation.utils import kernel_init

_MATMUL_PRECISION = jax.lax.Precision.HIGHEST
_PARAM_DTYPE = jnp.float32
_NORM_DTYPE = jnp.float32


def _l2_normalize(x, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def _mask_or_ones(mask, batch, length):
    if mask is None:
        return jnp.ones((batch, length), dtype=jnp.bool_)
    return mask.astype(jnp.bool_)


def _rematerialized_scan(step, carry, inputs, checkpoint_size):
    """Checkpoint a recurrent scan in fixed-size groups to bound backward memory."""
    length = inputs[0].shape[0]
    checkpoint_size = int(checkpoint_size)
    if checkpoint_size <= 0:
        raise ValueError("scan checkpoint size must be positive")
    if length <= checkpoint_size:
        return jax.lax.scan(step, carry, inputs, unroll=length)

    grouped_length = (length // checkpoint_size) * checkpoint_size
    group_count = grouped_length // checkpoint_size
    grouped_inputs = jax.tree.map(
        lambda value: value[:grouped_length].reshape(
            group_count,
            checkpoint_size,
            *value.shape[1:],
        ),
        inputs,
    )

    def scan_group(group_carry, group_inputs):
        return jax.lax.scan(
            step,
            group_carry,
            group_inputs,
            unroll=checkpoint_size,
        )

    checkpointed_group = jax.checkpoint(
        scan_group,
        prevent_cse=False,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    carry, grouped_outputs = jax.lax.scan(
        checkpointed_group,
        carry,
        grouped_inputs,
    )
    outputs = jax.tree.map(
        lambda value: value.reshape(grouped_length, *value.shape[2:]),
        grouped_outputs,
    )

    if grouped_length < length:
        remainder_length = length - grouped_length
        remainder_inputs = jax.tree.map(
            lambda value: value[grouped_length:],
            inputs,
        )

        def scan_remainder(remainder_carry, values):
            return jax.lax.scan(
                step,
                remainder_carry,
                values,
                unroll=remainder_length,
            )

        carry, remainder_outputs = jax.checkpoint(
            scan_remainder,
            prevent_cse=False,
            policy=jax.checkpoint_policies.nothing_saveable,
        )(carry, remainder_inputs)
        outputs = jax.tree.map(
            lambda grouped, remainder: jnp.concatenate([grouped, remainder], axis=0),
            outputs,
            remainder_outputs,
        )
    return carry, outputs


def _delta_recurrent_step(state, values):
    q_t, k_t, v_t, beta_t, decay_t, valid_t = values
    retained = state * decay_t[..., None, None]
    prediction = jnp.einsum(
        "bhvk,bhk->bhv",
        retained,
        k_t,
        precision=_MATMUL_PRECISION,
    )
    error = v_t - prediction
    candidate = retained + beta_t[..., None, None] * jnp.einsum(
        "bhv,bhk->bhvk",
        error,
        k_t,
        precision=_MATMUL_PRECISION,
    )
    valid_matrix = valid_t[:, None, None, None]
    next_state = jnp.where(valid_matrix, candidate, state)
    output = jnp.einsum(
        "bhvk,bhk->bhv",
        next_state,
        q_t,
        precision=_MATMUL_PRECISION,
    )
    output = jnp.where(valid_t[:, None, None], output, jnp.zeros_like(output))
    return next_state, output


def _recurrent_delta_scan(matrix, q, k, v, beta, decay, mask, checkpoint_size):
    """Reference event-wise recurrence used by incremental evaluation."""
    scan_inputs = tuple(
        jnp.swapaxes(value, 0, 1)
        for value in (q, k, v, beta, decay, mask.astype(jnp.bool_))
    )
    next_matrix, output = _rematerialized_scan(
        _delta_recurrent_step,
        matrix.astype(jnp.float32),
        scan_inputs,
        checkpoint_size,
    )
    return next_matrix, jnp.swapaxes(output, 0, 1)


def _wy_einsum(equation, *operands, matmul_dtype):
    operands = tuple(value.astype(matmul_dtype) for value in operands)
    return jnp.einsum(
        equation,
        *operands,
        precision=_MATMUL_PRECISION,
        preferred_element_type=jnp.float32,
    )


def _chunkwise_wy_delta_chunk(matrix, q, k, v, beta, decay, mask, *, matmul_dtype):
    """Apply one exact Gated DeltaNet chunk with a WY triangular solve.

    ``matrix`` is the recurrent state before the chunk. The returned outputs
    match the event-wise recurrence, while all event updates inside the chunk
    are represented by one lower-triangular system.
    """
    _, chunk_size, _, _ = q.shape
    q = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)
    k = jnp.transpose(k, (0, 2, 1, 3)).astype(jnp.float32)
    v = jnp.transpose(v, (0, 2, 1, 3)).astype(jnp.float32)
    beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
    decay = jnp.transpose(decay, (0, 2, 1)).astype(jnp.float32)
    valid = mask[:, None, :].astype(jnp.bool_)

    beta = jnp.where(valid, beta, 0.0)
    decay = jnp.where(valid, decay, 1.0)
    cumulative_decay = jnp.cumprod(decay, axis=-1)
    scaled_v = v / cumulative_decay[..., None]

    matrix_k = _wy_einsum(
        "bhvk,bhtk->bhtv",
        matrix,
        k,
        matmul_dtype=matmul_dtype,
    )
    rhs = beta[..., None] * (scaled_v - matrix_k)

    key_gram = _wy_einsum(
        "bhtk,bhsk->bhts",
        k,
        k,
        matmul_dtype=matmul_dtype,
    )
    system = (
        jnp.eye(chunk_size, dtype=jnp.float32)
        + beta[..., None] * jnp.tril(key_gram, k=-1)
    )
    updates = jax.lax.linalg.triangular_solve(
        system,
        rhs,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=True,
    )

    query_key = _wy_einsum(
        "bhtk,bhsk->bhts",
        q,
        k,
        matmul_dtype=matmul_dtype,
    )
    contribution = _wy_einsum(
        "bhts,bhsv->bhtv",
        jnp.tril(query_key),
        updates,
        matmul_dtype=matmul_dtype,
    )
    initial_read = _wy_einsum(
        "bhvk,bhtk->bhtv",
        matrix,
        q,
        matmul_dtype=matmul_dtype,
    )
    output = (initial_read + contribution) * cumulative_decay[..., None]
    output = jnp.where(valid[..., None], output, jnp.zeros_like(output))

    normalized_final = matrix + _wy_einsum(
        "bhtv,bhtk->bhvk",
        updates,
        k,
        matmul_dtype=matmul_dtype,
    )
    next_matrix = normalized_final * cumulative_decay[..., -1, None, None]
    return next_matrix, jnp.transpose(output, (0, 2, 1, 3))


def _chunkwise_wy_delta_scan(
    matrix,
    q,
    k,
    v,
    beta,
    decay,
    mask,
    chunk_size,
    *,
    matmul_dtype=jnp.float32,
):
    """Scan chunks recurrently while evaluating each chunk in parallel."""
    length = q.shape[1]
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("WY chunk size must be positive")
    padded_length = ((length + chunk_size - 1) // chunk_size) * chunk_size
    padding = padded_length - length

    def pad_time(value, fill_value=0):
        pad_width = [(0, 0)] * value.ndim
        pad_width[1] = (0, padding)
        return jnp.pad(value, pad_width, constant_values=fill_value)

    q = pad_time(q)
    k = pad_time(k)
    v = pad_time(v)
    beta = pad_time(beta)
    decay = pad_time(decay, fill_value=1)
    mask = pad_time(mask.astype(jnp.bool_))
    num_chunks = padded_length // chunk_size

    def as_chunks(value):
        reshaped = value.reshape(
            value.shape[0],
            num_chunks,
            chunk_size,
            *value.shape[2:],
        )
        return jnp.swapaxes(reshaped, 0, 1)

    chunk_inputs = tuple(as_chunks(value) for value in (q, k, v, beta, decay, mask))

    def scan_chunk(state, values):
        return _chunkwise_wy_delta_chunk(
            state,
            *values,
            matmul_dtype=matmul_dtype,
        )

    checkpointed_chunk = jax.checkpoint(
        scan_chunk,
        prevent_cse=False,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    next_matrix, chunk_outputs = jax.lax.scan(
        checkpointed_chunk,
        matrix.astype(jnp.float32),
        chunk_inputs,
    )
    output = jnp.swapaxes(chunk_outputs, 0, 1).reshape(
        q.shape[0],
        padded_length,
        q.shape[2],
        q.shape[3],
    )
    return next_matrix, output[:, :length]


class _MlpTokenizer(nnx.Module):
    def __init__(self, in_dim, hidden, out_dim, rngs, dtype):
        self.l1 = nnx.Linear(
            in_dim,
            hidden,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )
        self.l2 = nnx.Linear(
            hidden,
            out_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )

    def __call__(self, x):
        return self.l2(nnx.gelu(self.l1(x)))


class _CausalDepthwiseConv(nnx.Module):
    """Small causal depthwise convolution with an explicit streaming cache."""

    def __init__(self, dim, kernel_size, dtype):
        self.kernel_size = int(kernel_size)
        kernel = jnp.zeros((self.kernel_size, dim), dtype=_PARAM_DTYPE)
        kernel = kernel.at[-1].set(jnp.ones((dim,), dtype=_PARAM_DTYPE))
        self.kernel = nnx.Param(kernel)
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=_PARAM_DTYPE))

    def __call__(self, x, cache=None):
        batch, length, dim = x.shape
        kernel = self.kernel.value.astype(x.dtype)
        bias = self.bias.value.astype(x.dtype)
        cache_length = self.kernel_size - 1
        if cache is None:
            cache = jnp.zeros((batch, cache_length, dim), dtype=x.dtype)
        if cache_length == 0:
            return x * kernel[0] + bias, cache

        joined = jnp.concatenate([cache.astype(x.dtype), x], axis=1)
        mixed = jnp.zeros_like(x)
        for offset in range(self.kernel_size):
            mixed = mixed + joined[:, offset : offset + length] * kernel[offset]
        return mixed + bias, joined[:, -cache_length:]


class _GatedDeltaBlock(nnx.Module):
    """Reference recurrent Gated DeltaNet block implemented with ``jax.lax.scan``."""

    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        conv_kernel,
        ffn_inner,
        decay_bias,
        beta_bias,
        scan_checkpoint_size,
        wy_matmul_dtype,
        rngs,
        dtype,
    ):
        if dim != num_heads * head_dim:
            raise ValueError(f"delta heads must cover model width: {num_heads} * {head_dim} != {dim}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.decay_bias = float(decay_bias)
        self.beta_bias = float(beta_bias)
        self.scan_checkpoint_size = int(scan_checkpoint_size)
        if self.scan_checkpoint_size <= 0:
            raise ValueError("delta scan checkpoint size must be positive")
        self.wy_matmul_dtype = wy_matmul_dtype

        self.norm_in = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.conv = _CausalDepthwiseConv(dim, conv_kernel, dtype)
        linear_kwargs = {
            "rngs": rngs,
            "dtype": dtype,
            "param_dtype": _PARAM_DTYPE,
            "kernel_init": kernel_init,
            "precision": _MATMUL_PRECISION,
        }
        self.q_proj = nnx.Linear(dim, dim, **linear_kwargs)
        self.k_proj = nnx.Linear(dim, dim, **linear_kwargs)
        self.v_proj = nnx.Linear(dim, dim, **linear_kwargs)
        self.beta_proj = nnx.Linear(dim, num_heads, **linear_kwargs)
        self.decay_proj = nnx.Linear(dim, num_heads, **linear_kwargs)
        self.out_gate = nnx.Linear(dim, dim, **linear_kwargs)
        self.out_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.out_proj = nnx.Linear(dim, dim, **linear_kwargs)

        self.ffn_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.ffn_in = nnx.Linear(dim, ffn_inner, **linear_kwargs)
        self.ffn_out = nnx.Linear(ffn_inner, dim, **linear_kwargs)

    def initial_matrix(self, batch):
        return jnp.zeros(
            (batch, self.num_heads, self.head_dim, self.head_dim),
            dtype=jnp.float32,
        )

    def initial_conv_cache(self, batch, dtype):
        return jnp.zeros(
            (batch, self.conv.kernel_size - 1, self.dim),
            dtype=dtype,
        )

    def __call__(self, x, mask, matrix=None, conv_cache=None, *, scan_mode="recurrent"):
        batch, length, _ = x.shape
        if matrix is None:
            matrix = self.initial_matrix(batch)
        if conv_cache is None:
            conv_cache = self.initial_conv_cache(batch, x.dtype)

        normalized = self.norm_in(x)
        mixed, next_conv_cache = self.conv(normalized, conv_cache)
        mixed = jax.nn.silu(mixed)

        def split_heads(value):
            return value.reshape(batch, length, self.num_heads, self.head_dim)

        q = _l2_normalize(split_heads(self.q_proj(mixed))).astype(jnp.float32)
        k = _l2_normalize(split_heads(self.k_proj(mixed))).astype(jnp.float32)
        v = split_heads(self.v_proj(mixed)).astype(jnp.float32)
        beta = jax.nn.sigmoid(self.beta_proj(mixed).astype(jnp.float32) + self.beta_bias)
        decay = jnp.exp(
            -jax.nn.softplus(self.decay_proj(mixed).astype(jnp.float32) + self.decay_bias)
        )

        if scan_mode == "recurrent":
            next_matrix, output = _recurrent_delta_scan(
                matrix,
                q,
                k,
                v,
                beta,
                decay,
                mask,
                self.scan_checkpoint_size,
            )
        elif scan_mode == "chunkwise_wy":
            next_matrix, output = _chunkwise_wy_delta_scan(
                matrix,
                q,
                k,
                v,
                beta,
                decay,
                mask,
                self.scan_checkpoint_size,
                matmul_dtype=self.wy_matmul_dtype,
            )
        else:
            raise ValueError(f"unsupported gated delta scan mode: {scan_mode!r}")
        output = output.reshape(batch, length, self.dim).astype(x.dtype)

        gated = self.out_norm(output) * jax.nn.sigmoid(self.out_gate(mixed))
        x = x + self.out_proj(gated)
        x = x + self.ffn_out(nnx.gelu(self.ffn_in(self.ffn_norm(x))))
        x = jnp.where(mask[..., None], x, jnp.zeros_like(x))
        return x, next_matrix, next_conv_cache


class _RecentFusionBlock(nnx.Module):
    def __init__(self, dim, num_heads, rngs, dtype):
        self.cross_q_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.cross_kv_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            precision=_MATMUL_PRECISION,
        )
        self.self_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            precision=_MATMUL_PRECISION,
        )
        self.ffn_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.ffn_in = nnx.Linear(
            dim,
            2 * dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )
        self.ffn_out = nnx.Linear(
            2 * dim,
            dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )

    def __call__(self, queries, key_values, key_value_mask):
        attn_mask = key_value_mask[:, None, None, :]
        queries = queries + self.cross_attn(
            self.cross_q_norm(queries),
            self.cross_kv_norm(key_values),
            mask=attn_mask,
        )
        normalized = self.self_norm(queries)
        queries = queries + self.self_attn(normalized, normalized)
        return queries + self.ffn_out(nnx.gelu(self.ffn_in(self.ffn_norm(queries))))


class GatedDeltaBottleneckMemory(nnx.Module):
    """Long-term recurrent memory with exact recent-window fusion."""

    architecture_name = "gated_delta_hybrid_v1"

    def __init__(self, config, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        cb = config.cosmo_bottleneck_memory
        architecture_version = str(
            getattr(cb, "memory_architecture_version", self.architecture_name)
        )
        if architecture_version != self.architecture_name:
            raise ValueError(
                "gated delta architecture version mismatch: "
                f"{architecture_version!r} != {self.architecture_name!r}"
            )
        event_order = str(
            getattr(
                cb,
                "delta_event_order",
                "action_proprio_then_finalized_visual_v1",
            )
        )
        if event_order != "action_proprio_then_finalized_visual_v1":
            raise ValueError(f"unsupported gated delta event order: {event_order!r}")

        dim = int(config.memory_token_dim)
        self.dtype = dtype
        self.dim = dim
        self.num_latents = int(cb.num_latents)
        self.num_views = int(getattr(config, "num_views", 1))
        self.chunk_frames = int(getattr(cb, "temporal_factor", 1))
        pool_h = int(getattr(cb, "pool_h", 0) or getattr(cb, "pool_hw", 0) or 1)
        pool_w = int(getattr(cb, "pool_w", 0) or getattr(cb, "pool_hw", 0) or 1)
        self.tokens_per_chunk = self.num_views * pool_h * pool_w
        self.events_per_chunk = 2 * self.chunk_frames + self.tokens_per_chunk
        self.max_chunks = int(getattr(cb, "delta_max_chunks", 384))
        if self.chunk_frames <= 0 or self.tokens_per_chunk <= 0 or self.max_chunks <= 0:
            raise ValueError(
                "temporal_factor, pooled tokens per chunk, and delta_max_chunks must be positive"
            )

        num_layers = int(getattr(cb, "delta_num_layers", 2))
        num_heads = int(getattr(cb, "delta_num_heads", getattr(cb, "num_heads", 8)))
        head_dim = int(getattr(cb, "delta_head_dim", dim // num_heads))
        conv_kernel = int(getattr(cb, "delta_conv_kernel", 4))
        ffn_inner = int(getattr(cb, "delta_ffn_inner", 2 * dim))
        decay_bias = float(getattr(cb, "delta_decay_bias", -4.0))
        beta_bias = float(getattr(cb, "delta_beta_bias", -1.0))
        scan_checkpoint_size = int(
            getattr(cb, "delta_scan_checkpoint_size", self.events_per_chunk)
        )
        self.full_scan_mode = str(getattr(cb, "delta_full_scan_mode", "recurrent"))
        if self.full_scan_mode not in {"recurrent", "chunkwise_wy"}:
            raise ValueError(
                f"unsupported gated delta full scan mode: {self.full_scan_mode!r}"
            )
        wy_matmul_dtype_name = str(
            getattr(cb, "delta_wy_matmul_dtype", "float32")
        )
        try:
            wy_matmul_dtype = {
                "float32": jnp.float32,
                "bfloat16": jnp.bfloat16,
            }[wy_matmul_dtype_name]
        except KeyError as error:
            raise ValueError(
                "delta_wy_matmul_dtype must be 'float32' or 'bfloat16', "
                f"got {wy_matmul_dtype_name!r}"
            ) from error

        self.proj_cosmo = nnx.Linear(
            int(cb.cosmos_latent_dim),
            dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )
        self.proj_recent = nnx.Linear(
            int(cb.siglip_dim),
            dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )
        self.tok_action = _MlpTokenizer(
            int(cb.action_in_dim), int(cb.action_tokenizer_hidden), dim, rngs, dtype
        )
        self.tok_proprio = _MlpTokenizer(
            int(cb.proprio_in_dim), int(cb.proprio_tokenizer_hidden), dim, rngs, dtype
        )
        self.lang_film = nnx.Linear(
            int(cb.lang_emb_dim),
            2 * dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )

        self.latents = nnx.Param(
            0.02
            * jnp.asarray(
                np.random.default_rng(0).standard_normal((self.num_latents, dim)),
                _PARAM_DTYPE,
            )
        )
        self.event_type = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (3, dim), dtype=_PARAM_DTYPE)
        )
        self.event_slot = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.events_per_chunk, dim),
                dtype=_PARAM_DTYPE,
            )
        )
        self.time_proj = nnx.Linear(
            1,
            dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )

        self.delta_blocks = [
            _GatedDeltaBlock(
                dim,
                num_heads,
                head_dim,
                conv_kernel,
                ffn_inner,
                decay_bias,
                beta_bias,
                scan_checkpoint_size,
                wy_matmul_dtype,
                rngs,
                dtype,
            )
            for _ in range(num_layers)
        ]
        self.read_norm = nnx.LayerNorm(
            dim,
            rngs=rngs,
            dtype=_NORM_DTYPE,
            param_dtype=_PARAM_DTYPE,
        )
        self.read_q = nnx.Linear(
            dim,
            dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )
        self.read_out = nnx.Linear(
            dim,
            dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=_PARAM_DTYPE,
            kernel_init=kernel_init,
            precision=_MATMUL_PRECISION,
        )
        self.read_heads = num_heads
        self.read_head_dim = head_dim

        self.recent_null = nnx.Param(jnp.zeros((1, dim), dtype=_PARAM_DTYPE))
        recent_blocks = int(getattr(cb, "delta_recent_fusion_blocks", 1))
        self.recent_blocks = [
            _RecentFusionBlock(dim, int(getattr(cb, "num_heads", num_heads)), rngs, dtype)
            for _ in range(recent_blocks)
        ]

    def _chunk_type_ids(self):
        action_proprio = jnp.tile(jnp.asarray([0, 1], dtype=jnp.int32), self.chunk_frames)
        visual = jnp.full((self.tokens_per_chunk,), 2, dtype=jnp.int32)
        return jnp.concatenate([action_proprio, visual], axis=0)

    def _decorate_chunk_events(self, events, chunk_positions):
        type_embedding = self.event_type.value[
            self._chunk_type_ids()
        ].astype(self.dtype)
        slot_embedding = self.event_slot.value.astype(self.dtype)
        normalized_time = (
            chunk_positions.astype(jnp.float32) / max(float(self.max_chunks - 1), 1.0)
        )[..., None]
        time_embedding = self.time_proj(normalized_time.astype(self.dtype))
        return (
            events
            + type_embedding[None, None, :, :]
            + slot_embedding[None, None, :, :]
            + time_embedding[:, :, None, :]
        )

    def _build_full_events(
        self,
        cosmo_emb,
        action_emb,
        proprio_emb,
        cosmo_mask,
        action_mask,
        proprio_mask,
    ):
        batch = cosmo_emb.shape[0]
        static_chunks = min(
            cosmo_emb.shape[1] // self.tokens_per_chunk,
            action_emb.shape[1] // self.chunk_frames,
            proprio_emb.shape[1] // self.chunk_frames,
            self.max_chunks,
        )
        if static_chunks <= 0:
            raise ValueError("gated delta memory requires room for at least one complete chunk")

        valid_cosmo = jnp.sum(cosmo_mask.astype(jnp.int32), axis=1)
        num_chunks = jnp.clip(valid_cosmo // self.tokens_per_chunk, 0, static_chunks)
        finalized_steps = num_chunks * self.chunk_frames
        chunk_valid = jnp.arange(static_chunks)[None, :] < num_chunks[:, None]

        cosmo = self.proj_cosmo(
            cosmo_emb[:, : static_chunks * self.tokens_per_chunk]
        ).reshape(batch, static_chunks, self.tokens_per_chunk, self.dim)
        cosmo_valid = cosmo_mask[:, : static_chunks * self.tokens_per_chunk].reshape(
            batch, static_chunks, self.tokens_per_chunk
        )
        cosmo_valid = cosmo_valid & chunk_valid[:, :, None]

        action = self.tok_action(
            action_emb[:, : static_chunks * self.chunk_frames]
        ).reshape(batch, static_chunks, self.chunk_frames, self.dim)
        proprio = self.tok_proprio(
            proprio_emb[:, : static_chunks * self.chunk_frames]
        ).reshape(batch, static_chunks, self.chunk_frames, self.dim)
        action_valid = action_mask[:, : static_chunks * self.chunk_frames].reshape(
            batch, static_chunks, self.chunk_frames
        )
        proprio_valid = proprio_mask[:, : static_chunks * self.chunk_frames].reshape(
            batch, static_chunks, self.chunk_frames
        )
        action_valid = action_valid & chunk_valid[:, :, None]
        proprio_valid = proprio_valid & chunk_valid[:, :, None]

        action_proprio = jnp.stack([action, proprio], axis=3).reshape(
            batch, static_chunks, 2 * self.chunk_frames, self.dim
        )
        action_proprio_valid = jnp.stack([action_valid, proprio_valid], axis=3).reshape(
            batch, static_chunks, 2 * self.chunk_frames
        )
        events = jnp.concatenate([action_proprio, cosmo], axis=2)
        event_mask = jnp.concatenate([action_proprio_valid, cosmo_valid], axis=2)
        chunk_positions = jnp.broadcast_to(
            jnp.arange(static_chunks, dtype=jnp.int32)[None, :],
            (batch, static_chunks),
        )
        events = self._decorate_chunk_events(events, chunk_positions)
        events = jnp.where(event_mask[..., None], events, jnp.zeros_like(events))
        return (
            events.reshape(batch, static_chunks * self.events_per_chunk, self.dim),
            event_mask.reshape(batch, static_chunks * self.events_per_chunk),
            num_chunks,
            finalized_steps,
        )

    def _build_incremental_events(
        self,
        cosmo_chunk,
        action_chunk,
        proprio_chunk,
        chunk_positions,
        cosmo_mask,
        action_mask,
        proprio_mask,
    ):
        batch = cosmo_chunk.shape[0]
        if cosmo_chunk.shape[1] != self.tokens_per_chunk:
            raise ValueError(
                f"expected {self.tokens_per_chunk} finalized visual tokens, got {cosmo_chunk.shape[1]}"
            )
        if action_chunk.shape[1] != self.chunk_frames or proprio_chunk.shape[1] != self.chunk_frames:
            raise ValueError(
                f"expected {self.chunk_frames} action/proprio steps per finalized chunk"
            )

        action = self.tok_action(action_chunk)
        proprio = self.tok_proprio(proprio_chunk)
        action_proprio = jnp.stack([action, proprio], axis=2).reshape(
            batch, 2 * self.chunk_frames, self.dim
        )
        action_proprio_mask = jnp.stack([action_mask, proprio_mask], axis=2).reshape(
            batch, 2 * self.chunk_frames
        )
        events = jnp.concatenate([action_proprio, self.proj_cosmo(cosmo_chunk)], axis=1)
        event_mask = jnp.concatenate([action_proprio_mask, cosmo_mask], axis=1)
        events = self._decorate_chunk_events(
            events[:, None, :, :],
            chunk_positions[:, None],
        )[:, 0]
        return jnp.where(event_mask[..., None], events, jnp.zeros_like(events)), event_mask

    def _scan_events(
        self,
        events,
        event_mask,
        matrices=None,
        conv_caches=None,
        *,
        scan_mode="recurrent",
    ):
        batch = events.shape[0]
        if matrices is None:
            matrices = [
                block.initial_matrix(batch)
                for block in self.delta_blocks
            ]
        else:
            matrices = [matrices[:, i] for i in range(len(self.delta_blocks))]
        if conv_caches is None:
            conv_caches = [
                block.initial_conv_cache(batch, events.dtype)
                for block in self.delta_blocks
            ]
        else:
            conv_caches = [conv_caches[:, i] for i in range(len(self.delta_blocks))]

        next_matrices = []
        next_caches = []
        hidden = events
        for block, matrix, cache in zip(
            self.delta_blocks, matrices, conv_caches, strict=True
        ):
            hidden, next_matrix, next_cache = block(
                hidden,
                event_mask,
                matrix,
                cache,
                scan_mode=scan_mode,
            )
            next_matrices.append(next_matrix)
            next_caches.append(next_cache)
        return (
            hidden,
            jnp.stack(next_matrices, axis=1),
            jnp.stack(next_caches, axis=1),
        )

    def _language_queries(self, batch, lang_emb):
        queries = jnp.broadcast_to(
            self.latents.value.astype(self.dtype)[None],
            (batch, self.num_latents, self.dim),
        )
        if lang_emb is not None:
            gamma, beta = jnp.split(
                self.lang_film(lang_emb.astype(self.dtype)),
                2,
                axis=-1,
            )
            queries = queries * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return queries

    def _read_matrix(self, matrices, lang_emb):
        batch = matrices.shape[0]
        queries = self._language_queries(batch, lang_emb)
        q = self.read_q(self.read_norm(queries)).reshape(
            batch,
            self.num_latents,
            self.read_heads,
            self.read_head_dim,
        )
        q = _l2_normalize(q).astype(jnp.float32)
        matrix = matrices[:, -1].astype(jnp.float32)
        read = jnp.einsum("bhvk,bmhk->bmhv", matrix, q)
        read = read.reshape(batch, self.num_latents, self.dim).astype(self.dtype)
        return queries + self.read_out(read)

    def _fuse_recent(
        self,
        queries,
        recent_emb,
        action_emb,
        proprio_emb,
        recent_mask,
        action_mask,
        proprio_mask,
    ):
        batch = queries.shape[0]
        recent_null = self.recent_null.value.astype(self.dtype)
        null = jnp.broadcast_to(recent_null[None], (batch, 1, self.dim))
        key_values = jnp.concatenate(
            [
                null,
                self.proj_recent(recent_emb),
                self.tok_action(action_emb),
                self.tok_proprio(proprio_emb),
            ],
            axis=1,
        )
        key_value_mask = jnp.concatenate(
            [
                jnp.ones((batch, 1), dtype=jnp.bool_),
                recent_mask,
                action_mask,
                proprio_mask,
            ],
            axis=1,
        )
        for block in self.recent_blocks:
            queries = block(queries, key_values, key_value_mask)
        return queries

    def init_incremental_state(self, batch_size, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        matrices = jnp.stack(
            [block.initial_matrix(batch_size) for block in self.delta_blocks],
            axis=1,
        )
        conv_caches = jnp.stack(
            [
                block.initial_conv_cache(batch_size, dtype)
                for block in self.delta_blocks
            ],
            axis=1,
        )
        chunk_count = jnp.zeros((batch_size,), dtype=jnp.int32)
        return matrices, conv_caches, chunk_count

    def update_incremental(
        self,
        state,
        cosmo_chunk,
        action_chunk,
        proprio_chunk,
        *,
        cosmo_mask=None,
        action_mask=None,
        proprio_mask=None,
    ):
        matrices, conv_caches, chunk_count = state
        batch = cosmo_chunk.shape[0]
        cosmo_mask = _mask_or_ones(cosmo_mask, batch, cosmo_chunk.shape[1])
        action_mask = _mask_or_ones(action_mask, batch, action_chunk.shape[1])
        proprio_mask = _mask_or_ones(proprio_mask, batch, proprio_chunk.shape[1])
        events, event_mask = self._build_incremental_events(
            cosmo_chunk,
            action_chunk,
            proprio_chunk,
            chunk_count,
            cosmo_mask,
            action_mask,
            proprio_mask,
        )
        _, matrices, conv_caches = self._scan_events(
            events,
            event_mask,
            matrices,
            conv_caches,
            scan_mode="recurrent",
        )
        valid_chunk = jnp.any(cosmo_mask, axis=1)
        chunk_count = chunk_count + valid_chunk.astype(jnp.int32)
        return matrices, conv_caches, chunk_count

    def read_incremental(
        self,
        state,
        recent_emb,
        action_recent,
        proprio_recent,
        *,
        lang_emb=None,
        recent_mask=None,
        action_mask=None,
        proprio_mask=None,
    ):
        matrices, _, chunk_count = state
        batch = recent_emb.shape[0]
        recent_mask = _mask_or_ones(recent_mask, batch, recent_emb.shape[1])
        action_mask = _mask_or_ones(action_mask, batch, action_recent.shape[1])
        proprio_mask = _mask_or_ones(proprio_mask, batch, proprio_recent.shape[1])
        queries = self._read_matrix(matrices, lang_emb)
        queries = self._fuse_recent(
            queries,
            recent_emb,
            action_recent,
            proprio_recent,
            recent_mask,
            action_mask,
            proprio_mask,
        )
        stats = {"delta/finalized_chunks": chunk_count.astype(jnp.float32)}
        return queries, jnp.ones((batch, self.num_latents), dtype=jnp.bool_), stats

    def __call__(
        self,
        cosmo_emb,
        recent_emb,
        action_emb,
        proprio_emb,
        cosmo_mask,
        lang_emb=None,
        recent_mask=None,
        action_mask=None,
        proprio_mask=None,
    ):
        batch = cosmo_emb.shape[0]
        recent_mask = _mask_or_ones(recent_mask, batch, recent_emb.shape[1])
        action_mask = _mask_or_ones(action_mask, batch, action_emb.shape[1])
        proprio_mask = _mask_or_ones(proprio_mask, batch, proprio_emb.shape[1])
        cosmo_mask = cosmo_mask.astype(jnp.bool_)

        events, event_mask, num_chunks, finalized_steps = self._build_full_events(
            cosmo_emb,
            action_emb,
            proprio_emb,
            cosmo_mask,
            action_mask,
            proprio_mask,
        )
        _, matrices, _ = self._scan_events(
            events,
            event_mask,
            scan_mode=self.full_scan_mode,
        )

        step_positions = jnp.arange(action_emb.shape[1], dtype=jnp.int32)[None, :]
        recent_action_mask = action_mask & (step_positions >= finalized_steps[:, None])
        recent_proprio_mask = proprio_mask & (step_positions >= finalized_steps[:, None])
        queries = self._read_matrix(matrices, lang_emb)
        queries = self._fuse_recent(
            queries,
            recent_emb,
            action_emb,
            proprio_emb,
            recent_mask,
            recent_action_mask,
            recent_proprio_mask,
        )
        stats = {
            "delta/finalized_chunks": num_chunks.astype(jnp.float32),
            "delta/recent_steps": jnp.sum(recent_action_mask, axis=1).astype(jnp.float32),
        }
        return queries, jnp.ones((batch, self.num_latents), dtype=jnp.bool_), stats
