"""
Rotary Position Embedding (RoPE) implementations.

Paper: https://arxiv.org/abs/2104.09864

Provides functions for computing and applying rotary position embeddings
to query and key tensors in attention mechanisms.
"""

import torch


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_half_dim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency tensor for RoPE.

    Args:
        dim: The dimension of the embedding
        end: The maximum sequence length
        theta: The base for the exponential (default 10000.0)
        use_half_dim: If True, use dim//2 in the arange (DeepSeek style)

    Returns:
        Tuple of (freqs_cos, freqs_sin) tensors
    """
    if use_half_dim:
        freqs = 1.0 / (theta ** (torch.arange(0, dim // 2, 2)[: (dim // 2)].float() / dim))
    else:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # e^it = cos(t) + i sin(t)
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, check_shape: bool = True
) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with input tensor.

    Args:
        freqs_cis: Frequency tensor to reshape
        x: Input tensor to broadcast with
        check_shape: If True, assert shape compatibility

    Returns:
        Reshaped frequency tensor
    """
    ndim = x.dim()
    assert 1 < ndim

    if check_shape:
        assert freqs_cis.shape == (
            x.shape[1],
            x.shape[-1],
        ), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"

    # Keep 2nd (T) and last (freq) dim same, else make dim 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cis: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.

    This is the standard RoPE implementation used by Llama, Qwen, Mixtral, etc.
    Input shape: (B, T, num_heads, head_dim)

    Args:
        q: Query tensor of shape (B, T, num_heads, head_dim)
        k: Key tensor of shape (B, T, num_heads, head_dim)
        cis: Tuple of (freqs_cos, freqs_sin) from precompute_freqs_cis

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    _, seq_len, _, _ = q.shape
    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]

    # Reshape to complex representation: (B,T,nhead,C) -> (B,T,nhead,C//2,2)
    q_cis = q.float().reshape(q.shape[:-1] + (-1, 2))
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))

    # Split into real and imaginary parts
    xq_r, xq_i = q_cis.unbind(-1)
    xk_r, xk_i = k_cis.unbind(-1)

    # Reshape frequencies for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply rotation using complex multiplication:
    # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Recombine real and imaginary parts
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(q), xk_out.type_as(k)


def apply_rope_with_transpose(
    q: torch.Tensor, k: torch.Tensor, cis: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding with initial transpose.

    This variant is used by DeepSeek models where the input tensors need
    to be transposed before applying RoPE.
    Input shape: (B, num_heads, T, head_dim)

    Args:
        q: Query tensor of shape (B, num_heads, T, head_dim)
        k: Key tensor of shape (B, num_heads, T, head_dim)
        cis: Tuple of (freqs_cos, freqs_sin) from precompute_freqs_cis

    Returns:
        Tuple of (rotated_q, rotated_k) with shape (B, T, num_heads, head_dim)
    """
    # Transpose to (B, T, num_heads, head_dim)
    k = k.transpose(1, 2)
    q = q.transpose(1, 2)

    _, seq_len, _, _ = q.shape
    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]

    # Reshape to complex representation
    q_cis = q.float().reshape(q.shape[:-1] + (-1, 2))
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))

    # Split into real and imaginary parts
    xq_r, xq_i = q_cis.unbind(-1)
    xk_r, xk_i = k_cis.unbind(-1)

    # Reshape frequencies for broadcasting (skip shape check for DeepSeek)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r, check_shape=False)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r, check_shape=False)

    # Apply rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Recombine
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(q), xk_out.type_as(k)
