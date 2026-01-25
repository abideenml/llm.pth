"""
Attention mechanism implementations shared across LLM architectures.

Includes standard multi-head attention with support for:
- Grouped Query Attention (GQA)
- Flash Attention (when available)
- Rotary Position Embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Protocol, Optional

from .rope import apply_rope


class AttentionConfig(Protocol):
    """Protocol for attention configuration."""

    d_model: int
    num_heads: int
    num_kv_heads: int
    dropout: float
    bias: bool


class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention support.

    Supports both Flash Attention (PyTorch 2.0+) and standard attention.
    Implements GQA by repeating key/value heads when num_kv_heads < num_heads.

    Args:
        config: Configuration object with attention parameters
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_kv_heads = (
            config.num_heads if config.num_kv_heads == 0 else config.num_kv_heads
        )
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_heads)
        self.query = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.proj = nn.Linear(d_model, d_model, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Attention mask (optional, used when flash attention unavailable)
            freqs_cis: Tuple of (freqs_cos, freqs_sin) for RoPE

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(batch, seq_len, -1, self.head_dim)
        q = q.view(batch, seq_len, -1, self.head_dim)
        v = v.view(batch, seq_len, -1, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention: repeat KV heads
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        # Transpose for attention: (B, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)
            output = torch.matmul(attn_mtx, v)

        # Restore shape: (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.proj(output)
        output = self.res_dropout(output)
        return output
