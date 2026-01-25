"""
Gemma2 Model Implementation.

Paper: https://arxiv.org/abs/2408.00118

Implements:
- Soft-capping for attention logits
- Soft-capping for final logits
- Sliding window attention (alternating layers)
- Offset-based RMSNorm ((1 + w) scaling)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from llm.core import (
    RMSNorm,
    SwiGLU,
    precompute_freqs_cis,
    apply_rope,
    model_summary,
    init_weights,
)


@dataclass
class Gemma2Config:
    vocab_size: int = 50280
    seq_len: int = 2048
    d_model: int = 768
    hidden_dim: int = None
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 6
    dropout: float = 0.2
    multiple_of: int = 2
    bias: bool = False
    query_pre_attn_scalar: int = 256
    sliding_window: int = 4096
    attn_logit_softcapping: int = 50
    final_logit_softcapping: int = 30


class Gemma2Attention(nn.Module):
    """Gemma2 Attention with soft-capping and sliding window support."""

    def __init__(self, model_args: Gemma2Config, layer_idx: Optional[int] = None):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.num_kv_heads = (
            model_args.num_heads if model_args.num_kv_heads == 0 else model_args.num_kv_heads
        )
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_heads)
        self.query = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.proj = nn.Linear(d_model, d_model, model_args.bias)

        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.res_dropout = nn.Dropout(model_args.dropout)
        self.flash_attn = hasattr(F, "scaled_dot_product_attention")

        self.layer_idx = layer_idx
        self.scaling = model_args.query_pre_attn_scalar**-0.5
        self.sliding_window = (
            model_args.sliding_window if not bool(layer_idx % 2) else None
        )
        self.attn_logit_softcapping = model_args.attn_logit_softcapping

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, freqs_cis
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(batch, seq_len, -1, self.head_dim)
        q = q.view(batch, seq_len, -1, self.head_dim)
        v = v.view(batch, seq_len, -1, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Soft-capping for attention logits
        if self.attn_logit_softcapping is not None:
            attn_mtx = attn_mtx / self.attn_logit_softcapping
            attn_mtx = torch.tanh(attn_mtx)
            attn_mtx = attn_mtx * self.attn_logit_softcapping

        attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
        attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
        attn_mtx = self.attn_dropout(attn_mtx)

        output = torch.matmul(attn_mtx, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, model_args: Gemma2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.attn = Gemma2Attention(model_args, layer_idx)
        self.ff = SwiGLU(
            dim=model_args.d_model,
            hidden_dim=model_args.hidden_dim,
            dropout=model_args.dropout,
            bias=model_args.bias,
        )
        self.is_sliding = not bool(layer_idx % 2)
        # Gemma2 uses offset-based RMSNorm
        self.norm1 = RMSNorm(model_args.d_model, use_offset=True)
        self.norm2 = RMSNorm(model_args.d_model, use_offset=True)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


class Gemma2(nn.Module):
    def __init__(self, model_args: Gemma2Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model_args
        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.layers = nn.ModuleList(
            [Block(model_args, layer_idx) for layer_idx in range(model_args.num_layers)]
        )
        self.norm = RMSNorm(model_args.d_model, use_offset=True)
        self.vocab_proj = nn.Linear(
            model_args.d_model, model_args.vocab_size, bias=False
        )
        self.token_emb.weight = self.vocab_proj.weight
        self.cis = precompute_freqs_cis(
            model_args.d_model // model_args.num_heads, model_args.seq_len * 2
        )
        self.final_logit_softcapping = model_args.final_logit_softcapping
        self.sliding_window = model_args.sliding_window

        if hasattr(F, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, model_args.seq_len, model_args.seq_len), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, targets=None):
        batch, seqlen = x.shape
        x = self.token_emb(x)
        device = self.token_emb.weight.device
        freqs_cis = self.cis[0][:seqlen].to(device), self.cis[1][:seqlen].to(device)

        for layer in self.layers:
            x = layer(x, self.mask, freqs_cis)

        x = self.norm(x)
        logits = self.vocab_proj(x)

        # Soft-capping for final logits
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    device = "mps"
    model = Gemma2(Gemma2Config()).to(device)
    model = torch.compile(model)
    print(model)
    print(model_summary(model))
