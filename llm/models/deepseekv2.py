"""
DeepSeek V2 Model Implementation.

Paper: https://arxiv.org/abs/2405.04434

Implements:
- Multi-head Latent Attention (MLA) with LoRA compression
- Mixture of Experts with shared experts
- Group-limited greedy routing
""" 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from llm.core import (
    RMSNorm,
    SwiGLU,
    precompute_freqs_cis,
    apply_rope_with_transpose,
    model_summary,
    init_weights,
)


@dataclass
class DeepseekConfig:
    vocab_size: int = 50280
    seq_len: int = 2048
    d_model: int = 768
    hidden_dim: int = None
    num_heads: int = 2
    num_kv_heads: int = 0
    v_head_dim: int = 64
    q_lora_rank: int = 384
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    kv_lora_rank: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    multiple_of: int = 2
    bias: bool = False
    moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2
    routed_scaling_factor: int = 4
    topk_method: str = "group_limited_greedy"
    n_group: int = 2
    topk_group: int = 1
    n_routed_experts: int = 12
    n_shared_experts: int = 2


class DeepseekAttention(nn.Module):
    """DeepSeek Multi-head Latent Attention with LoRA compression."""

    def __init__(self, model_args: DeepseekConfig):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.res_dropout = nn.Dropout(model_args.dropout)
        self.flash_attn = hasattr(F, "scaled_dot_product_attention")

        self.q_lora_rank = model_args.q_lora_rank
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.kv_lora_rank = model_args.kv_lora_rank
        self.v_head_dim = model_args.v_head_dim
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.q_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim

        # LoRA projections
        self.q_a_proj = nn.Linear(d_model, model_args.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(model_args.q_lora_rank)
        self.q_b_proj = nn.Linear(
            model_args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            d_model, model_args.kv_lora_rank + model_args.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(model_args.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            model_args.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, freqs_cis
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(batch, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        q_pe, k_pe = apply_rope_with_transpose(q_pe, k_pe, freqs_cis)
        k_pe = k_pe.transpose(2, 1)
        q_pe = q_pe.transpose(2, 1)

        query_states = k_pe.new_empty(batch, self.num_heads, seq_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(batch, self.num_heads, seq_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        attn_mtx = (
            torch.matmul(query_states, key_states.transpose(2, 3))
            / math.sqrt(self.head_dim)
        )
        attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
        attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(key_states)
        attn_mtx = self.attn_dropout(attn_mtx)

        output = torch.matmul(attn_mtx, value_states)
        output = output.transpose(1, 2).contiguous().view(
            batch, seq_len, self.num_heads * self.v_head_dim
        )
        output = self.o_proj(output)
        output = self.res_dropout(output)
        return output


class MoEGate(nn.Module):
    """Router for Mixture of Experts with group-limited greedy routing."""

    def __init__(
        self,
        num_experts_per_tok: int,
        n_routed_experts: int,
        routed_scaling_factor: int,
        topk_method: str,
        n_group: int,
        topk_group: int,
        hidden_size: int,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        batch, seq_len, h = x.shape
        hidden_states = x.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(batch * seq_len, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    batch * seq_len,
                    self.n_group,
                    self.n_routed_experts // self.n_group,
                )
                .reshape(batch * seq_len, -1)
            )
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        return topk_idx, topk_weight


class MoE(nn.Module):
    """Mixture of Experts with shared experts."""

    def __init__(
        self,
        dim: int,
        routed_scaling_factor: int,
        topk_method: str,
        n_group: int,
        topk_group: int,
        hidden_dim: int | None = None,
        n_routed_experts: int = 12,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 2,
    ):
        super().__init__()
        self.experts_per_rank = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.experts = nn.ModuleList(
            [SwiGLU(dim, hidden_dim) for _ in range(n_routed_experts)]
        )
        self.gate = MoEGate(
            num_experts_per_tok,
            n_routed_experts,
            routed_scaling_factor,
            topk_method,
            n_group,
            topk_group,
            dim,
        )
        self.shared_experts = SwiGLU(dim, hidden_dim * n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        y = y.type(x.dtype)

        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=x.dtype)

        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)
        output = y + self.shared_experts(identity)
        return output


class Block(nn.Module):
    def __init__(self, model_args: DeepseekConfig):
        super().__init__()
        self.attn = DeepseekAttention(model_args)
        if model_args.moe:
            self.ff = MoE(
                model_args.d_model,
                model_args.routed_scaling_factor,
                model_args.topk_method,
                model_args.n_group,
                model_args.topk_group,
                model_args.multiple_of * model_args.d_model,
                model_args.n_routed_experts,
                model_args.num_experts_per_tok,
                model_args.n_shared_experts,
            )
        else:
            self.ff = SwiGLU(
                dim=model_args.d_model,
                hidden_dim=model_args.hidden_dim,
                dropout=model_args.dropout,
                bias=model_args.bias,
            )
        self.norm1 = RMSNorm(model_args.d_model)
        self.norm2 = RMSNorm(model_args.d_model)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


class Deepseekv2(nn.Module):
    def __init__(self, model_args: DeepseekConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model_args
        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.layers = nn.ModuleList(
            [Block(model_args) for _ in range(model_args.num_layers)]
        )
        self.norm = RMSNorm(model_args.d_model)
        self.vocab_proj = nn.Linear(
            model_args.d_model, model_args.vocab_size, bias=False
        )
        self.token_emb.weight = self.vocab_proj.weight
        self.cis = precompute_freqs_cis(
            model_args.v_head_dim, model_args.seq_len, use_half_dim=True
        )

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
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    device = "mps"
    model = Deepseekv2(DeepseekConfig()).to(device)
    model = torch.compile(model)
    print(model)
    print(model_summary(model))
