"""
Arctic Model Implementation (Hybrid Dense-MoE).

Snowflake's Arctic architecture with dense + MoE parallel residual paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from llm.core import (
    RMSNorm,
    SwiGLU,
    Attention,
    precompute_freqs_cis,
    model_summary,
    init_weights,
)


@dataclass
class HybridConfig:
    vocab_size: int = 50280
    seq_len: int = 2048
    d_model: int = 768
    hidden_dim: int = None
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 8
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = False
    hybrid: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class MoE(nn.Module):
    """Sparse Mixture of Experts layer."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(
            [SwiGLU(dim, hidden_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size * seq_len, dim)
        scores = self.gate(x)
        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1
        )
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        output = torch.empty_like(x, dtype=x.dtype, device=x.device)

        for idx, expert in enumerate(self.experts):
            filtered_x = x[flat_expert_indices == idx]
            output[flat_expert_indices == idx] = expert(filtered_x)

        output = output.view(*expert_weights.shape, -1)
        expert_weights = expert_weights.unsqueeze(-1)
        output = output * expert_weights
        output = output.sum(dim=1)
        return output


class Block(nn.Module):
    """Arctic block with parallel dense + MoE residual paths."""

    def __init__(self, model_args: HybridConfig):
        super().__init__()
        self.attn = Attention(model_args)
        self.ff1 = SwiGLU(
            dim=model_args.d_model,
            hidden_dim=model_args.hidden_dim,
            dropout=model_args.dropout,
            bias=model_args.bias,
        )
        self.ff2 = MoE(
            model_args.d_model,
            model_args.multiple_of * model_args.d_model,
            model_args.num_experts,
            model_args.num_experts_per_tok,
        )
        self.norm1 = RMSNorm(model_args.d_model)
        self.norm2 = RMSNorm(model_args.d_model)

    def forward(self, x, mask, freqs_cis):
        normx = self.norm1(x)
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff1(self.norm2(x)) + self.ff2(normx)
        return x


class Arctic(nn.Module):
    def __init__(self, model_args: HybridConfig, *args, **kwargs) -> None:
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
            model_args.d_model // model_args.num_heads, model_args.seq_len * 2
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
    model = Arctic(HybridConfig()).to(device)
    model = torch.compile(model)
    print(model)
    print(model_summary(model))
