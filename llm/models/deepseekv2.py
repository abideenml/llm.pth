from __future__ import annotations
import math, torch, lightning as L
from dataclasses import dataclass
from typing import Any
from lightning.pytorch.utilities.model_summary import ModelSummary

@dataclass
class DeepseekConfig:
    vocab_size: int = 50280
    seq_len: int = 2048
    d_model: int = 768
    hidden_dim: int =  None
    num_heads: int = 2
    num_kv_heads: int = 0
    v_head_dim: int = 64
    q_lora_rank: int = 384
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    kv_lora_rank: int = 256
    num_layers: int = 2
    dropout: int = 0.2
    multiple_of: int = 2
    bias: int = False
    moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2
    routed_scaling_factor: int = 4
    topk_method: str = "group_limited_greedy"
    n_group: int = 2
    topk_group: int = 1
    n_routed_experts: int = 12
    num_experts_per_tok: int = 4
    n_shared_experts: int = 2


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        bias: bool = False,
    ):
        """
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202v1

        order in which W1,W2,W3 are multiplied is as per llama (for compatiblity)
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.w2(torch.nn.functional.F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        """
        Paper: https://arxiv.org/abs/1910.07467
        """
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim//2, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector
    freqs_cos = torch.cos(freqs)  # real
    freqs_sin = torch.sin(freqs)  # imaginary
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.dim()
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rope(k, q, cis):
    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    _, seq_len, _, _ = q.shape
    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]
    q_cis = q.float().reshape(q.shape[:-1] + (-1, 2))  # (B,T,nhead,C) -> (B,T,nhead,Cc,2) # Cc = C//2
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))  # (B,T,nhead,C) -> (B,T,nhead,Cc,2)
    xq_r, xq_i = q_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))
    xk_r, xk_i = k_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # freqs.shape = (1,T,1,Cc)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # (ac-bd) 
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # (ad+bc) * i
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # (ac-bd)
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # (ad+bc) * i
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xq_out = xq_out.flatten(3)  # (B,T,nhead,C)
    xk_out = xk_out.flatten(3)  # (B,T,nhead,C)
    return xq_out.type_as(q), xk_out.type_as(q)


class Attention(torch.nn.Module):
    def __init__(self, model_args: DeepseekConfig):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.attn_dropout = torch.nn.Dropout(model_args.dropout)
        self.res_dropout = torch.nn.Dropout(model_args.dropout)
        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        
        self.q_lora_rank = model_args.q_lora_rank
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.kv_lora_rank = model_args.kv_lora_rank
        self.v_head_dim = model_args.v_head_dim
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.q_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.q_a_proj = torch.nn.Linear(d_model, model_args.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(model_args.q_lora_rank)
        self.q_b_proj = torch.nn.Linear(model_args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = torch.nn.Linear(d_model,model_args.kv_lora_rank + model_args.qk_rope_head_dim,bias=False,)
        self.kv_a_layernorm = RMSNorm(model_args.kv_lora_rank)
        self.kv_b_proj = torch.nn.Linear(model_args.kv_lora_rank,self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + 
            self.v_head_dim),bias=False,)
        self.o_proj = torch.nn.Linear(self.num_heads * self.v_head_dim,d_model, bias=False,)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(batch, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2))
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_pe, k_pe = apply_rope(q_pe, k_pe, freqs_cis)
        k_pe = k_pe.transpose(2, 1)
        q_pe = q_pe.transpose(2, 1)
        query_states = k_pe.new_empty(batch, self.num_heads, seq_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        key_states = k_pe.new_empty(batch, self.num_heads, seq_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        attn_mtx = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
        attn_mtx = torch.nn.functional.softmax(attn_mtx.float(), dim=-1).type_as(key_states)
        attn_mtx = self.attn_dropout(attn_mtx)
        output = torch.matmul(attn_mtx, value_states)  # (batch, n_head, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.v_head_dim)
        # final projection into the residual stream
        output = self.o_proj(output)
        output = self.res_dropout(output)
        return output

class MoEGate(torch.nn.Module):
    def __init__(self, num_experts_per_tok: int, n_routed_experts: int, routed_scaling_factor: int, topk_method: str, n_group: int, topk_group: int, hidden_size: int):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.weight = torch.nn.Parameter(torch.empty((self.n_routed_experts, hidden_size)))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x: torch.Tensor):
        batch, seq_len, h = x.shape
        hidden_states = x.view(-1, h)
        logits = torch.nn.functional.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = (scores.view(batch * seq_len, self.n_group, -1).max(dim=-1).values)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    batch * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(batch * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        return topk_idx, topk_weight

class MoE(torch.nn.Module):
    def __init__(self, dim: int, routed_scaling_factor: int, topk_method: str, n_group: int, topk_group: int, hidden_dim: int | None = None, n_routed_experts: int = 12, num_experts_per_tok: int = 4, n_shared_experts: int = 2, mlp: str = "swiglu"):
        super().__init__()
        self.experts_per_rank = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        mlp_block = SwiGLU
        self.experts = torch.nn.ModuleList([mlp_block(dim, hidden_dim) for i in range(n_routed_experts)])
        self.gate = MoEGate(num_experts_per_tok, n_routed_experts, routed_scaling_factor, topk_method, n_group, topk_group, dim)
        self.shared_experts = mlp_block(dim, hidden_dim * n_shared_experts)
        
    def forward(self, x: torch.Tensor):
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

class Block(torch.nn.Module):
    def __init__(self, model_args: DeepseekConfig):
        super().__init__()

        self.attn = Attention(model_args)
        if model_args.moe:
            self.ff = MoE(model_args.d_model, model_args.routed_scaling_factor, model_args.topk_method, model_args.n_group, model_args.topk_group, model_args.multiple_of * model_args.d_model, model_args.n_routed_experts, model_args.num_experts_per_tok, model_args.n_shared_experts)
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


class Deepseekv2(torch.nn.Module):
    def __init__(self, model_args: DeepseekConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model_args
        self.token_emb = torch.nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.layers = torch.nn.ModuleList([Block(model_args) for _ in range(model_args.num_layers)])
        self.norm = RMSNorm(model_args.d_model)
        self.vocab_proj = torch.nn.Linear(model_args.d_model, model_args.vocab_size, bias=False)
        self.token_emb.weight = self.vocab_proj.weight
        self.cis = precompute_freqs_cis(model_args.v_head_dim, model_args.seq_len)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, model_args.seq_len, model_args.seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

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
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class Model(L.LightningModule):
    def __init__(self, model, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

def convert_int_to_shortened_string(num):
    if abs(num) < 1000:
        return str(num)
    elif abs(num) < 1000000:
        return f"{num / 1000:.1f}K"
    elif abs(num) < 1000000000:
        return f"{num / 1000000:.1f}M"
    elif abs(num) < 1000000000000:
        return f"{num / 1000000000:.1f}B"
    else:
        return f"{num / 1000000000000:.1f}T"

def model_summary(model: torch.nn.Module, print_summary=False):
    "conver normal model to lightning model and print model summary"
    model = Model(model)
    summary = ModelSummary(model)
    if print_summary:
        print(summary)
    return {
        "summary": summary,
        "total_parameters": convert_int_to_shortened_string(summary.total_parameters),
        "trainable_parameters": convert_int_to_shortened_string(summary.trainable_parameters),
    }


if __name__ == "__main__":
    device = "mps"
    model = Deepseekv2(DeepseekConfig).to(device)
    model = torch.compile(model)
    print(model)
    print(model_summary(model))