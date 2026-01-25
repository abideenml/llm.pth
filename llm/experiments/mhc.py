from __future__ import annotations
import math, torch, lightning as L
from dataclasses import dataclass
from typing import Any
from lightning.pytorch.utilities.model_summary import ModelSummary

from llm.core import (
    RMSNorm,
    SwiGLU,
    precompute_freqs_cis,
    apply_rope,
)

@dataclass
class Deepseekv32Config:
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
    # DSA Configs
    dsa_topk: int = 64
    dsa_index_dim: int = 64
    dsa_index_heads: int = 4



def sinkhorn_knopp(logits, iters=20, eps=1e-6):
    M = torch.exp(logits)
    for _ in range(iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class LightningIndexer(torch.nn.Module):
    def __init__(self, d_model, d_index, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(d_model, num_heads * d_index, bias=False) # (d_model, H * dI)
        self.k_proj = torch.nn.Linear(d_model, d_index, bias=False) # (d_model, dI)
        self.w_proj = torch.nn.Linear(d_model, num_heads, bias=False) # (d_model, H)

    def forward(self, x):
        """
        x: (B, T, d_model)
        returns: index_scores (B, T, T)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, -1)   # (B,T,H,dI)
        k = self.k_proj(x)                                 # (B,T,dI)
        w = self.w_proj(x)                                 # (B,T,H)

        scores = []
        for j in range(self.num_heads):
            # (B,T,dI) @ (B,dI,T) → (B,T,T)
            dot = torch.matmul(q[:, :, j], k.transpose(1, 2))
            scores.append(w[:, :, j].unsqueeze(-1) * torch.relu(dot))

        return sum(scores)  # I_{t,s}


class TokenMHC(torch.nn.Module):
    def __init__(self, n_streams: int, stream_dim: int):
        super().__init__()
        self.n = n_streams
        self.c = stream_dim
        d = n_streams * stream_dim

        self.alpha_pre  = torch.nn.Parameter(torch.tensor(0.01))
        self.alpha_post = torch.nn.Parameter(torch.tensor(0.01))
        self.alpha_res  = torch.nn.Parameter(torch.tensor(0.01))

        self.phi_pre  = torch.nn.Linear(d, n_streams, bias=False)
        self.phi_post = torch.nn.Linear(d, n_streams, bias=False)
        self.phi_res  = torch.nn.Linear(d, n_streams * n_streams, bias=False)

        self.b_pre  = torch.nn.Parameter(torch.zeros(n_streams))
        self.b_post = torch.nn.Parameter(torch.zeros(n_streams))
        self.b_res  = torch.nn.Parameter(torch.zeros(n_streams, n_streams))

        self.norm = RMSNorm(d)

    def forward(self, x):
        """
        x: [B, T, n_streams, stream_dim]
        """
        B, T, n, c = x.shape
        x_flat = self.norm(x.reshape(B*T, -1))

        H_pre  = torch.sigmoid(
            self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        ).view(B*T, 1, n)

        H_post = (2.0 * torch.sigmoid(
            self.alpha_post * self.phi_post(x_flat) + self.b_post
        )).view(B*T, n, 1)

        H_res_logits = (
            self.alpha_res * self.phi_res(x_flat)
        ).view(B*T, n, n) + self.b_res

        H_res = sinkhorn_knopp(H_res_logits)

        return H_pre, H_post, H_res


class Attention(torch.nn.Module):
    def __init__(self, model_args: Deepseekv32Config):
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

        self.dsa_topk = model_args.dsa_topk
        self.indexer = LightningIndexer(d_model, model_args.dsa_index_dim, model_args.dsa_index_heads)

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

        # ---- DSA INDEXING ----
        index_scores = self.indexer(x)  # (B, T, T)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        index_scores = index_scores.masked_fill(causal_mask == 0, float("-inf"))
        topk = self.dsa_topk
        _, topk_idx = torch.topk(index_scores, k=topk, dim=-1)  # (B, T, K)
        
        # key_states: (B, num_heads, T, q_head_dim)
        # topk_idx: (B, T, K) - for each query position, indices of top-k keys
        # We want k_sparse: (B, num_heads, T, K, q_head_dim)
        
        # Permute key_states to (B, T, num_heads, q_head_dim) for easier indexing
        key_states_permuted = key_states.permute(0, 2, 1, 3)  # (B, T, num_heads, q_head_dim)
        value_states_permuted = value_states.permute(0, 2, 1, 3)  # (B, T, num_heads, v_head_dim)
        batch_idx = torch.arange(batch, device=x.device)[:, None, None]  # (B, 1, 1)
        k_sparse = key_states_permuted[batch_idx, topk_idx]  # (B, T, K, num_heads, q_head_dim)
        v_sparse = value_states_permuted[batch_idx, topk_idx]  # (B, T, K, num_heads, v_head_dim)
        
        # Permute back to (B, num_heads, T, K, head_dim)
        k_sparse = k_sparse.permute(0, 3, 1, 2, 4)  # (B, num_heads, T, K, q_head_dim)
        v_sparse = v_sparse.permute(0, 3, 1, 2, 4)  # (B, num_heads, T, K, v_head_dim)
        # Compute attention scores: (B, num_heads, T, K)
        scores = (query_states.unsqueeze(-2) * k_sparse).sum(dim=-1)  # (B, num_heads, T, K)
        scores /= math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)  # (B, num_heads, T, K)
        attn = self.attn_dropout(attn)
        
        # Apply attention weights to values
        # attn: (B, num_heads, T, K)
        # v_sparse: (B, num_heads, T, K, v_head_dim)
        # Multiply: (B, num_heads, T, K, 1) * (B, num_heads, T, K, v_head_dim) = (B, num_heads, T, K, v_head_dim)
        output = (attn.unsqueeze(-1) * v_sparse).sum(dim=-2)  # Sum over K dimension -> (B, num_heads, T, v_head_dim)
        
        # Reshape for output projection
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
    def __init__(self, model_args: Deepseekv32Config):
        super().__init__()

        self.attn = Attention(model_args)

        if model_args.moe:
            self.ff = MoE(model_args.d_model, model_args.routed_scaling_factor, model_args.topk_method, model_args.n_group, model_args.topk_group, model_args.multiple_of * model_args.d_model, model_args.n_routed_experts, model_args.num_experts_per_tok, model_args.n_shared_experts)
        else:
            self.ff = SwiGLU(model_args.d_model, model_args.hidden_dim, model_args.dropout, model_args.bias)

        self.norm1 = RMSNorm(model_args.d_model)
        self.norm2 = RMSNorm(model_args.d_model)

        # ---- mHC config ----
        self.n_streams = 8
        assert model_args.d_model % self.n_streams == 0
        self.stream_dim = model_args.d_model // self.n_streams

        self.mhc_attn = TokenMHC(self.n_streams, self.stream_dim)
        self.mhc_ff   = TokenMHC(self.n_streams, self.stream_dim)

    def apply_mhc(self, x, fx, mhc):
        """
        x, fx: [B, T, d_model]
        """
        B, T, D = x.shape
        n, c = self.n_streams, self.stream_dim

        x_s  = x.view(B, T, n, c)
        fx_s = fx.view(B, T, n, c)

        H_pre, H_post, H_res = mhc(x_s)

        x_flat  = x_s.reshape(B*T, n, c)
        fx_flat = fx_s.reshape(B*T, n, c)

        layer_in = torch.bmm(H_pre, x_flat)           # [BT,1,c]
        layer_out = fx_flat.mean(dim=1, keepdim=True) # already computed

        out = (
            torch.bmm(H_res, x_flat)
            + torch.bmm(H_post, layer_out)
        )

        return out.view(B, T, D)

    def forward(self, x, mask, freqs_cis):
        # ---- Attention with mHC ----
        attn_out = self.attn(self.norm1(x), mask, freqs_cis)
        x = self.apply_mhc(x, attn_out, self.mhc_attn)

        # ---- FFN / MoE with mHC ----
        ff_out = self.ff(self.norm2(x))
        x = self.apply_mhc(x, ff_out, self.mhc_ff)

        return x


class Deepseekv32(torch.nn.Module):
    def __init__(self, model_args: Deepseekv32Config, *args, **kwargs) -> None:
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
    model = Deepseekv32(Deepseekv32Config).to(device)
    model = torch.compile(model)
    print(model)
    print(model_summary(model))