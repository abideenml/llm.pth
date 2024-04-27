from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary


@dataclass
class MOEConfig:
    vocab_size: int = 51200
    seq_len: int = 2048
    d_model: int = 768
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True
    eps: float = 1e-5
    rotary_dim: float = 0.4
    moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 2


class KVCache:
    def __init__(self, shape, max_seq_length, idx: int | None = None, device=None, dtype=None):
        self.idx = idx
        self.key: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.value: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_length = max_seq_length

    def forward(self, keys: Tensor, values: Tensor, start_pos: Tensor) -> tuple[Tensor, Tensor]:
        bsz, T, _, _ = keys.shape
        # print(f"start_pos={start_pos}, T={T}, bsz={bsz}")
        self.key[:bsz, start_pos : start_pos + T] = keys
        self.value[:bsz, start_pos : start_pos + T] = values
        keys = self.key[:bsz, : start_pos + T]
        values = self.value[:bsz, : start_pos + T]
        return keys, values


class RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()

        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def _extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}"

    def _compute_rope(self, costheta, sintheta, x) -> Tensor:
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = torch.concatenate([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = torch.concatenate([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        assert self.dims < x.shape[-1], "RoPE doesn't implement partial traditional application"

        rx = torch.cat([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def forward(self, x: Tensor, offset: int = 0):
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1])
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N,
            self.dims,
            offset=offset,
            base=self.base,
            scale=self.scale,
            dtype=x.dtype,
            device=x.device,
        )

        rope = self._compute_traditional_rope if self.traditional else self._compute_rope
        rx = rope(costheta, sintheta, x)

        return torch.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        scale: float = 1.0,
        dtype=torch.float32,
        device=None,
    ):
        D = D // 2
        positions = torch.arange(offset, N, dtype=dtype, device=device) * scale
        freqs = torch.exp(-torch.arange(0.0, D, dtype=dtype, device=device) * (math.log(base) / D))
        theta = torch.reshape(positions, (-1, 1)) * torch.reshape(freqs, (1, -1))
        return torch.cos(theta), torch.sin(theta)


def new_gelu(x: Tensor) -> Tensor:
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = new_gelu(x)
        return self.fc2(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        """
        Paper: https://arxiv.org/abs/1910.07467
        """
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MHA(nn.Module):
    def __init__(self, layer_idx, dim, num_heads, rotary_dim) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)

        self.rope = RoPE(int(rotary_dim * self.head_dim), traditional=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        kv_cache: KVCache | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_length, d_model = x.shape
        # print(f"{batch_size}, {seq_length}, {d_model}")
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        if kv_cache is not None:
            k, v = kv_cache.forward(k, v, position_ids)

        k: Tensor = k.transpose(1, 2).to(torch.float32)  # shape = (B, num_heads, seq_len, head_dim)
        q: Tensor = q.transpose(1, 2).to(torch.float32)
        v: Tensor = v.transpose(1, 2).to(torch.float32)

        offset = position_ids if kv_cache else 0

        q = self.rope.forward(q, offset=offset).to(torch.float32)
        k = self.rope.forward(k).to(torch.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / q.shape[-1])
        scores = (q @ k.transpose(-1, -2)) * scale

        if mask is not None:
            mask = mask[:, :, :seq_length, :seq_length]
            scores = scores + mask

        scores = torch.softmax(scores, dim=-1).type_as(v)
        output = (scores @ v).type_as(x)
        output = output.transpose(1, 2).type_as(x)
        output = output.reshape(batch_size, seq_length, self.dim)

        return self.dense(output)

    def scaled_dot_product_attention(self, q, k, v, mask):
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores, dim=-1).type_as(v)
        output = scores @ v
        return output

class SwiGLU(nn.Module):
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

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        mlp: str = "swiglu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        mlp_block = SwiGLU

        self.experts = nn.ModuleList([mlp_block(dim, hidden_dim) for i in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape  # (batch_size, seq_len, dim)

        # (batch_size , seq_len, dim) -> (batch_size * seq_len, dim)
        x = x.view(batch_size * seq_len, dim)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len, num_experts)
        scores = self.gate(x)

        # expert_weights -> (batch_size * seq_len, num_experts_per_tok)
        # expert_indices -> (batch_size * seq_len, num_experts_per_tok)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)

        # -> (batch_size * seq_len, num_experts_per_tok)
        expert_weights = expert_weights.softmax(dim=-1)

        #  -> (batch_size * seq_len * num_experts_per_tok ) 1D
        flat_expert_indices = expert_indices.view(-1)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len * num_experts_per_tok, dim)
        # create copied of inputs for each expert
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)

        # (total_tokens,dim)
        output = torch.empty_like(x, dtype=x.dtype, device=x.device)

        for idx, expert in enumerate(self.experts):
            # filtered_x - selected toks that to be sent to nth expert
            filtered_x = x[flat_expert_indices == idx]
            output[flat_expert_indices == idx] = expert(filtered_x)

        # ->B,T,num_experts_per_tok,dim
        output = output.view(*expert_weights.shape, -1)
        expert_weights = expert_weights.unsqueeze(-1)

        output = output * expert_weights

        # sum up experts outputs
        # batch_size * seq_len, num_experts_per_tok, dim -> batch_size * seq_len, dim
        output = output.sum(dim=1)

        return output

class Block(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()

        self.attn = MHA(config)
        if config.moe:
            self.ff = MoE(config.d_model, config.multiple_of * config.d_model, config.num_experts, config.num_experts_per_tok)
        else: 
            self.ff = SwiGLU(
                dim=config.d_model,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                bias=config.bias,
            )

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x



class Mixtral(nn.Module):
    def __init__(self, config: MOEConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_layers)])

        self.ln = RMSNorm(config.d_model, eps=config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        mask = torch.full((1, 1, config.seq_len, config.seq_len), float("-inf"))

        (
            1,
            config.seq_len,
            config.num_heads,
            config.d_model // config.num_heads,
        )

        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x, kv_cache: list[KVCache] | None = None, position_ids=None):
        mask = self.mask
        if kv_cache is not None:
            x = x[:, position_ids:]
            mask = None
        # print(f"{x.shape=} {kv_cache=}")
        x = self.wte(x)
        # print(f"{x.sum(dim=-1)=}")
        cache = None
        for idx, layer in enumerate(self.layers):
            if kv_cache is not None:
                cache = kv_cache[idx]
            x = layer(x.to(self.wte.weight.dtype), mask, cache, position_ids=position_ids)

        x = self.ln(x)
        x = self.lm_head(x)
        return x

    def loss(self, logits: Tensor, labels: Tensor):
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def build_kv_cache(self) -> list[KVCache]:
        shape = (
            1,
            self.config.seq_len,
            self.config.num_heads,
            self.config.d_model // self.config.num_heads,
        )
        kv_cache = []
        dtype = self.wte.weight.dtype
        device = self.wte.weight.device

        for idx in range(self.config.num_layers):
            kv_cache.append(KVCache(shape, self.config.seq_len, idx, device=device, dtype=dtype))
        return kv_cache

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


def model_summary(model: nn.Module, print_summary=False):
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
    model = Mixtral(MOEConfig).to(device)
    model = torch.compile(model)

    print("-" * 100)
    print(model)
    print(model_summary(model))
    print("-" * 100)