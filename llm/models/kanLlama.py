import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary
import math
from dataclasses import dataclass
from kan import *

@dataclass
class Config:
    vocab_size: int = 50280
    seq_len: int = 2048
    d_model: int = 768
    hidden_dim: int =  None
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 6
    dropout: int = 0.2
    multiple_of: int = 2
    bias: int = False

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
        
        self.w1 = KAN(width=[dim,5,hidden_dim], grid=5, k=3, seed=0)
        self.w2 = KAN(width=[hidden_dim,5,dim], grid=5, k=3, seed=0)
        self.w3 = KAN(width=[dim,5,hidden_dim], grid=5, k=3, seed=0)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [: (dim // 2)] for odd number truncation
    # torch.arange(0, dim, 2) -> 2(i-1)//d while i= 1,2,..,(d//2)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector

    # e^it = cos(t) + i sin(t)
    freqs_cos = torch.cos(freqs)  # real
    freqs_sin = torch.sin(freqs)  # imaginary
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.dim()
    assert 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"

    # keep 2nd (T) and last(freq) dim same else make dim 1 for freq_cis
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # print(shape)
    return freqs_cis.view(shape)


def apply_rope(k, q, cis):
    # Idea suppose vector v = [x,y,x1,y1,...] # v.shape = dim
    # convert vetor into complex num # ie two vec one real, one imagery
    # [x,y,x1,y1,...] -> x+iy, x1+iy1
    # Multiplying by complex num == roatate vector
    # => (x + iy) * (cos + isin) -> x'+iy'
    # restack
    # x'+iy' -> [x',y',x1',y1'...]
    # you roated vector in chunks of two lfg!!!
    _, seq_len, _, _ = q.shape

    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]

    #  rehsape a shape (...,n )-> (..., n//2,2)
    q_cis = q.float().reshape(
        q.shape[:-1] + (-1, 2)
    )  # (B,T,nhead,C) -> (B,T,nhead,Cc,2) # Cc = C//2
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))  # (B,T,nhead,C) -> (B,T,nhead,Cc,2)

    xq_r, xq_i = q_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc)) split into two tuple
    xk_r, xk_i = k_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # freqs.shape = (1,T,1,Cc)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # e+if = (a+ib) * (c+di) = (ac-bd) + i (ad+bc)
    # a = xq_r , b = xq_i
    # c = fcos , d = fsin
    # ...
    # e = (ac-bd) = xq_r * freqs_cos - xq_i * freqs_sin
    # f = (c+di)  = xq_r * freqs_sin + xq_i * freqs_cos

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # (ac-bd)   # shape =  # (B,T,nhead,Cc)
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # (ad+bc) * i
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # (ac-bd)
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # (ad+bc) * i

    # now we stack r,i -> [r,i,r2,i2]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)  # (B,T,nhead,Cc,2)

    # flatten last two dimensions
    xq_out = xq_out.flatten(3)  # (B,T,nhead,C)
    xk_out = xk_out.flatten(3)  # (B,T,nhead,C)

    return xq_out.type_as(q), xk_out.type_as(q)


class Attention(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.num_kv_heads = (
            model_args.num_heads if model_args.num_kv_heads == 0 else model_args.num_kv_heads
        )
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = KAN(width=[d_model,5,self.head_dim * self.num_heads], grid=5, k=3, seed=0, device='cpu')
        self.query = KAN(width=[d_model,5,self.head_dim * self.num_kv_heads], grid=5, k=3, seed=0, device='cpu')
        self.value = KAN(width=[d_model,5,self.head_dim * self.num_kv_heads], grid=5, k=3, seed=0, device='cpu')
        self.proj = KAN(width=[d_model,5,d_model], grid=5, k=3, seed=0, device='cpu')

        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.res_dropout = nn.Dropout(model_args.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order impotent
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()

        self.attn = Attention(model_args)
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


class LLAMA(nn.Module):
    def __init__(self, model_args: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = model_args

        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)

        self.layers = nn.ModuleList([Block(model_args) for _ in range(model_args.num_layers)])

        self.norm = RMSNorm(model_args.d_model)
        self.vocab_proj = nn.Linear(model_args.d_model, model_args.vocab_size, bias=False)
        
        
        self.token_emb.weight = self.vocab_proj.weight

        self.cis = precompute_freqs_cis(
            model_args.d_model // model_args.num_heads, model_args.seq_len * 2
        )

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, model_args.seq_len, model_args.seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        batch, seqlen = x.shape
        x = self.token_emb(x)
        device = self.token_emb.weight.device
        freqs_cis = self.cis[0][:seqlen].to(device), self.cis[1][:seqlen].to(device)

        for layer in self.layers:
            x = layer(x, self.mask, freqs_cis)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class Model(L.LightningModule):
    def __init__(self, model, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model


def convert_int_to_shortened_string(num):
    """
    Converts an integer into a shortened format using suffixes like K, M, B, and T for
    thousands, millions, billions, and trillions, respectively, with one decimal place.

    Args:
    - num (int): The number to convert.

    Returns:
    - str: The converted, shortened number as a string.
    """
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
    device = "cpu"
    model = LLAMA(Config).to(device)
    model = torch.compile(model)

    print("-" * 100)
    print(model)
    print(model_summary(model))
    print("-" * 100)