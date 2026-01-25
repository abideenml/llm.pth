"""
Core layer implementations shared across LLM architectures.

Includes:
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU: Swish-Gated Linear Unit feed-forward network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Paper: https://arxiv.org/abs/1910.07467

    Args:
        dim: The dimension of the input tensor
        eps: Small constant for numerical stability
        use_offset: If True, uses (1 + weight) scaling (Gemma2 style)
    """

    def __init__(self, dim: int, eps: float = 1e-5, use_offset: bool = False):
        super().__init__()
        self.eps = eps
        self.use_offset = use_offset
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        if self.use_offset:
            # Gemma2 style: (x * (1 + w)).to(float16)
            output = output * (1.0 + self.weight.float())
        else:
            output = output * self.weight
        return output.type_as(x)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    GLU Variants Improve Transformer
    Paper: https://arxiv.org/abs/2002.05202v1

    The order of W1, W2, W3 multiplication follows the Llama convention.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (if None, computed as 4 * dim * 2/3 rounded to multiple_of)
        multiple_of: Round hidden_dim to this multiple
        dropout: Dropout rate (None for no dropout)
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
