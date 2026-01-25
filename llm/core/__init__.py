"""
Core shared components for LLM architectures.

This module contains reusable building blocks that are shared across
multiple model implementations to reduce code duplication.
"""

from .layers import RMSNorm, SwiGLU
from .rope import precompute_freqs_cis, reshape_for_broadcast, apply_rope, apply_rope_with_transpose
from .attention import Attention
from .utils import (
    convert_int_to_shortened_string,
    model_summary,
    init_weights,
    LightningModelWrapper,
)

__all__ = [
    # Layers
    "RMSNorm",
    "SwiGLU",
    # RoPE
    "precompute_freqs_cis",
    "reshape_for_broadcast",
    "apply_rope",
    "apply_rope_with_transpose",
    # Attention
    "Attention",
    # Utils
    "convert_int_to_shortened_string",
    "model_summary",
    "init_weights",
    "LightningModelWrapper",
]
