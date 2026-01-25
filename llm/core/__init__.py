"""
Core shared components for LLM architectures.

This module contains reusable building blocks that are shared across
multiple model implementations to reduce code duplication.

Modules:
- layers: RMSNorm, SwiGLU
- rope: RoPE implementations and utilities
- attention: Multi-head attention with GQA
- utils: Model summary, weight initialization, utilities
- dataset: Dataset classes and data utilities
- scheduler: Learning rate schedulers
- sftdata: Supervised fine-tuning data utilities
"""

# Core layers
from .layers import RMSNorm, SwiGLU

# RoPE implementations
from .rope import (
    precompute_freqs_cis,
    reshape_for_broadcast,
    apply_rope,
    apply_rope_with_transpose,
)

# Attention
from .attention import Attention

# Model utilities
from .utils import (
    convert_int_to_shortened_string,
    model_summary,
    init_weights,
    LightningModelWrapper,
)

# Dataset utilities
from .dataset import (
    PreTokenizedDataset,
    TinyShakespeareDataset,
    BetterCycle,
    auto_accelerator,
    build_mask,
    get_tokenizer,
)

# Schedulers
from .scheduler import Scheduler, CosineScheduler

# SFT data utilities
from .sftdata import format_dataset, sft_dataset, get_sft_collate_fn

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
    # Dataset
    "PreTokenizedDataset",
    "TinyShakespeareDataset",
    "BetterCycle",
    "auto_accelerator",
    "build_mask",
    "get_tokenizer",
    # Schedulers
    "Scheduler",
    "CosineScheduler",
    # SFT data
    "format_dataset",
    "sft_dataset",
    "get_sft_collate_fn",
]
