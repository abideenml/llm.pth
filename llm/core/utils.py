"""
Utility functions shared across LLM implementations.

Includes:
- Parameter counting and formatting
- Model summary utilities
- Weight initialization
- Lightning module wrapper
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Any
from lightning.pytorch.utilities.model_summary import ModelSummary


def convert_int_to_shortened_string(num: int) -> str:
    """
    Convert a number to a human-readable shortened string.

    Args:
        num: The number to convert

    Returns:
        Shortened string representation (e.g., "1.5M", "2.3B")
    """
    if abs(num) < 1000:
        return str(num)
    elif abs(num) < 1_000_000:
        return f"{num / 1000:.1f}K"
    elif abs(num) < 1_000_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) < 1_000_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    else:
        return f"{num / 1_000_000_000_000:.1f}T"


class LightningModelWrapper(L.LightningModule):
    """
    Wrapper to convert a PyTorch model to a Lightning module.

    Used primarily for model summary functionality.
    """

    def __init__(self, model: nn.Module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model


# Alias for backward compatibility
Model = LightningModelWrapper


def model_summary(model: nn.Module, print_summary: bool = False) -> dict:
    """
    Generate a summary of model parameters.

    Args:
        model: The PyTorch model to summarize
        print_summary: If True, print the summary to stdout

    Returns:
        Dictionary with 'summary', 'total_parameters', and 'trainable_parameters'
    """
    wrapped = LightningModelWrapper(model)
    summary = ModelSummary(wrapped)
    if print_summary:
        print(summary)
    return {
        "summary": summary,
        "total_parameters": convert_int_to_shortened_string(summary.total_parameters),
        "trainable_parameters": convert_int_to_shortened_string(
            summary.trainable_parameters
        ),
    }


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Initialize weights for a module.

    Linear layers use normal initialization with specified std.
    Embedding layers use normal initialization with specified std.
    Biases are initialized to zero.

    Args:
        module: The module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
