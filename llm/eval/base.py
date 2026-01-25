"""
Base classes and utilities for LLM evaluation benchmarks.

Provides:
- BaseEvaluator: Abstract base class for all evaluators
- MultipleChoiceEvaluator: Base class for multiple choice benchmarks
- Utility functions for computing metrics
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from tqdm import tqdm
import requests


@dataclass
class EvalResult:
    """Container for evaluation results."""

    benchmark: str
    accuracy: float
    accuracy_norm: float = None
    num_correct: int = 0
    num_correct_norm: int = 0
    num_total: int = 0
    per_category: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        result = f"{self.benchmark}: {self.accuracy:.4f}"
        if self.accuracy_norm is not None:
            result += f" (norm: {self.accuracy_norm:.4f})"
        result += f" [{self.num_correct}/{self.num_total}]"
        return result


def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Download a file from URL with progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=os.path.basename(fname),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def compute_completion_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the loss for completions given logits, tokens, and mask.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        tokens: Input token ids [batch, seq_len]
        mask: Binary mask where 1 indicates completion region [batch, seq_len]
        reduction: 'mean' for average loss, 'sum' for total loss

    Returns:
        Loss tensor [batch] with loss for each completion
    """
    # Shift for autoregressive loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # Compute per-token loss
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_tokens, reduction="none")
    per_token_loss = per_token_loss.view(tokens.size(0), -1)

    # Apply mask and reduce
    masked_loss = per_token_loss * shift_mask

    if reduction == "sum":
        return masked_loss.sum(dim=1)
    else:  # mean
        return masked_loss.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)


def get_model_output(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Get model logits, handling both HuggingFace and custom model interfaces.

    Args:
        model: The model (HuggingFace or custom)
        tokens: Input tokens [batch, seq_len]
        device: Device to use

    Returns:
        Logits tensor [batch, seq_len, vocab_size]
    """
    if device is not None:
        tokens = tokens.to(device)

    with torch.no_grad():
        output = model(tokens)

        # Handle different output formats
        if hasattr(output, "logits"):
            # HuggingFace style
            return output.logits
        elif isinstance(output, tuple):
            # Custom model returning (logits, loss) or similar
            return output[0]
        else:
            # Direct logits output
            return output


class BaseEvaluator(ABC):
    """Abstract base class for evaluation benchmarks."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        device: torch.device = None,
        batch_size: int = 1,
        cache_dir: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._auto_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), self.benchmark_name
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Name of the benchmark."""
        pass

    @abstractmethod
    def load_data(self, split: str = "val") -> Iterator[dict]:
        """Load benchmark data."""
        pass

    @abstractmethod
    def evaluate(self, split: str = "val", **kwargs) -> EvalResult:
        """Run evaluation and return results."""
        pass

    def _auto_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class MultipleChoiceEvaluator(BaseEvaluator):
    """
    Base class for multiple-choice evaluation benchmarks.

    Handles the common pattern of:
    1. Context + multiple completions
    2. Compute loss for each completion
    3. Select lowest loss as prediction
    """

    @abstractmethod
    def render_example(self, example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Render an example into tokens, mask, and label.

        Returns:
            tokens: [num_choices, seq_len] token ids
            mask: [num_choices, seq_len] binary mask for completion region
            label: correct answer index
        """
        pass

    def predict(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[int, int, torch.Tensor]:
        """
        Make prediction for a multiple choice example.

        Returns:
            pred: prediction using sum of losses
            pred_norm: prediction using normalized (average) loss
            losses: per-choice losses
        """
        tokens = tokens.to(self.device)
        mask = mask.to(self.device)

        logits = get_model_output(self.model, tokens, self.device)
        avg_loss = compute_completion_loss(logits, tokens, mask, reduction="mean")
        sum_loss = compute_completion_loss(logits, tokens, mask, reduction="sum")

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        return pred, pred_norm, avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "val",
        max_examples: int = None,
        verbose: bool = True,
    ) -> EvalResult:
        """
        Run evaluation on the benchmark.

        Args:
            split: Data split to evaluate on
            max_examples: Maximum number of examples (for debugging)
            verbose: Whether to print progress

        Returns:
            EvalResult with accuracy metrics
        """
        self.model.eval()
        self.model.to(self.device)

        num_correct = 0
        num_correct_norm = 0
        num_total = 0

        examples = self.load_data(split)
        if verbose:
            examples = tqdm(examples, desc=f"Evaluating {self.benchmark_name}")

        for i, example in enumerate(examples):
            if max_examples is not None and i >= max_examples:
                break

            tokens, mask, label = self.render_example(example)
            pred, pred_norm, losses = self.predict(tokens, mask)

            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)

        accuracy = num_correct / num_total if num_total > 0 else 0
        accuracy_norm = num_correct_norm / num_total if num_total > 0 else 0

        return EvalResult(
            benchmark=self.benchmark_name,
            accuracy=accuracy,
            accuracy_norm=accuracy_norm,
            num_correct=num_correct,
            num_correct_norm=num_correct_norm,
            num_total=num_total,
        )
