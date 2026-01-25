"""
Learning rate schedulers for LLM training.

Includes:
- Scheduler: Constant learning rate
- CosineScheduler: Cosine annealing with warmup
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Scheduler:
    """Constant learning rate scheduler."""

    learning_rate: float = 3e-4

    def __call__(self, *args: Any, **kwds: Any) -> float:
        return self.learning_rate


@dataclass
class CosineScheduler:
    """
    Cosine annealing learning rate scheduler with warmup.

    Implements the learning rate schedule from the Chinchilla paper:
    - Linear warmup from 0 to learning_rate
    - Cosine decay from learning_rate to min_lr
    - Constant min_lr after max_iters

    Args:
        learning_rate: Peak learning rate (default: 3e-4, Karpathy constant)
        min_lr: Minimum learning rate (default: 1/10 of lr as per Chinchilla)
        warmup_iters: Number of warmup iterations
        max_iters: Maximum number of iterations
    """

    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 1000
    max_iters: int = 100_0000

    def __call__(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters

        if iteration > self.max_iters:
            return self.min_lr

        decay_ratio = (iteration - self.warmup_iters) / (
            self.max_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


if __name__ == "__main__":
    scheduler = CosineScheduler(
        learning_rate=0.1, min_lr=0.001, warmup_iters=5, max_iters=100
    )
    print(scheduler)
