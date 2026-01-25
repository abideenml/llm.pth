"""
Benchmark Runner - Evaluate models on multiple benchmarks.

Provides a unified interface to run all available benchmarks
and aggregate results.
"""

import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass, field

from .base import EvalResult
from .hellaswag import HellaSwagEvaluator
from .mmlu import MMLUEvaluator
from .arc import ARCEvaluator
from .winogrande import WinograndeEvaluator
from .truthfulqa import TruthfulQAEvaluator


@dataclass
class BenchmarkResults:
    """Container for results from multiple benchmarks."""

    results: dict[str, EvalResult] = field(default_factory=dict)
    model_name: str = ""

    def add(self, result: EvalResult) -> None:
        """Add a benchmark result."""
        self.results[result.benchmark] = result

    def __str__(self) -> str:
        lines = [f"Benchmark Results for {self.model_name}:", "=" * 50]
        for name, result in self.results.items():
            lines.append(f"  {name}: {result.accuracy:.4f}")
        lines.append("=" * 50)

        # Compute average
        if self.results:
            avg = sum(r.accuracy for r in self.results.values()) / len(self.results)
            lines.append(f"  Average: {avg:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "results": {
                name: {
                    "accuracy": r.accuracy,
                    "accuracy_norm": r.accuracy_norm,
                    "num_correct": r.num_correct,
                    "num_total": r.num_total,
                }
                for name, r in self.results.items()
            },
        }


class BenchmarkRunner:
    """
    Run multiple benchmarks on a model.

    Example usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        runner = BenchmarkRunner(model, tokenizer)
        results = runner.run_all()
        print(results)
    """

    AVAILABLE_BENCHMARKS = [
        "hellaswag",
        "mmlu",
        "arc_challenge",
        "arc_easy",
        "winogrande",
        "truthfulqa_mc1",
        "truthfulqa_mc2",
    ]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device = None,
        model_name: str = "unknown",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._auto_device()
        self.model_name = model_name

    def _auto_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def run_benchmark(
        self,
        benchmark: str,
        max_examples: int = None,
        verbose: bool = True,
        **kwargs,
    ) -> EvalResult:
        """
        Run a single benchmark.

        Args:
            benchmark: Benchmark name (e.g., 'hellaswag', 'mmlu', 'arc_challenge')
            max_examples: Limit examples for faster testing
            verbose: Show progress
            **kwargs: Additional arguments for specific benchmarks

        Returns:
            EvalResult with accuracy metrics
        """
        if benchmark == "hellaswag":
            evaluator = HellaSwagEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            return evaluator.evaluate(max_examples=max_examples, verbose=verbose)

        elif benchmark == "mmlu":
            evaluator = MMLUEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                n_few_shot=kwargs.get("n_few_shot", 5),
            )
            return evaluator.evaluate(max_examples=max_examples, verbose=verbose)

        elif benchmark == "arc_challenge":
            evaluator = ARCEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                difficulty="challenge",
            )
            return evaluator.evaluate(max_examples=max_examples, verbose=verbose)

        elif benchmark == "arc_easy":
            evaluator = ARCEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                difficulty="easy",
            )
            return evaluator.evaluate(max_examples=max_examples, verbose=verbose)

        elif benchmark == "winogrande":
            evaluator = WinograndeEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            return evaluator.evaluate(max_examples=max_examples, verbose=verbose)

        elif benchmark == "truthfulqa_mc1":
            evaluator = TruthfulQAEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            return evaluator.evaluate(
                task="mc1", max_examples=max_examples, verbose=verbose
            )

        elif benchmark == "truthfulqa_mc2":
            evaluator = TruthfulQAEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            return evaluator.evaluate(
                task="mc2", max_examples=max_examples, verbose=verbose
            )

        else:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Available: {self.AVAILABLE_BENCHMARKS}"
            )

    def run_all(
        self,
        benchmarks: list[str] = None,
        max_examples: int = None,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Run multiple benchmarks.

        Args:
            benchmarks: List of benchmark names (default: common subset)
            max_examples: Limit examples per benchmark
            verbose: Show progress

        Returns:
            BenchmarkResults with all results
        """
        if benchmarks is None:
            # Default to common benchmarks (skip MMLU which requires separate download)
            benchmarks = ["hellaswag", "arc_challenge", "winogrande", "truthfulqa_mc1"]

        results = BenchmarkResults(model_name=self.model_name)

        for benchmark in benchmarks:
            print(f"\n{'='*50}")
            print(f"Running {benchmark}...")
            print("=" * 50)

            try:
                result = self.run_benchmark(
                    benchmark, max_examples=max_examples, verbose=verbose
                )
                results.add(result)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error running {benchmark}: {e}")
                continue

        return results


def evaluate_model(
    model_name: str = "gpt2",
    device: str = "cuda",
    benchmarks: list[str] = None,
    max_examples: int = None,
) -> BenchmarkResults:
    """
    Evaluate a HuggingFace model on multiple benchmarks.

    Args:
        model_name: HuggingFace model name
        device: Device to use
        benchmarks: List of benchmarks to run
        max_examples: Limit examples per benchmark

    Returns:
        BenchmarkResults with all results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    runner = BenchmarkRunner(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        model_name=model_name,
    )

    return runner.run_all(benchmarks=benchmarks, max_examples=max_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM Benchmarks")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt2", help="HuggingFace model name"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to run (default: hellaswag, arc_challenge, winogrande, truthfulqa_mc1)",
    )
    parser.add_argument(
        "-n", "--max-examples", type=int, default=None, help="Max examples per benchmark"
    )
    args = parser.parse_args()

    results = evaluate_model(
        model_name=args.model,
        device=args.device,
        benchmarks=args.benchmarks,
        max_examples=args.max_examples,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(results)
