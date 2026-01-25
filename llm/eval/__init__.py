"""
LLM Evaluation Benchmarks.

This module provides implementations of common LLM evaluation benchmarks:
- HellaSwag: Commonsense reasoning
- MMLU: Multitask language understanding
- ARC: Science reasoning (Challenge and Easy)
- WinoGrande: Pronoun resolution
- TruthfulQA: Truthfulness evaluation

Example usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm.eval import BenchmarkRunner

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    runner = BenchmarkRunner(model, tokenizer)
    results = runner.run_all()
    print(results)
"""

from .base import (
    BaseEvaluator,
    MultipleChoiceEvaluator,
    EvalResult,
    compute_completion_loss,
    download_file,
    get_cache_dir,
)

from .hellaswag import (
    HellaSwagEvaluator,
    iterate_examples,
    render_example,
    get_most_likely_row,
    evaluate as evaluate_hellaswag,
)

from .mmlu import (
    MMLUEvaluator,
    MMLU_SUBJECTS,
    ALL_SUBJECTS,
    evaluate as evaluate_mmlu,
)

from .arc import (
    ARCEvaluator,
    evaluate as evaluate_arc,
)

from .winogrande import (
    WinograndeEvaluator,
    evaluate as evaluate_winogrande,
)

from .truthfulqa import (
    TruthfulQAEvaluator,
    evaluate as evaluate_truthfulqa,
)

from .runner import (
    BenchmarkRunner,
    BenchmarkResults,
    evaluate_model,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    "MultipleChoiceEvaluator",
    "EvalResult",
    "compute_completion_loss",
    "download_file",
    "get_cache_dir",
    # HellaSwag
    "HellaSwagEvaluator",
    "iterate_examples",
    "render_example",
    "get_most_likely_row",
    "evaluate_hellaswag",
    # MMLU
    "MMLUEvaluator",
    "MMLU_SUBJECTS",
    "ALL_SUBJECTS",
    "evaluate_mmlu",
    # ARC
    "ARCEvaluator",
    "evaluate_arc",
    # WinoGrande
    "WinograndeEvaluator",
    "evaluate_winogrande",
    # TruthfulQA
    "TruthfulQAEvaluator",
    "evaluate_truthfulqa",
    # Runner
    "BenchmarkRunner",
    "BenchmarkResults",
    "evaluate_model",
]
