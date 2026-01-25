"""
HellaSwag Evaluation Benchmark.

HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/abs/1905.07830

Tests commonsense reasoning by asking models to complete sentences
with the most plausible continuation from 4 choices.
"""

import os
import json
import torch
from typing import Iterator, Any

from .base import (
    MultipleChoiceEvaluator,
    EvalResult,
    download_file,
)


HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


class HellaSwagEvaluator(MultipleChoiceEvaluator):
    """
    HellaSwag benchmark evaluator.

    Example usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        evaluator = HellaSwagEvaluator(model, tokenizer)
        result = evaluator.evaluate("val")
        print(result)
    """

    @property
    def benchmark_name(self) -> str:
        return "hellaswag"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        device: torch.device = None,
        batch_size: int = 1,
        cache_dir: str = None,
    ):
        super().__init__(model, tokenizer, device, batch_size, cache_dir)

        # Set up tokenizer
        if tokenizer is None:
            # Default to tiktoken GPT-2 encoding for backward compatibility
            import tiktoken

            self._tiktoken_enc = tiktoken.get_encoding("gpt2")
            self._encode = self._tiktoken_enc.encode
        else:
            self._tiktoken_enc = None
            self._encode = tokenizer.encode

    def _download(self, split: str) -> str:
        """Download HellaSwag data if not present."""
        url = HELLASWAG_URLS[split]
        filename = os.path.join(self.cache_dir, f"hellaswag_{split}.jsonl")
        if not os.path.exists(filename):
            print(f"Downloading HellaSwag {split} split...")
            download_file(url, filename)
        return filename

    def load_data(self, split: str = "val") -> Iterator[dict]:
        """
        Load HellaSwag examples.

        Args:
            split: One of 'train', 'val', 'test'

        Yields:
            Example dictionaries with 'ctx', 'endings', 'label' keys
        """
        filename = self._download(split)
        with open(filename, "r") as f:
            for line in f:
                yield json.loads(line)

    def render_example(self, example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Render a HellaSwag example into tokens and mask.

        Args:
            example: Dict with 'ctx', 'endings', 'label'

        Returns:
            tokens: [4, max_len] tensor of token ids
            mask: [4, max_len] tensor with 1s in completion region
            label: correct answer index (0-3)
        """
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        ctx_tokens = self._encode(ctx)

        tok_rows = []
        mask_rows = []

        for ending in endings:
            # Prepend space for proper tokenization (GPT-2 style)
            end_tokens = self._encode(" " + ending)
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        # Pad to same length
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "val",
        max_examples: int = None,
        verbose: bool = True,
        print_examples: int = 0,
    ) -> EvalResult:
        """
        Evaluate on HellaSwag benchmark.

        Args:
            split: Data split ('train', 'val', 'test')
            max_examples: Limit number of examples (for debugging)
            verbose: Show progress bar
            print_examples: Number of examples to print for debugging

        Returns:
            EvalResult with accuracy metrics
        """
        from tqdm import tqdm

        self.model.eval()
        self.model.to(self.device)

        num_correct = 0
        num_correct_norm = 0
        num_total = 0

        examples_iter = self.load_data(split)
        if verbose:
            # Count total for progress bar
            total = sum(1 for _ in self.load_data(split))
            if max_examples:
                total = min(total, max_examples)
            examples_iter = tqdm(
                self.load_data(split), total=total, desc="HellaSwag"
            )

        for i, example in enumerate(examples_iter):
            if max_examples is not None and i >= max_examples:
                break

            tokens, mask, label = self.render_example(example)
            pred, pred_norm, losses = self.predict(tokens, mask)

            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)

            # Print debug examples
            if i < print_examples:
                print("---")
                print(f"Context: {example['ctx']}")
                print("Endings:")
                for j, end in enumerate(example["endings"]):
                    marker = "✓" if j == label else " "
                    pred_marker = "→" if j == pred_norm else " "
                    print(f"  {marker}{pred_marker} {j}: (loss: {losses[j]:.4f}) {end}")
                print(f"Predicted: {pred_norm}, Actual: {label}")

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


# Convenience functions for backward compatibility
def iterate_examples(split: str = "val") -> Iterator[dict]:
    """Iterate over HellaSwag examples (legacy function)."""
    evaluator = HellaSwagEvaluator(model=None)
    yield from evaluator.load_data(split)


def render_example(example: dict) -> tuple[dict, torch.Tensor, torch.Tensor, int]:
    """Render a HellaSwag example (legacy function)."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": enc.encode(ctx),
        "ending_tokens": [],
    }

    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(data["ctx_tokens"] + end_tokens)
        mask_rows.append([0] * len(data["ctx_tokens"]) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def get_most_likely_row(
    tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> int:
    """Get the most likely completion (legacy function)."""
    from .base import compute_completion_loss

    avg_loss = compute_completion_loss(logits, tokens, mask, reduction="mean")
    return avg_loss.argmin().item()


def evaluate(model_type: str = "gpt2", device: str = "cuda") -> EvalResult:
    """
    Evaluate a HuggingFace model on HellaSwag (legacy function).

    Args:
        model_type: HuggingFace model name
        device: Device to use

    Returns:
        EvalResult with accuracy metrics
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_type}...")
    model = AutoModelForCausalLM.from_pretrained(model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    evaluator = HellaSwagEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
    )

    return evaluator.evaluate(split="val", verbose=True, print_examples=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on HellaSwag")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt2", help="HuggingFace model name"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "-n", "--max-examples", type=int, default=None, help="Max examples to evaluate"
    )
    args = parser.parse_args()

    result = evaluate(args.model, args.device)
    print(f"\nResults: {result}")
