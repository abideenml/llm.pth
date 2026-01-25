"""
WinoGrande Evaluation Benchmark.

Paper: WinoGrande: An Adversarial Winograd Schema Challenge at Scale
https://arxiv.org/abs/1907.10641

Tests commonsense reasoning through pronoun resolution tasks.
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


WINOGRANDE_URLS = {
    "train_xs": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/train_xs.jsonl",
    "train_s": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/train_s.jsonl",
    "train_m": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/train_m.jsonl",
    "train_l": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/train_l.jsonl",
    "train_xl": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/train_xl.jsonl",
    "dev": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/dev.jsonl",
    "test": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1/test.jsonl",
}


class WinograndeEvaluator(MultipleChoiceEvaluator):
    """
    WinoGrande benchmark evaluator.

    Tests commonsense reasoning through fill-in-the-blank pronoun resolution.

    Example usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        evaluator = WinograndeEvaluator(model, tokenizer)
        result = evaluator.evaluate("dev")
        print(result)
    """

    @property
    def benchmark_name(self) -> str:
        return "winogrande"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device = None,
        batch_size: int = 1,
        cache_dir: str = None,
    ):
        super().__init__(model, tokenizer, device, batch_size, cache_dir)
        self._encode = tokenizer.encode

    def _download(self, split: str) -> str:
        """Download WinoGrande data if not present."""
        url = WINOGRANDE_URLS[split]
        filename = os.path.join(self.cache_dir, f"winogrande_{split}.jsonl")
        if not os.path.exists(filename):
            print(f"Downloading WinoGrande {split} split...")
            download_file(url, filename)
        return filename

    def load_data(self, split: str = "dev") -> Iterator[dict]:
        """
        Load WinoGrande examples.

        Args:
            split: One of 'train_xs', 'train_s', 'train_m', 'train_l', 'train_xl', 'dev', 'test'

        Yields:
            Example dictionaries with sentence, options, and answer
        """
        filename = self._download(split)
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                # WinoGrande uses "_" as placeholder
                sentence = data["sentence"]
                option1 = data["option1"]
                option2 = data["option2"]

                # Answer is "1" or "2"
                answer = data.get("answer", "0")
                if answer == "0":
                    # Test set doesn't have labels
                    answer_idx = -1
                else:
                    answer_idx = int(answer) - 1

                yield {
                    "sentence": sentence,
                    "option1": option1,
                    "option2": option2,
                    "answer": answer_idx,
                    "id": data.get("qID", ""),
                }

    def render_example(self, example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Render a WinoGrande example into tokens and mask.

        WinoGrande uses sentences with "_" as placeholder.
        We evaluate the likelihood of the complete sentence with each option.
        """
        sentence = example["sentence"]
        options = [example["option1"], example["option2"]]

        tok_rows = []
        mask_rows = []

        for option in options:
            # Replace placeholder with option
            filled_sentence = sentence.replace("_", option)
            tokens = self._encode(filled_sentence)
            tok_rows.append(tokens)
            # For WinoGrande, we evaluate the full sentence
            mask_rows.append([1] * len(tokens))

        # Pad to same length
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((2, max_len), dtype=torch.long)
        mask = torch.zeros((2, max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, example["answer"]

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "dev",
        max_examples: int = None,
        verbose: bool = True,
    ) -> EvalResult:
        """
        Evaluate on WinoGrande benchmark.

        Args:
            split: Data split
            max_examples: Limit number of examples
            verbose: Show progress

        Returns:
            EvalResult with accuracy metrics
        """
        from tqdm import tqdm

        self.model.eval()
        self.model.to(self.device)

        num_correct = 0
        num_correct_norm = 0
        num_total = 0

        examples = list(self.load_data(split))
        if max_examples:
            examples = examples[:max_examples]

        if verbose:
            examples = tqdm(examples, desc="WinoGrande")

        for example in examples:
            if example["answer"] == -1:
                # Skip test examples without labels
                continue

            tokens, mask, label = self.render_example(example)
            pred, pred_norm, _ = self.predict(tokens, mask)

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


def evaluate_winogrande(
    model_name: str = "gpt2",
    device: str = "cuda",
) -> EvalResult:
    """
    Evaluate a HuggingFace model on WinoGrande.

    Args:
        model_name: HuggingFace model name
        device: Device to use

    Returns:
        EvalResult with accuracy metrics
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluator = WinograndeEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
    )

    return evaluator.evaluate(split="dev", verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on WinoGrande")
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

    result = evaluate_winogrande(args.model, args.device)
    print(f"\nResults: {result}")
