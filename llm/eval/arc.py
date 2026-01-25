"""
ARC (AI2 Reasoning Challenge) Evaluation Benchmark.

Paper: Think you have Solved Question Answering? Try ARC
https://arxiv.org/abs/1803.05457

Tests science reasoning with questions from standardized tests.
Two splits: ARC-Easy and ARC-Challenge.
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


ARC_URLS = {
    "challenge": {
        "train": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl",
        "dev": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl",
        "test": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl",
    },
    "easy": {
        "train": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl",
        "dev": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl",
        "test": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl",
    },
}


class ARCEvaluator(MultipleChoiceEvaluator):
    """
    ARC benchmark evaluator.

    Evaluates science reasoning with multiple choice questions.

    Example usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        evaluator = ARCEvaluator(model, tokenizer, difficulty="challenge")
        result = evaluator.evaluate("test")
        print(result)
    """

    @property
    def benchmark_name(self) -> str:
        return f"arc_{self.difficulty}"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device = None,
        batch_size: int = 1,
        cache_dir: str = None,
        difficulty: str = "challenge",
        n_few_shot: int = 0,
    ):
        """
        Initialize ARC evaluator.

        Args:
            difficulty: 'challenge' or 'easy'
            n_few_shot: Number of few-shot examples (0 for zero-shot)
        """
        self.difficulty = difficulty
        self.n_few_shot = n_few_shot
        super().__init__(model, tokenizer, device, batch_size, cache_dir)
        self._encode = tokenizer.encode

    def _download(self, split: str) -> str:
        """Download ARC data if not present."""
        url = ARC_URLS[self.difficulty][split]
        filename = os.path.join(
            self.cache_dir, f"arc_{self.difficulty}_{split}.jsonl"
        )
        if not os.path.exists(filename):
            print(f"Downloading ARC-{self.difficulty.title()} {split} split...")
            download_file(url, filename)
        return filename

    def load_data(self, split: str = "test") -> Iterator[dict]:
        """
        Load ARC examples.

        Args:
            split: One of 'train', 'dev', 'test'

        Yields:
            Example dictionaries with question, choices, and answer
        """
        filename = self._download(split)
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]

                # Extract choices and answer
                choices = []
                labels = []
                for choice in question["choices"]:
                    choices.append(choice["text"])
                    labels.append(choice["label"])

                # Find answer index
                answer_label = data["answerKey"]
                try:
                    answer_idx = labels.index(answer_label)
                except ValueError:
                    # Some answers use numbers instead of letters
                    try:
                        answer_idx = int(answer_label) - 1
                    except ValueError:
                        continue

                yield {
                    "question": question["stem"],
                    "choices": choices,
                    "labels": labels,
                    "answer": answer_idx,
                    "id": data.get("id", ""),
                }

    def _format_prompt(
        self,
        question: str,
        choices: list[str],
        labels: list[str],
        few_shot_examples: list[dict] = None,
    ) -> str:
        """Format a question with optional few-shot examples."""
        prompt = "Answer the following science question by selecting the correct option.\n\n"

        # Add few-shot examples
        if few_shot_examples:
            for ex in few_shot_examples:
                prompt += f"Question: {ex['question']}\n"
                for label, choice in zip(ex["labels"], ex["choices"]):
                    prompt += f"{label}. {choice}\n"
                correct_label = ex["labels"][ex["answer"]]
                prompt += f"Answer: {correct_label}\n\n"

        # Add the actual question
        prompt += f"Question: {question}\n"
        for label, choice in zip(labels, choices):
            prompt += f"{label}. {choice}\n"
        prompt += "Answer:"

        return prompt

    def render_example(self, example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Render an ARC example into tokens and mask.
        """
        # Load few-shot examples
        few_shot = []
        if self.n_few_shot > 0:
            train_examples = list(self.load_data("train"))
            few_shot = train_examples[: self.n_few_shot]

        prompt = self._format_prompt(
            example["question"],
            example["choices"],
            example["labels"],
            few_shot if few_shot else None,
        )

        prompt_tokens = self._encode(prompt)

        # Create tokens and mask for each choice label
        tok_rows = []
        mask_rows = []

        for label in example["labels"]:
            choice_tokens = self._encode(" " + label)
            tok_rows.append(prompt_tokens + choice_tokens)
            mask_rows.append([0] * len(prompt_tokens) + [1] * len(choice_tokens))

        # Pad to same length
        max_len = max(len(row) for row in tok_rows)
        num_choices = len(example["choices"])
        tokens = torch.zeros((num_choices, max_len), dtype=torch.long)
        mask = torch.zeros((num_choices, max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, example["answer"]

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "test",
        max_examples: int = None,
        verbose: bool = True,
    ) -> EvalResult:
        """
        Evaluate on ARC benchmark.

        Args:
            split: Data split ('train', 'dev', 'test')
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
            examples = tqdm(examples, desc=f"ARC-{self.difficulty.title()}")

        for example in examples:
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


def evaluate_arc(
    model_name: str = "gpt2",
    device: str = "cuda",
    difficulty: str = "challenge",
) -> EvalResult:
    """
    Evaluate a HuggingFace model on ARC.

    Args:
        model_name: HuggingFace model name
        device: Device to use
        difficulty: 'challenge' or 'easy'

    Returns:
        EvalResult with accuracy metrics
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluator = ARCEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        difficulty=difficulty,
    )

    return evaluator.evaluate(split="test", verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on ARC")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt2", help="HuggingFace model name"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="challenge",
        choices=["challenge", "easy"],
        help="ARC difficulty",
    )
    parser.add_argument(
        "-n", "--max-examples", type=int, default=None, help="Max examples to evaluate"
    )
    args = parser.parse_args()

    result = evaluate_arc(args.model, args.device, args.difficulty)
    print(f"\nResults: {result}")
