"""
TruthfulQA Evaluation Benchmark.

Paper: TruthfulQA: Measuring How Models Mimic Human Falsehoods
https://arxiv.org/abs/2109.07958

Tests whether models generate truthful and informative answers,
particularly for questions where humans might give false answers
due to misconceptions or false beliefs.
"""

import os
import json
import torch
import csv
from typing import Iterator, Any

from .base import (
    MultipleChoiceEvaluator,
    EvalResult,
    download_file,
)


TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"


class TruthfulQAEvaluator(MultipleChoiceEvaluator):
    """
    TruthfulQA benchmark evaluator (MC1 and MC2 tasks).

    MC1: Single correct answer among choices
    MC2: Multiple correct/incorrect answers (scored by probability mass)

    Example usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        evaluator = TruthfulQAEvaluator(model, tokenizer)
        result = evaluator.evaluate()
        print(result)
    """

    @property
    def benchmark_name(self) -> str:
        return "truthfulqa"

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

    def _download(self) -> str:
        """Download TruthfulQA data if not present."""
        filename = os.path.join(self.cache_dir, "TruthfulQA.csv")
        if not os.path.exists(filename):
            print("Downloading TruthfulQA dataset...")
            download_file(TRUTHFULQA_URL, filename)
        return filename

    def load_data(self, split: str = "val") -> Iterator[dict]:
        """
        Load TruthfulQA examples.

        TruthfulQA only has one split (817 questions).

        Yields:
            Example dictionaries with question, correct answers, and incorrect answers
        """
        filename = self._download()

        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"]
                category = row["Category"]

                # MC1: Best answer is first correct, others are incorrect
                best_answer = row.get("Best Answer", "")
                correct_answers = row.get("Correct Answers", "").split("; ")
                incorrect_answers = row.get("Incorrect Answers", "").split("; ")

                # Filter empty strings
                correct_answers = [a.strip() for a in correct_answers if a.strip()]
                incorrect_answers = [a.strip() for a in incorrect_answers if a.strip()]

                if not correct_answers or not incorrect_answers:
                    continue

                yield {
                    "question": question,
                    "category": category,
                    "best_answer": best_answer,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                }

    def render_example(self, example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Render a TruthfulQA example for MC1 task.

        MC1: Pick the single best answer from choices.
        We use the best answer as correct and first incorrect answer as wrong.
        """
        question = example["question"]
        correct = example["best_answer"] or example["correct_answers"][0]
        incorrect = example["incorrect_answers"][0]

        choices = [correct, incorrect]
        # Correct answer is at index 0
        label = 0

        prompt = f"Q: {question}\nA:"
        prompt_tokens = self._encode(prompt)

        tok_rows = []
        mask_rows = []

        for choice in choices:
            choice_tokens = self._encode(" " + choice)
            tok_rows.append(prompt_tokens + choice_tokens)
            mask_rows.append([0] * len(prompt_tokens) + [1] * len(choice_tokens))

        # Pad to same length
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((len(choices), max_len), dtype=torch.long)
        mask = torch.zeros((len(choices), max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label

    def render_example_mc2(
        self, example: dict
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Render a TruthfulQA example for MC2 task.

        MC2: Score based on probability mass on correct vs incorrect answers.
        Returns all choices with labels indicating which are correct.
        """
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]

        all_choices = correct_answers + incorrect_answers
        # Labels: 1 for correct, 0 for incorrect
        labels = [1] * len(correct_answers) + [0] * len(incorrect_answers)

        prompt = f"Q: {question}\nA:"
        prompt_tokens = self._encode(prompt)

        tok_rows = []
        mask_rows = []

        for choice in all_choices:
            choice_tokens = self._encode(" " + choice)
            tok_rows.append(prompt_tokens + choice_tokens)
            mask_rows.append([0] * len(prompt_tokens) + [1] * len(choice_tokens))

        # Pad to same length
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((len(all_choices), max_len), dtype=torch.long)
        mask = torch.zeros((len(all_choices), max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, labels

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "val",
        max_examples: int = None,
        verbose: bool = True,
        task: str = "mc1",
    ) -> EvalResult:
        """
        Evaluate on TruthfulQA benchmark.

        Args:
            split: Ignored (TruthfulQA has single split)
            max_examples: Limit number of examples
            verbose: Show progress
            task: 'mc1' for single best answer, 'mc2' for probability mass scoring

        Returns:
            EvalResult with accuracy metrics
        """
        from tqdm import tqdm
        from .base import compute_completion_loss, get_model_output

        self.model.eval()
        self.model.to(self.device)

        examples = list(self.load_data(split))
        if max_examples:
            examples = examples[:max_examples]

        if verbose:
            examples = tqdm(examples, desc=f"TruthfulQA ({task.upper()})")

        if task == "mc1":
            # MC1: Pick single best answer
            num_correct = 0
            num_total = 0

            for example in examples:
                tokens, mask, label = self.render_example(example)
                _, pred_norm, _ = self.predict(tokens, mask)

                num_total += 1
                num_correct += int(pred_norm == label)

            accuracy = num_correct / num_total if num_total > 0 else 0

            return EvalResult(
                benchmark=f"{self.benchmark_name}_mc1",
                accuracy=accuracy,
                num_correct=num_correct,
                num_total=num_total,
            )

        else:  # mc2
            # MC2: Probability mass on correct answers
            total_score = 0.0
            num_total = 0

            for example in examples:
                tokens, mask, labels = self.render_example_mc2(example)
                tokens = tokens.to(self.device)
                mask = mask.to(self.device)

                logits = get_model_output(self.model, tokens, self.device)
                losses = compute_completion_loss(logits, tokens, mask, reduction="mean")

                # Convert to probabilities (lower loss = higher probability)
                probs = torch.softmax(-losses, dim=0)

                # Score = sum of probabilities on correct answers
                labels_tensor = torch.tensor(labels, device=self.device)
                correct_prob = (probs * labels_tensor.float()).sum().item()

                total_score += correct_prob
                num_total += 1

            accuracy = total_score / num_total if num_total > 0 else 0

            return EvalResult(
                benchmark=f"{self.benchmark_name}_mc2",
                accuracy=accuracy,
                num_total=num_total,
                metadata={"total_score": total_score},
            )


def evaluate_truthfulqa(
    model_name: str = "gpt2",
    device: str = "cuda",
    task: str = "mc1",
) -> EvalResult:
    """
    Evaluate a HuggingFace model on TruthfulQA.

    Args:
        model_name: HuggingFace model name
        device: Device to use
        task: 'mc1' or 'mc2'

    Returns:
        EvalResult with accuracy metrics
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluator = TruthfulQAEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
    )

    return evaluator.evaluate(task=task, verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on TruthfulQA")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt2", help="HuggingFace model name"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mc1",
        choices=["mc1", "mc2"],
        help="Evaluation task",
    )
    parser.add_argument(
        "-n", "--max-examples", type=int, default=None, help="Max examples to evaluate"
    )
    args = parser.parse_args()

    result = evaluate_truthfulqa(args.model, args.device, args.task)
    print(f"\nResults: {result}")
