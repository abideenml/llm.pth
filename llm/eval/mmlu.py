"""
MMLU (Massive Multitask Language Understanding) Evaluation Benchmark.

Paper: Measuring Massive Multitask Language Understanding
https://arxiv.org/abs/2009.03300

Tests knowledge across 57 subjects spanning STEM, humanities, social sciences, etc.
"""

import os
import json
import torch
import csv
from typing import Iterator, Any
from collections import defaultdict

from .base import (
    MultipleChoiceEvaluator,
    EvalResult,
    download_file,
)


# MMLU subjects organized by category
MMLU_SUBJECTS = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "social_sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "other": [
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}

ALL_SUBJECTS = [subj for subjects in MMLU_SUBJECTS.values() for subj in subjects]

MMLU_BASE_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


class MMLUEvaluator(MultipleChoiceEvaluator):
    """
    MMLU benchmark evaluator.

    Evaluates models on 57 subjects across STEM, humanities, social sciences, and other.

    Example usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        evaluator = MMLUEvaluator(model, tokenizer)
        result = evaluator.evaluate("test")
        print(result)
    """

    CHOICES = ["A", "B", "C", "D"]

    @property
    def benchmark_name(self) -> str:
        return "mmlu"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device = None,
        batch_size: int = 1,
        cache_dir: str = None,
        n_few_shot: int = 5,
    ):
        super().__init__(model, tokenizer, device, batch_size, cache_dir)
        self.n_few_shot = n_few_shot
        self._encode = tokenizer.encode

    def _get_data_path(self, split: str, subject: str) -> str:
        """Get path to MMLU CSV file."""
        return os.path.join(self.cache_dir, "data", split, f"{subject}_{split}.csv")

    def _ensure_data(self) -> None:
        """Ensure MMLU data is downloaded and extracted."""
        data_dir = os.path.join(self.cache_dir, "data")
        if os.path.exists(data_dir):
            return

        print("MMLU data not found. Please download from:")
        print("  https://people.eecs.berkeley.edu/~hendrycks/data.tar")
        print(f"And extract to: {self.cache_dir}/data/")
        print("\nAlternatively, use HuggingFace datasets:")
        print("  from datasets import load_dataset")
        print("  dataset = load_dataset('cais/mmlu', 'all')")
        raise FileNotFoundError(f"MMLU data not found at {data_dir}")

    def _load_subject_data(self, split: str, subject: str) -> list[dict]:
        """Load examples for a single subject."""
        filepath = self._get_data_path(split, subject)
        examples = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    examples.append(
                        {
                            "question": row[0],
                            "choices": row[1:5],
                            "answer": row[5] if len(row) > 5 else row[4],
                            "subject": subject,
                        }
                    )
        return examples

    def _format_prompt(
        self,
        question: str,
        choices: list[str],
        few_shot_examples: list[dict] = None,
        subject: str = None,
    ) -> str:
        """Format a question with optional few-shot examples."""
        prompt = ""

        # Add subject header
        if subject:
            subject_formatted = subject.replace("_", " ").title()
            prompt += f"The following are multiple choice questions about {subject_formatted}.\n\n"

        # Add few-shot examples
        if few_shot_examples:
            for ex in few_shot_examples:
                prompt += f"Question: {ex['question']}\n"
                for i, choice in enumerate(ex["choices"]):
                    prompt += f"{self.CHOICES[i]}. {choice}\n"
                prompt += f"Answer: {ex['answer']}\n\n"

        # Add the actual question
        prompt += f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{self.CHOICES[i]}. {choice}\n"
        prompt += "Answer:"

        return prompt

    def load_data(
        self, split: str = "test", subjects: list[str] = None
    ) -> Iterator[dict]:
        """
        Load MMLU examples.

        Args:
            split: One of 'dev', 'val', 'test'
            subjects: List of subjects to load (default: all)

        Yields:
            Example dictionaries
        """
        self._ensure_data()
        subjects = subjects or ALL_SUBJECTS

        for subject in subjects:
            try:
                examples = self._load_subject_data(split, subject)
                for ex in examples:
                    yield ex
            except FileNotFoundError:
                print(f"Warning: Could not find data for subject {subject}")
                continue

    def render_example(self, example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Render an MMLU example into tokens and mask.

        For MMLU, we evaluate the probability of each answer choice (A, B, C, D)
        given the prompt.
        """
        # Load few-shot examples from dev set
        few_shot = []
        if self.n_few_shot > 0:
            try:
                dev_examples = self._load_subject_data("dev", example["subject"])
                few_shot = dev_examples[: self.n_few_shot]
            except FileNotFoundError:
                pass

        prompt = self._format_prompt(
            example["question"],
            example["choices"],
            few_shot,
            example["subject"],
        )

        prompt_tokens = self._encode(prompt)

        # Create tokens and mask for each choice
        tok_rows = []
        mask_rows = []

        for choice in self.CHOICES:
            choice_tokens = self._encode(" " + choice)
            tok_rows.append(prompt_tokens + choice_tokens)
            mask_rows.append([0] * len(prompt_tokens) + [1] * len(choice_tokens))

        # Pad to same length
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        # Convert answer letter to index
        label = self.CHOICES.index(example["answer"])

        return tokens, mask, label

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "test",
        subjects: list[str] = None,
        max_examples: int = None,
        verbose: bool = True,
    ) -> EvalResult:
        """
        Evaluate on MMLU benchmark.

        Args:
            split: Data split ('dev', 'val', 'test')
            subjects: List of subjects to evaluate (default: all)
            max_examples: Limit number of examples per subject
            verbose: Show progress

        Returns:
            EvalResult with accuracy metrics and per-category breakdown
        """
        from tqdm import tqdm

        self.model.eval()
        self.model.to(self.device)

        subjects = subjects or ALL_SUBJECTS
        per_subject = defaultdict(lambda: {"correct": 0, "total": 0})
        per_category = defaultdict(lambda: {"correct": 0, "total": 0})

        num_correct = 0
        num_total = 0

        for subject in tqdm(subjects, desc="MMLU Subjects", disable=not verbose):
            category = None
            for cat, subjs in MMLU_SUBJECTS.items():
                if subject in subjs:
                    category = cat
                    break

            try:
                examples = self._load_subject_data(split, subject)
            except FileNotFoundError:
                continue

            if max_examples:
                examples = examples[:max_examples]

            for example in examples:
                tokens, mask, label = self.render_example(example)
                _, pred_norm, _ = self.predict(tokens, mask)

                correct = int(pred_norm == label)
                num_correct += correct
                num_total += 1
                per_subject[subject]["correct"] += correct
                per_subject[subject]["total"] += 1
                if category:
                    per_category[category]["correct"] += correct
                    per_category[category]["total"] += 1

        accuracy = num_correct / num_total if num_total > 0 else 0

        # Compute per-category accuracy
        category_acc = {}
        for cat, stats in per_category.items():
            if stats["total"] > 0:
                category_acc[cat] = stats["correct"] / stats["total"]

        return EvalResult(
            benchmark=self.benchmark_name,
            accuracy=accuracy,
            accuracy_norm=accuracy,
            num_correct=num_correct,
            num_correct_norm=num_correct,
            num_total=num_total,
            per_category=category_acc,
            metadata={"per_subject": dict(per_subject)},
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on MMLU")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt2", help="HuggingFace model name"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "-n", "--max-examples", type=int, default=None, help="Max examples per subject"
    )
    parser.add_argument(
        "--n-shot", type=int, default=5, help="Number of few-shot examples"
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    evaluator = MMLUEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(args.device),
        n_few_shot=args.n_shot,
    )

    result = evaluator.evaluate(split="test", max_examples=args.max_examples)
    print(f"\nResults: {result}")
    print("\nPer-category accuracy:")
    for cat, acc in result.per_category.items():
        print(f"  {cat}: {acc:.4f}")
