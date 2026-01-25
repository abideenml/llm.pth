"""
Supervised Fine-Tuning (SFT) data utilities.

Includes:
- format_dataset: Format conversation data for training
- sft_dataset: Process and tokenize SFT data
- get_sft_collate_fn: Collate function for batching SFT data
"""

from typing import List, Dict, Optional, Callable, Any
import torch


def format_dataset(dataset_partition: dict) -> List[dict]:
    """
    Format conversation dataset into instruction-output format.

    Args:
        dataset_partition: Dataset with 'conversations' field

    Returns:
        List of dicts with 'instruction', 'input', and 'output' keys
    """
    formatted_ds = []

    for entry in dataset_partition:
        convo = entry["conversations"]
        try:
            formatted_ds.append(
                {"instruction": convo[0], "input": "", "output": convo[1]}
            )
        except:
            pass

    return formatted_ds


def sft_dataset(
    data: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int = -1,
    mask_prompt: bool = True,
    ignore_index: int = -100,
    transform: Optional[Callable[[Any], Any]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Process data for supervised fine-tuning.

    Args:
        data: List of dicts with 'instruction' and 'output' keys
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length (-1 for no limit)
        mask_prompt: If True, mask prompt tokens in labels
        ignore_index: Index to use for masked tokens
        transform: Optional transform to apply to each example

    Returns:
        List of dicts with 'input_ids' and 'labels' tensors
    """
    processed_data = []

    for example in data:
        if transform is not None:
            example = transform(example)

        prompt = "Instruction: " + example["instruction"]
        prompt_and_response = prompt + example["output"]
        tokenizer.pad_token = tokenizer.eos_token

        encoded_prompt = tokenizer.encode(
            prompt, max_length=max_seq_length, truncation=True
        )
        encoded_prompt_and_response = tokenizer.encode(
            prompt_and_response, max_length=max_seq_length, truncation=True
        )

        # Convert to tensors
        input_ids = torch.tensor(encoded_prompt_and_response, dtype=torch.long)
        labels = torch.tensor(encoded_prompt_and_response, dtype=torch.long).clone()

        if mask_prompt:
            # Mask the prompt section in labels
            labels[: len(encoded_prompt)] = ignore_index

        processed_data.append({"input_ids": input_ids, "labels": labels})

    return processed_data


def get_sft_collate_fn(
    max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
):
    """
    Create a collate function for SFT data batching.

    Args:
        max_seq_length: Maximum sequence length (-1 for no limit)
        pad_id: Padding token ID for input_ids
        ignore_index: Padding value for labels

    Returns:
        Collate function for DataLoader
    """

    def sft_collate_fn(
        samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        batched = {}
        for key in ("input_ids", "labels"):
            pad_value = pad_id if key == "input_ids" else ignore_index
            batched[key] = torch.nn.utils.rnn.pad_sequence(
                [sample[key] for sample in samples],
                batch_first=True,
                padding_value=pad_value,
            )

            if max_seq_length > 0:
                batched[key] = batched[key][:, :max_seq_length]

        return batched

    return sft_collate_fn
