from typing import List, Dict, Optional, Callable, Any
import torch

def format_dataset(dataset_partition: dict) -> List[dict]:
    formatted_ds = []

    for entry in dataset_partition:
        convo = entry["conversations"]
        try:
            formatted_ds.append({"instruction": convo[0], "input": "", "output": convo[1]})
        except:
            pass

    return formatted_ds

def sft_dataset(
    data: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int = -1,
    mask_prompt: bool = True,
    ignore_index: int = -100,
    transform: Optional[Callable[[Any], Any]] = None
) -> List[Dict[str, torch.Tensor]]:
    processed_data = []
    
    for example in data:
        if transform is not None:
            example = transform(example)
        
        prompt = "Instruction: " + example["instruction"]
        prompt_and_response = prompt + example["output"]
        tokenizer.pad_token = tokenizer.eos_token
        
        encoded_prompt = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
        encoded_prompt_and_response = tokenizer.encode(prompt_and_response, max_length=max_seq_length, truncation=True)
        
        # Convert to tensors
        input_ids = torch.tensor(encoded_prompt_and_response, dtype=torch.long)
        labels = torch.tensor(encoded_prompt_and_response, dtype=torch.long).clone()  # Create a copy for labels
        
        if mask_prompt:
            # Mask the prompt section in labels
            labels[:len(encoded_prompt)] = ignore_index
        
        processed_data.append({
            "input_ids": input_ids,
            "labels": labels
        })
    
    return processed_data

def get_sft_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100):
    def sft_collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batched = {}
        for key in ("input_ids", "labels"):
            pad_value = pad_id if key == "input_ids" else ignore_index
            batched[key] = torch.nn.utils.rnn.pad_sequence(
                [sample[key] for sample in samples], batch_first=True, padding_value=pad_value
            )

            if max_seq_length > 0:
                batched[key] = batched[key][:, :max_seq_length]

        return batched

    return sft_collate_fn

