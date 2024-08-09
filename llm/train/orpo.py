import torch
from typing import Tuple
import torch.nn.functional as F
import time
from llm.utils.scheduler import CosineScheduler
from llm.utils.dataset import auto_accelerator
from rich import print
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.functional import log_softmax, sigmoid
from llm.models.phi3 import Phi, PhiConfig, model_summary
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.pytorch.loggers import WandbLogger

### CONFIGS
wandb_project_name = "Phi-ORPO-Fabric"
wandb_run_name = "run01"

learning_rate: float = 5e-4
min_learning_rate: float = 0.0
max_seq_length: int = 200
warmup_iters: int = 100
max_iters: int = 1010
batch_size: int = 32
micro_batch: int = 4
eval_iters: int = 100
save_ckpt_iters: int = 1000
alpha: float = 0.05
seq_len: int = 128
min: int = 0
max: int = batch_size
plugins = None
dataset_name = "argilla/dpo-mix-7k"  
tokenizer_name = "microsoft/phi-2"
model_name = "microsoft/phi-2"
device = auto_accelerator()  # auto chose device (cuda, mps)
resume_traning = False
# precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
precision = "16-true"


def compute_custom_loss(loss_fct, logits, labels):
    logits = logits.contiguous()
    if labels is not None:
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.transpose(2, 1), shift_labels).mean(dim=-1)
        return loss
    return None

def compute_logps(logits, chosen_inputs, chosen_attention_mask):
    mask = chosen_attention_mask[:, :-1]
    per_token_logps = log_softmax(logits[:, :-1, :], dim=-1)
    masked_inputs = mask * chosen_inputs[:, 1:]
    indexed_logps = per_token_logps.gather(2, masked_inputs.unsqueeze(2)).squeeze(2)
    return torch.mul(indexed_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1) / mask.sum(dim=1)

def calculate_nll_loss(model, inputs, labels, pad):
    labels[labels == pad] = -100
    outputs = model(input_ids=inputs['input_ids'], labels=labels, output_hidden_states=True)
    return outputs.loss

def calculate_metrics(pos_prob, neg_prob):
    log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob)))
    sig_ratio = sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    return torch.mean(ratio)

def compute_final_loss(pos_loss, alpha, ratio):
    return torch.mean(pos_loss - alpha * ratio).to(dtype=torch.bfloat16)
    

def compute_loss(fabric, model, inputs, alpha, pad, label_smoother, disable_prompt_loss):
    if label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    
    neg_labels = inputs['prompt_disprefered_ids'].clone()
    pos_labels = inputs['prompt_prefered_ids'].clone()

    if disable_prompt_loss:
        mask = inputs['attention_mask'] * inputs['positive_attention_mask']
        pos_labels = pos_labels * (~mask).logical_not()
        pos_labels[pos_labels == 0] = pad
    
    neg_labels[neg_labels == pad] = -100
    pos_labels[pos_labels == pad] = -100

    outputs_neg = model(input_ids=inputs['prompt_disprefered_ids'], attention_mask=inputs['prompt_disprefered_mask'], labels=neg_labels, output_hidden_states=True)
    outputs_pos = model(input_ids=inputs['prompt_prefered_ids'], attention_mask=inputs['prompt_prefered_mask'], labels=pos_labels, output_hidden_states=True)

    pos_loss = calculate_nll_loss(model, inputs, pos_labels, pad)
    pos_prob = compute_logps(outputs_pos.logits, inputs['prompt_prefered_ids'], inputs['prompt_prefered_mask'])
    neg_prob = compute_logps(outputs_neg.logits, inputs['prompt_disprefered_ids'], inputs['prompt_disprefered_mask'])

    ratio = calculate_metrics(pos_prob, neg_prob)
    final_loss = compute_final_loss(pos_loss, alpha, ratio)

    fabric.log({
        'Positive Geometric Mean': torch.mean(pos_prob).item(),
        'Negative Geometric Mean': torch.mean(neg_prob).item(),
        'Log Odds Ratio': torch.mean(ratio).item(),
    })

    return final_loss

def collate_fn(batch, tokenizer, max_length, device):
    prompts = ['Instruct: ' + item[0]['content'] + '\n' for item in batch['chosen']]
    chosen_responses = ['Output: ' + item[1]['content'] for item in batch['chosen']]
    rejected_responses = ['Output: ' + item[1]['content'] for item in batch['rejected']]

    prompt_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    prefered_ids = tokenizer.batch_encode_plus(chosen_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    disprefered_ids = tokenizer.batch_encode_plus(rejected_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)

    prompt_prefered_ids = torch.cat([prompt_ids, prefered_ids], dim=-1)
    prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)

    prompt_prefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(prefered_ids)], dim=-1)
    prompt_disprefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(disprefered_ids)], dim=-1)

    return {'prompt_prefered_ids': prompt_prefered_ids,
            'prompt_disprefered_ids': prompt_disprefered_ids,
            'prompt_prefered_mask': prompt_prefered_mask,
            'prompt_disprefered_mask': prompt_disprefered_mask}


def train(fabric, model, optimizer, dataset, tokenizer, alpha, epochs=1):
    model.train()
    start_time: float = time.perf_counter()
    idx: int = 0
    for epoch in range(epochs):
        min = 0
        max = 0
        for _ in range(int(len(dataset['train'])/batch_size)):
            max+=batch_size
            batch = collate_fn(
                dataset['train'][min:max],
                tokenizer=tokenizer,
                max_length=max_seq_length,
                device="cpu",    
            )
            optimizer.zero_grad()

            loss = compute_loss(fabric, model, batch, alpha, tokenizer.eos_token, label_smoother=False, disable_prompt_loss=False)

            loss.backward()
            optimizer.step()
            min+=max

        curr_time: float = time.perf_counter()
        elapsed_time: float = curr_time - start_time
        print(
            f"iter: {idx} | loss: {loss:.4f} | time: {elapsed_time:.4f}s"
        )

        if idx % save_ckpt_iters == 0:
            state = {"model": model, "optimizer": optimizer, "idx": idx, "lr": 1}
            fabric.save("./ckpt/model.pt", state)

            


def main():
    hyper_params = {
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
        "alpha": alpha,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "dataset_name": dataset_name,
        "resume_traning": resume_traning,
        "save_ckpt_iters": save_ckpt_iters,
    }
    # fabric init
    logger = WandbLogger(project=wandb_project_name, resume=resume_traning)
    dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
    plugins = BitsandbytesPrecision("nf4", dtype)
    fabric = L.Fabric(loggers=[logger],precision=precision, plugins=plugins)
    fabric.logger.log_hyperparams(hyper_params)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = "argilla/dpo-mix-7k"
    dataset = load_dataset(dataset)
    model: Phi = Phi.from_pretrained("microsoft/phi-2").to(device).eval().to(torch.float16)

    model: L.LightningModule = fabric.setup(model)
    model = torch.compile(model)

    print(model)
    print(model_summary(model))

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )
    optimizer_cls = bnb.optim.PagedAdamW
    optimizer = optimizer_cls(model.parameters(),  lr=get_lr(0))
    optimzer = fabric.setup_optimizers(optimizer)
    # Lets GO!!
    train(
        fabric,
        model,
        optimzer,
        dataset,
        tokenizer,
        alpha,
    )


if __name__ == "__main__":
    main()
