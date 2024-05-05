import sys
import time
import math
from datetime import datetime

from typing import Any, Dict
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from llm.models.phi import Phi, PhiConfig, model_summary
from llm.utils.scheduler import CosineScheduler
from llm.utils.dataset import *
from llm.utils.sftdata import *


from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from rich import print, traceback

import lightning as L
from lightning.pytorch.loggers import WandbLogger

traceback.install()


### CONFIGS
wandb_project_name = "Ohara-LLAMA-Fabric"
wandb_run_name = "run01"

learning_rate: float = 5e-4
min_learning_rate: float = 0.0

warmup_iters: int = 100
max_iters: int = 1010
batch_size: int = 32
micro_batch: int = 4
eval_iters: int = 100
save_ckpt_iters: int = 1000

d_model: int = 1024 // 16
seq_len: int = 256
num_layers: int = 4
num_heads: int = 4
multiple_of: int = 4

assert d_model % num_heads == 0

compile_model = not bool(sys.platform == "darwin")

### Dataset and Tokenizer
dataset_name = "roneneldan/TinyStories"  # run pretokeinze first
tokenizer_name = "microsoft/phi-2"

### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)

# for restarting training from last checkout
resume_traning = False


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    dataloader: DataLoader,
    max_iters: int,
    ignore_index: int = -1,
    device=None,
) -> int:
    fabric.barrier()
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(max_iters, device=device)
        for idx, (data, target) in enumerate(dataloader):
            if idx >= max_iters:
                break
            logits: torch.Tensor = model(data)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=ignore_index,
            )
            losses[idx] = loss.item()
        val_loss = losses.mean()

    model.train()
    fabric.barrier()
    return val_loss


def train(
    fabric: L.Fabric,
    model: nn.Module,
    optimizer: optim.Optimizer,
    micro_batch: int,
    train_dataloader,
    val_dataloader,
    eval_iters: int,
    save_ckpt_iters: int,
    get_lr,
    ignore_index=-1,
):
    fabric.launch()
    ignore_index = ignore_index if ignore_index else -1
    # sanity test
    validate(fabric, model, val_dataloader, 5, device=device)
    # cyclining loder so you can runit indefinitely
    train_dataloader = BetterCycle(iter(train_dataloader))
    val_dataloader = BetterCycle(iter(val_dataloader))
    (data, target) = next(val_dataloader)
    tokerns_per_iter = int(math.prod(data.shape) * micro_batch)

    micro_batch_loss: float = 0
    idx: int = 0
    while True:
        start_time: float = time.perf_counter()
        micro_batch_loss = 0
        if idx >= max_iters:
            break
        idx += 1

        lr = get_lr(idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # ...
        for _ in range(micro_batch):
            (data, target) = next(train_dataloader)
            with fabric.no_backward_sync(model, enabled=micro_batch == 1):
                logits: torch.Tensor = model(data)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=ignore_index,
                )
                loss = loss / micro_batch
                micro_batch_loss += loss.item()
                fabric.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        curr_time: float = time.perf_counter()
        elapsed_time: float = curr_time - start_time
        print(
            f"iter: {idx} | loss: {micro_batch_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s"
        )

        if idx % eval_iters == 0:
            val_loss = validate(fabric, model, val_dataloader, 100, device=device)
            try:
                fabric.log_dict(
                    {
                        "traning_loss": micro_batch_loss,
                        "test_loss": val_loss,
                        "iter": idx,
                        "tokens": idx * tokerns_per_iter,
                        "lr": lr,
                        "time": elapsed_time,
                    },
                    step=idx,
                )
            except Exception as e:
                print(f"Error logging: {e}")

        if idx % save_ckpt_iters == 0:
            state = {"model": model, "optimizer": optimizer, "idx": idx, "lr": lr}
            fabric.save("./ckpt/model.pt", state)


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


# Code inspired by https://github.com/catid/dora/blob/main/dora.py
class LinearWithDoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        new_weight = self.m * V
        return F.linear(x, new_weight, self.linear.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






def main():
    hyper_params = {
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "d_model": d_model,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_layers,
        "multiple_of": multiple_of,
        "compile_model": compile_model,
        "tokenizer_name": tokenizer_name,
        "dataset_name": dataset_name,
        "resume_traning": resume_traning,
        "save_ckpt_iters": save_ckpt_iters,
    }
    # fabric init
    # logger = WandbLogger(project=wandb_project_name, resume=resume_traning)
    # fabric = L.Fabric(loggers=[logger])
    fabric = L.Fabric()
    # fabric.logger.log_hyperparams(hyper_params)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    config = PhiConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_layers,
        multiple_of=multiple_of,
    )

    train_ds = PreTokenizedDataset(
        dataset_name="GAIR/lima",
        tokenizer=tokenizer,
        split="train",
        max_length=100,
    ) 
    test_ds = PreTokenizedDataset(
        dataset_name="GAIR/lima",
        tokenizer=tokenizer,
        split="validation",
        max_length=100,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)
    train_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, test_dataloader)
    #  model_name = "cognitivecomputations/dolphin-2_6-phi-2"
    #  model: Phi = Phi.from_pretrained(model_name).to(device).eval().to(torch.float16)
    model = Phi(config)
    for param in model.parameters():
        param.requires_grad = False
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_query = True
    lora_key = False
    lora_value = True
    lora_projection = False
    lora_mlp = False
    lora_head = False

    layers = []

    assign_lora = partial(LinearWithDoRA, rank=lora_r, alpha=lora_alpha)

    for layer in model.layers:
        if lora_query:
            layer.mixer.q_proj = assign_lora(layer.mixer.q_proj)
        if lora_key:
            layer.mixer.k_proj = assign_lora(layer.mixer.k_proj)
        if lora_value:
            layer.mixer.v_proj = assign_lora(layer.mixer.v_proj)
        if lora_projection:
            layer.attention.out_lin = assign_lora(layer.attention.out_lin)
        if lora_mlp:
            layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
            layer.ffn.lin2 = assign_lora(layer.ffn.lin2)

    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    
    print("Total number of trainable parameters:", count_parameters(model))

    model: L.LightningModule = fabric.setup(model)

    if compile_model:
        model = torch.compile(model)

    print(model)
    print(model_summary(model))

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )

    # inputs = torch.tensor(tokenizer.encode("The")).unsqueeze(0).clone().detach()
    optimzer = optim.AdamW(model.parameters(), lr=get_lr(0))
    optimzer = fabric.setup_optimizers(optimzer)
    # Lets GO!!
    train(
        fabric,
        model,
        optimzer,
        micro_batch=micro_batch,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        eval_iters=eval_iters,
        save_ckpt_iters=save_ckpt_iters,
        get_lr=get_lr,
        ignore_index=tokenizer.pad_token_id,
    )


if __name__ == "__main__":
    main()
