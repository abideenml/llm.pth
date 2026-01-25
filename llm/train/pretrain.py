import inspect
import math
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from lightning import Fabric
from lightning.pytorch.loggers import WandbLogger
from rich import print, traceback
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.experiments.mhc import Deepseekv32, Deepseekv32Config

traceback.install()


# Constants
DEFAULT_DATA_ROOT = "./data/edu_fineweb10B"
DEFAULT_SEED = 1337
DEFAULT_TOTAL_BATCH_SIZE = 524288  # 2**19, ~0.5M tokens
DEFAULT_MICRO_BATCH = 8
DEFAULT_SEQ_LEN = 1024
DEFAULT_MAX_LR = 6e-4
DEFAULT_MIN_LR_RATIO = 0.1
DEFAULT_WARMUP_STEPS = 715
DEFAULT_EVAL_STEPS = 250
DEFAULT_MAX_STEPS = 19073  # ~1 epoch for 10B tokens with 0.5M token batch size
DEFAULT_SAVE_CKPT_STEPS = 5000
DEFAULT_LOG_DIR = "checkpoints"
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_LEARNING_RATE = 6e-4


class DataLoaderLite:
    """Lightweight data loader for sharded tokenized datasets."""
    
    def __init__(self, B, T, process_rank, num_processes, split, master_process=False, data_root=None):
        """
        Initialize the data loader.
        
        Args:
            B: Batch size
            T: Sequence length
            process_rank: Process rank for distributed training
            num_processes: Total number of processes
            split: Dataset split ('train' or 'val')
            master_process: Whether this is the master process
            data_root: Root directory containing data shards
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process
        assert split in {'train', 'val'}, f"split must be 'train' or 'val', got {split}"

        # Get the shard filenames
        if data_root is None:
            data_root = DEFAULT_DATA_ROOT
        self.data_root = data_root
        
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        
        assert len(shards) > 0, f"no shards found for split {split} in {data_root}"
        if self.master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        """Reset the data loader to the beginning."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Get the next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        
        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        
        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        
        return x, y


def configure_optimizers(model, weight_decay, learning_rate, device_type, master_process=False):
    """
    Configure optimizer with weight decay for 2D parameters only.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay value
        learning_rate: Learning rate
        device_type: Device type ('cuda' or 'cpu')
        master_process: Whether this is the master process
        
    Returns:
        Configured AdamW optimizer
    """
    # Start with all candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    if master_process:
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    
    if master_process:
        print(f"Using fused AdamW: {use_fused}")
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused
    )
    return optimizer


def load_tokens(filename):
    """Load tokenized data from a numpy file."""
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Calculate learning rate with cosine decay and warmup.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Maximum number of steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"decay_ratio should be in [0, 1], got {decay_ratio}"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def setup_distributed():
    """Setup distributed training if DDP is enabled."""
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        
        # Attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def evaluate_model(model, val_loader, device, device_type, ddp, master_process, eval_steps=20):
    """Evaluate the model on validation data."""
    model.eval()
    val_loader.reset()
    
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(eval_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / eval_steps
            val_loss_accum += loss.detach()
    
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    
    return val_loss_accum.item()


def generate_samples(model, enc, device, device_type, ddp_rank, max_length=32, num_return_sequences=4):
    """Generate text samples from the model."""
    model.eval()
    prompt = "Hi, I am an LLM. How can I help"
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen)  # (B, T, vocab_size)
            
            # Take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Get the probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            # Select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            
            # Gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            
            # Append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    
    # Print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"Rank {ddp_rank} sample {i}: {decoded}")


def main():
    """Main training loop."""
    # Setup distributed training
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_distributed()
    
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    # Set random seeds
    torch.manual_seed(DEFAULT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DEFAULT_SEED)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Training hyperparameters
    total_batch_size = DEFAULT_TOTAL_BATCH_SIZE
    micro_batch = DEFAULT_MICRO_BATCH
    seq_len = DEFAULT_SEQ_LEN
    
    assert total_batch_size % (micro_batch * seq_len * ddp_world_size) == 0, \
        f"total_batch_size ({total_batch_size}) must be divisible by micro_batch * seq_len * ddp_world_size ({micro_batch * seq_len * ddp_world_size})"
    
    grad_accum_steps = total_batch_size // (micro_batch * seq_len * ddp_world_size)
    
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")
    
    # Create data loaders
    train_loader = DataLoaderLite(
        B=micro_batch,
        T=seq_len,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        master_process=master_process
    )
    val_loader = DataLoaderLite(
        B=micro_batch,
        T=seq_len,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        master_process=master_process
    )
    
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    
    # Create model
    config = Deepseekv32Config()
    model = Deepseekv32(config)
    model.to(device)
    
    use_compile = True
    if use_compile:
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # Learning rate schedule parameters
    max_lr = DEFAULT_MAX_LR
    min_lr = max_lr * DEFAULT_MIN_LR_RATIO
    warmup_steps = DEFAULT_WARMUP_STEPS
    eval_steps = DEFAULT_EVAL_STEPS
    max_steps = DEFAULT_MAX_STEPS
    save_ckpt_steps = DEFAULT_SAVE_CKPT_STEPS
    log_dir = DEFAULT_LOG_DIR
    
    # Create optimizer
    optimizer = configure_optimizers(
        raw_model,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        learning_rate=DEFAULT_LEARNING_RATE,
        device_type=device_type,
        master_process=master_process
    )
    
    # Hyperparameters for logging
    hyper_params = {
        "max_learning_rate": max_lr,
        "min_learning_rate": min_lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "batch_size": total_batch_size,
        "micro_batch": micro_batch,
        "seq_len": seq_len,
        "use_compile": use_compile,
        "device": device,
        "save_ckpt_steps": save_ckpt_steps,
    }
    
    # Initialize Fabric for logging
    logger = WandbLogger(project="pretrain", resume=False)
    fabric = Fabric(loggers=[logger])
    fabric.logger.log_hyperparams(hyper_params)
    
    # Setup model and optimizer with Fabric
    model = fabric.setup(model)
    optimizer = fabric.setup_optimizers(optimizer)
    
    fabric.launch()
    
    # Training loop
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        
        # Evaluate validation loss periodically
        if step % eval_steps == 0 or last_step:
            val_loss = evaluate_model(
                model, val_loader, device, device_type, ddp, master_process
            )
            
            if master_process:
                print(f"Validation loss: {val_loss:.4f}")
                try:
                    fabric.log_dict({"val": val_loss, "iter": step}, step=step)
                except Exception as e:
                    print(f"Error logging: {e}")
                
                # Save checkpoint
                if step > 0 and (step % save_ckpt_steps == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    state = {
                        'model': raw_model,
                        'config': raw_model.config,
                        "optimizer": optimizer,
                        'step': step,
                        'val_loss': val_loss
                    }
                    fabric.save(checkpoint_path, state)
        
        # Generate samples periodically (skip if using torch.compile)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            generate_samples(model, enc, device, device_type, ddp_rank)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # DDP gradient sync control
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()  # Wait for GPU to finish work
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            print(
                f"Step {step:5d} | Loss: {loss_accum.item():.6f} | "
                f"LR: {lr:.4e} | Norm: {norm:.4f} | "
                f"DT: {dt*1000:.2f}ms | Tok/sec: {tokens_per_sec:.2f}"
            )
            try:
                fabric.log_dict(
                    {
                        "train_loss": loss_accum.item(),
                        "iter": step,
                        "lr": lr,
                        "norm": norm,
                        "dt": dt,
                        "tok/sec": tokens_per_sec,
                        "tokens_processed": tokens_processed,
                    },
                    step=step,
                )
            except Exception as e:
                print(f"Error logging: {e}")
    
    # Cleanup
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()