# ollama run MlChat
# ollama run MlCopilot
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
from llm.models.phi3 import Phi, PhiConfig, model_summary
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.pytorch.loggers import WandbLogger

### CONFIGS
wandb_project_name = "Phi-DPO-Fabric"
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
seq_len: int = 128
min: int = 0
max: int = batch_size

d_model: int = 1024 // 16
seq_len: int = 256
num_layers: int = 4
num_heads: int = 4
multiple_of: int = 4
plugins = None
### Dataset and Tokenizer
dataset_name = "argilla/dpo-mix-7k"  # run pretokeinze first
tokenizer_name = "microsoft/phi-2"
model_name = "cognitivecomputations/dolphin-2_6-phi-2"
### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)
precision = "16-true"
# for restarting training from last checkout
resume_traning = False
# precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".


class DPOTrainer:
    def __init__(self, beta:float=0.05, label_smoothing:float=0.1, loss_type:str='sigmoid'):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

    def dpo_loss_hf_implementation(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.
        """
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
    

def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):

    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

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

def train(fabric, model, ref_model, optimizer, dataset,tokenizer, epochs=1, beta=0.1):
    model.train()
    ref_model.eval()
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

            prompt_prefered_ids = batch['prompt_prefered_ids']
            prompt_disprefered_ids = batch['prompt_disprefered_ids']
            prompt_prefered_mask = batch['prompt_prefered_mask']
            prompt_disprefered_mask = batch['prompt_disprefered_mask']

            model_prefered_log_prob = get_log_prob(model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
            model_disprefered_log_prob = get_log_prob(model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

            ref_prefered_log_prob = get_log_prob(ref_model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
            ref_disprefered_log_prob = get_log_prob(ref_model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

            loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(model_prefered_log_prob, model_disprefered_log_prob,
                                          ref_prefered_log_prob, ref_disprefered_log_prob,
                                          beta=beta)

            loss.backward()
            optimizer.step()
            min+=max

        curr_time: float = time.perf_counter()
        elapsed_time: float = curr_time - start_time
        print(
            f"iter: {idx} | loss: {loss:.4f} | time: {elapsed_time:.4f}s"
        )

        if idx % eval_iters == 0:
            # val_loss = validate(fabric, model, val_dataloader, 100, device=device)
            try:
                fabric.log({'loss': loss.item(),
                       'prefered_relative_logprob': prefered_relative_logprob,
                       'disprefered_relative_logprob': disprefered_relative_logprob,
                       'reward_accuracy': reward_accuracies,
                       'reward_margin': reward_margins})
                
            except Exception as e:
                print(f"Error logging: {e}")

        if idx % save_ckpt_iters == 0:
            state = {"model": model, "optimizer": optimizer, "idx": idx, "lr": 1}
            fabric.save("./ckpt/model.pt", state)

            


def main():
    hyper_params = {
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
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
    ref_model: Phi = Phi.from_pretrained("microsoft/phi-2").to(device).eval().to(torch.float16)
    ref_model: L.LightningModule = fabric.setup(ref_model)
    model = torch.compile(model)
    ref_model = torch.compile(ref_model)

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
        ref_model,
        optimzer,
        dataset,
        tokenizer,
    )


if __name__ == "__main__":
    main()
