import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass


class MoD(nn.Module):
    """
    Paper: https://arxiv.org/abs/2404.02258
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.seq_len = cfg.seq_len
        self.capacity_factor = cfg.capacity_factor
        self.dim = cfg.d_model

        self.transformer_decoder_block = Block(cfg)
        self.router = nn.Linear(self.dim, 1, bias=False)
        self.aux_router = nn.Sequential(
                            nn.Linear(self.dim,self.dim//2),
                            nn.SiLU(),
                            nn.Linear(self.dim//2,1),
                            )

    def forward(
        self, x: Tensor, mask, freqs_cis, mode="train", auxiliary_loss=False, *args, **kwargs
    ):
        batch_size, seq_len, dim = x.shape

        # if mode == "inference":
        #     return self.inference(x, *args, **kwargs)
        # S = seq_len, C = capacity  , C = int(seq_length * capacity_factor)
        #  page 6 above eq 1 | ( C<S ) | here top_k = beta
        top_k = int(seq_len * self.capacity_factor)  # may be i should use math.ceil

        # eq1 page 6
        # scaler weights for each token
        router_logits = self.router(x)  # (x) batch,seq_len,dim -> r batch,seq_len,1

        #  𝑟𝑙> 𝑃𝛽 (R)  ... eqution 1
        token_weights, token_index = torch.topk(router_logits, top_k, dim=1, sorted=False)

        # now we have idx, we can copy this weights to another tensor and pass them to attn+mlp

        # since its auto regressive model we need to keep casual nature of it
        # that why we need sort the tokens by idx before we pass it to attn
        selected_tokens, index = torch.sort(token_index, dim=1)

        # select idx for copying for original tensor
        indices_expanded = selected_tokens.expand(-1, -1, dim)

        # This are fillted topk tokens with capactiy C
        filtered_x = torch.gather(input=x, dim=1, index=indices_expanded)  # -> batch, capacity, dim

        x_out, _ = self.transformer_decoder_block(filtered_x, mask, freqs_cis)

        # selecting router wight by idx ( in sorted maner)
        # ~~NOTE~~
        # paper is using softmax instead of sigmoid, softmax is non casual which is not good
        # I tried replacing it with sigmoid and too my surprise it "works"
        # I suspect author did not use it because they are using jax, jax does funny things
        # ...
        token_weights = F.softmax(token_weights, dim=1) # <<<== use this if you want execact paper replication
        # token_weights = F.sigmoid(token_weights)
        r_weights = torch.gather(token_weights, dim=1, index=index)

        # muliply by router weights, this add router in gradient stream
        xw_out = r_weights * x_out

        # batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, top_k)
        # # # https://discuss.pytorch.org/t/when-inplace-operation-are-allowed-and-when-not/169583/2
        # out = x.clone()
        # # add back to resuidal strean
        # out[batch_indices, selected_tokens.squeeze(-1),: ] += xw_out
        # # ^ this can be done with torch.scatter_add
        out = torch.scatter_add(input=x, dim=1, index=indices_expanded, src=xw_out)

        if auxiliary_loss:
            aux_loss = self.aux_loss(x, router_logits, selected_tokens)
            return out, aux_loss
        return out, _

    def aux_loss(self, x: Tensor, router_logits: Tensor, selected_tokens: Tensor):
        batch_size, seq_len, dim = x.shape
        # Page 7, Section 3.5 sampling
        router_targets = torch.zeros_like(router_logits).view(
            -1
        )  # i think torch to scatter will work here TODO
        router_targets[selected_tokens.view(-1)] = 1.0
        aux_router_logits = self.aux_router(x.detach().view(batch_size * seq_len, -1))
        # aux_router_logits = F.sigmoid(aux_router_logits)  # keep output in range [0,1)
        # RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
        # so binary_cross_entropy_with_logits == sigmoid + bce_loss
        return F.binary_cross_entropy_with_logits(aux_router_logits.view(-1), router_targets)
