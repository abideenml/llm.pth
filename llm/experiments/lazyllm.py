import torch
import torch.nn as nn
import torch.nn.functional as F

# Paper implementation of "LazyLLM: DYNAMIC TOKEN PRUNING FOR EFFICIENT LONG CONTEXT LLM INFERENCE" (https://arxiv.org/pdf/2407.14057)

class LazyLLM(nn.Module):
    def __init__(self, transformer_model, top_k=0.7, aux_cache_size=1000):
        super(LazyLLM, self).__init__()
        self.transformer = transformer_model
        self.top_k = top_k
        self.aux_cache = {}  # Store pruned tokens' hidden states
        self.aux_cache_size = aux_cache_size

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.transformer.model.embed_tokens(input_ids)
        batch_size, seq_len, d_model = hidden_states.size()

        # Progressive pruning
        for layer in self.transformer.model.layers:
            # Compute attention and importance scores
            q_proj = layer.self_attn.q_proj(hidden_states)
            k_proj = layer.self_attn.k_proj(hidden_states)
            v_proj = layer.self_attn.v_proj(hidden_states)
            k = k_proj.view(batch_size, seq_len, 32 , -1)  # shape = (B, seq_len, num_heads, head_dim)
            q = q_proj.view(batch_size, seq_len, 32, -1)
            v = v_proj.view(batch_size, seq_len, 32, -1)
            k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # Compute attention weights and outputs
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (k.size(-1) ** 0.5)
            if attention_mask is not None:
                attention_scores += attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            attn_output = torch.matmul(attention_probs, v)
            output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            attn_output = layer.self_attn.o_proj(output)
            
            # Layer normalization and MLP
            hidden_states = hidden_states + attn_output
            hidden_states = layer.input_layernorm(hidden_states)
            # Feed-forward network (MLP)
            weight = layer.mlp.gate_proj.weight
            weight_t = weight.t()
            gate_output = torch.matmul(hidden_states, weight_t)
            weight_up = layer.mlp.up_proj.weight 
            weight_upt = weight_up.t()
            up_output = torch.matmul(hidden_states, weight_upt)
            inter = gate_output * up_output
            weight_down = layer.mlp.down_proj.weight 
            weight_downt = weight_down.t()
            down_output = torch.matmul(inter, weight_downt)

            hidden_states = hidden_states + down_output
            hidden_states = layer.post_attention_layernorm(hidden_states)
            
            # Calculate importance scores for pruning
            importance_scores = self.calculate_importance(attention_probs)
            
            # Prune tokens
            hidden_states, pruned_tokens = self.prune_tokens(hidden_states, importance_scores)

            # Store pruned tokens in auxiliary cache
            self.update_aux_cache(pruned_tokens)

        # Final transformer output
        output = self.transformer.lm_head(hidden_states)
        return output

    def calculate_importance(self, attention_probs):
        # Assuming attention_probs has shape (batch_size, num_heads, seq_len, seq_len)
        # Calculate the average attention score over all heads for the last token
        importance_scores = attention_probs.mean(dim=1)[:, :, -1]
        return importance_scores

    def prune_tokens(self, hidden_states, importance_scores):
        # Prune tokens based on the importance score threshold
        batch_size, seq_len, hidden_size = hidden_states.size()
        threshold = torch.kthvalue(importance_scores.view(-1), int(seq_len * self.top_k)).values
        mask = importance_scores >= threshold
        pruned_hidden_states = hidden_states * mask.unsqueeze(-1).float()
        pruned_tokens = hidden_states[~mask]
        return pruned_hidden_states, pruned_tokens

    def update_aux_cache(self, pruned_tokens):
        # Store pruned tokens in auxiliary cache
        for token in pruned_tokens:
            if len(self.aux_cache) >= self.aux_cache_size:
                # Remove the oldest entry if the cache is full
                self.aux_cache.pop(next(iter(self.aux_cache)))
            self.aux_cache[token] = token


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    sd = model.state_dict()
    for k, v in sd.items():
        print(k, v.shape)
    lazy_llm = LazyLLM(model)
    input = tokenizer.encode("Are you a language model")
    tokens = torch.tensor(input, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(4, 1)  # Repeat for batch size of 4

    # Specify the number of tokens to generate
    num_tokens_to_generate = 10  # Change this to generate more or fewer tokens

    # Initialize generator for reproducibility
    sample_rng = torch.Generator()
    sample_rng.manual_seed(42)

    # Loop to generate tokens
    for _ in range(num_tokens_to_generate):
        output_ids = lazy_llm(tokens)
        logits = output_ids[:, -1, :]  # Get the logits for the last token in each sequence
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # Sample from top-k probabilities
        xcol = torch.gather(topk_indices, -1, ix)  # Get the sampled token indices
        tokens = torch.cat((tokens, xcol), dim=1)  # Append the new tokens to the input sequence

    # Decode and print the generated sequences
    for i in range(4):
        tokenss = tokens[i, :].tolist()  # Get the entire generated sequence for each batch element
        decoded = tokenizer.decode(tokenss)
        print(f"Generated sequence {i + 1}: {decoded}")