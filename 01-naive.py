import torch 
import torch.nn as nn
import torch.nn.functional as F

import config
import tokenizer

# print(config.DIM)
# print(tokenizer.enc.encode("Hello, world!"))

# bpe tokenizer
# model config
# rms norm 
# rope
# mha
# ffn


# ffn_norm weights shape (2048)
class RMSNorm(nn.Module):
    def __init__(self, dim, weight, eps=config.NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weight)

    def _norm(self, x):
        return torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * x * self.weight



# head_dim = 64 
def precompute_freqs_cis(head_dim, end, theta=config.ROPE_THETA):
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end) #  0,1,2...seq_len-1
    freqs_matrix = torch.outer(t, freq) # theta = pos * freq -> shape: (seq_len, head_dim//2)
    freqs_cos = torch.cos(freqs_matrix)
    freqs_sin = torch.sin(freqs_matrix)
    return freqs_cos, freqs_sin



def apply_RoPE(x, freqs_cos, freqs_sin):
    x_even = x[:,::2]
    x_odd = x[:, 1::2]
    x_even_rot = x_even * freqs_cos - x_odd * freqs_sin
    x_odd_rot = x_even * freqs_sin + x_odd * freqs_cos
    x_rot = torch.zeros_like(x)
    x_rot[:,::2] = x_even_rot
    x_rot[:,1::2] = x_odd_rot
    return x_rot



class MHA(nn.Module):
    pass


class FeedForward(nn.Module):
    pass



class TransformerBlock(nn.Module):
    pass



class Transformer(nn.Module):
    pass









# need to chat template prompt
# prompt = "Are you a pirate?"
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|><br><br>Are you a pirate?<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br>"

encoded = tokenizer.enc.encode(prompt, allowed_special="all")
print(encoded)

token_ids = torch.tensor(encoded)
print(token_ids)

print(token_ids.shape)

# create embedding layer
emb_layer = nn.Embedding(config.VOCAB_SIZE, config.DIM)
# print(emb_layer)
print(emb_layer.weight.shape)

# input 
x = emb_layer(token_ids)
print("x", x)
print("x.shape", x.shape)



