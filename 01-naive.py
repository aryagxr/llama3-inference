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


class RMSNorm(nn.Module):
    def __init__(self, dim, weight, eps=config.NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weight)

    def _norm(self, x):
        pass

    def forward(self, x):
        pass




# head_dim = 64 
def precompute_freqs_cis(head_dim, end, theta=config.ROPE_THETA):
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end) #  0,1,2...seq_len-1
    freqs_matrix = torch.outer(t, freq) # theta = pos * freq -> shape: (seq_len, head_dim//2)
    freqs_cos = torch.cos(freqs_matrix)
    freqs_sin = torch.sin(freqs_matrix)
    return freqs_cos, freqs_sin



def RoPE(x, cos, sin):
    pass


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



