import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

import config
import tokenizer

# print(config.DIM)
# print(tokenizer.enc.encode("Hello, world!"))

model = torch.load("Llama3.2-1B-Instruct/consolidated.00.pth", map_location=torch.device('cpu'))


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


    



def apply_RoPE(x, freqs_cos, freqs_sin, n_heads):

    # x shape: (seq_len, n_heads, head_dim)
    x_even = x[:, :, ::2]
    x_odd = x[:, :, 1::2]
    # Expand freqs to match n_heads dimension - reshape for broadcasting func later
    freqs_cos_reshaped = freqs_cos.unsqueeze(1).expand(-1, n_heads, -1)  # (seq_len, n_heads, head_dim//2)
    freqs_sin_reshaped = freqs_sin.unsqueeze(1).expand(-1, n_heads, -1)  # (seq_len, n_heads, head_dim//2)
    x_even_rot = x_even * freqs_cos_reshaped - x_odd * freqs_sin_reshaped
    x_odd_rot = x_even * freqs_sin_reshaped + x_odd * freqs_cos_reshaped
    x_rot = torch.zeros_like(x)
    x_rot[:, :,::2] = x_even_rot
    x_rot[:, :, 1::2] = x_odd_rot
   
    return x_rot


# for gqa
def repeat_kv():
    pass

# Find query, key, value matrices (Q = wq * input...)
# reshape all matrices to (batch, n_heads, seq_len, head_dim)
# appply rope to Q,K
# multiply q,kT divide by sqrt d
# apply masking to this
# apply softmax to this
# multiply the whole thing by V
class MHA(nn.Module):
    def __init__(self, wq, wk, wv, wo, seq_len, n_heads=config.N_HEADS, n_kv_heads=config.N_KV_HEADS):
        super().__init__()
        self.wq = nn.Parameter(wq)
        self.wk = nn.Parameter(wk)
        self.wv = nn.Parameter(wv)
        self.wo = nn.Parameter(wo)
        self.seq_len = seq_len # input prompt length
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = config.DIM // self.n_heads # 32 heads * 64 dim = 2048 dim


    # x is input embeddings 
    def forward(self, x):
        
        # QKV per token matrix: [seq_len, 2048]
        x = x.to(self.wq.dtype)
        Q = torch.matmul(x, self.wq.T)
        K = torch.matmul(x, self.wk.T)
        V = torch.matmul(x, self.wv.T)
        # print("Q", Q)
        # print("Q.shape", Q.shape)
        # print("K", K)
        # print("K.shape", K.shape)
        # print("V", V)
        # print("V.shape", V.shape)
        

        # reshape them to (batch, n_heads, seq_len, head_dim)
        Q = Q.view(self.seq_len, self.n_heads, self.head_dim) # seq_len, 32, 64
        K = K.view(self.seq_len, self.n_kv_heads, self.head_dim) # seq_len, 8, 64
        V = V.view(self.seq_len, self.n_kv_heads, self.head_dim) # seq_len, 8, 64

        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, self.seq_len)
        # print("Q before RoPE", Q.shape)
        # print("K before RoPE", K.shape)
        
        Q = apply_RoPE(Q, freqs_cos, freqs_sin, self.n_heads)
        K = apply_RoPE(K, freqs_cos, freqs_sin, self.n_kv_heads)
        # print("Q after RoPE", Q.shape)
        # print("K after RoPE", K.shape)

        K_repeated = torch.repeat_interleave(K, self.n_heads//self.n_kv_heads, dim=1)
        V_repeated = torch.repeat_interleave(V, self.n_heads//self.n_kv_heads, dim=1)
        print("K_repeated", K_repeated.shape)
        print("V_repeated", V_repeated.shape)

        # swap seq_len and n_heads dimensions
        Q = Q.transpose(0, 1)
        print("Q transposed", Q.shape)
        K_repeated = K_repeated.transpose(0, 1)
        V_repeated = V_repeated.transpose(0, 1)
        print("K_repeated transposed", K_repeated.shape)

        attn_scores = Q @ K_repeated.transpose(1,2) / math.sqrt(self.head_dim)
        print("attn_scores", attn_scores.shape) # output: (n_heads, seq_len, seq_len)

        mask = torch.full((self.seq_len, self.seq_len), float('-inf'))
        mask = mask.triu(diagonal=1)
        print("mask", mask)

        attn_scores = attn_scores + mask
        print("attn_scores with mask", attn_scores)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(Q)
        print("attn_probs", attn_probs.shape)

        output = attn_probs @ V_repeated
        print("output", output.shape)

        output = output.transpose(0, 1).contiguous().view(self.seq_len, -1)
        print("output", output.shape)

        return torch.matmul(output, self.wo.T)
        


        

        


    


 


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




# call mha and test forward pass
# pass in weights from model
MHA = MHA(model["layers.0.attention.wq.weight"], model["layers.0.attention.wk.weight"], model["layers.0.attention.wv.weight"], model["layers.0.attention.wo.weight"], 25)
MHA(x)
print("MHA(x)", MHA(x))
print("MHA(x).shape", MHA(x).shape)