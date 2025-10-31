import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import time

import config
import tokenizer

# print(config.DIM)
# print(tokenizer.enc.encode("Hello, world!"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.load("Llama3.2-1B-Instruct/consolidated.00.pth", map_location=device)


prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|><br><br>If mountains could talk<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br>"


encoded = tokenizer.enc.encode(prompt, allowed_special="all")
# print(encoded)

token_ids = torch.tensor(encoded, device=device)
# print(token_ids)

print(token_ids.shape)
# seq_len = token_ids.shape[0]


# ffn_norm weights shape (2048)
class RMSNorm(nn.Module):
    def __init__(self, weight, eps=config.NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weight)

    def _norm(self, x):
        return torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * x * self.weight



# head_dim = 64 


def precompute_freqs_cis(head_dim, end, theta=config.ROPE_THETA, device=None):
    if device is None:
        device=torch.device("cpu")
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, device=device) #  0,1,2...seq_len-1
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
    x_rot = torch.zeros_like(x, device=device)
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
    def __init__(self, wq, wk, wv, wo, n_heads=config.N_HEADS, n_kv_heads=config.N_KV_HEADS):
        super().__init__()
        self.wq = nn.Parameter(wq)
        self.wk = nn.Parameter(wk)
        self.wv = nn.Parameter(wv)
        self.wo = nn.Parameter(wo)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = config.DIM // self.n_heads # 32 heads * 64 dim = 2048 dim
        
        # Initialize KV cache
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0


    # x is input embeddings - now only processes new tokens
    def forward(self, x, mask, freqs_cos, freqs_sin, start_idx=0):
        seq_len = x.shape[0]  # Should be 1 for single token generation
        
        # print(f"Processing {seq_len} new tokens starting at position {start_idx}")
        
        # Step 1: Compute Q, K, V for new tokens only
        x = x.to(self.wq.dtype)
        Q = torch.matmul(x, self.wq.T)  # [seq_len, 2048]
        K = torch.matmul(x, self.wk.T)  # [seq_len, 2048] 
        V = torch.matmul(x, self.wv.T)  # [seq_len, 2048]
        
        # print("Q.shape", Q.shape)
        # print("K.shape", K.shape) 
        # print("V.shape", V.shape)

        # Step 2: Reshape to head format
        Q = Q.view(seq_len, self.n_heads, self.head_dim)     # [seq_len, 32, 64]
        K = K.view(seq_len, self.n_kv_heads, self.head_dim)  # [seq_len, 8, 64]
        V = V.view(seq_len, self.n_kv_heads, self.head_dim)  # [seq_len, 8, 64]

        # Step 3: Apply RoPE for current position only
        current_seq_len = start_idx + seq_len
        # freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, current_seq_len)
        
        # Only apply RoPE to the new tokens at their current positions
        Q = apply_RoPE(Q, freqs_cos[start_idx:], freqs_sin[start_idx:], self.n_heads)
        K = apply_RoPE(K, freqs_cos[start_idx:], freqs_sin[start_idx:], self.n_kv_heads)
        
        # print("Q after RoPE", Q.shape)
        # print("K after RoPE", K.shape)

        # Step 4: Update KV cache
        if self.k_cache is None:
            # Initialize cache for first time
            max_cache_len = 1000  # Set reasonable max length
            self.k_cache = torch.zeros(max_cache_len, self.n_kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)
            self.v_cache = torch.zeros(max_cache_len, self.n_kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)
            self.cache_len = 0
        
        # Append new K and V to cache
        self.k_cache[start_idx:start_idx+seq_len] = K
        self.v_cache[start_idx:start_idx+seq_len] = V
        self.cache_len = current_seq_len
        
        # print(f"Cache updated: k_cache.shape = {self.k_cache[:self.cache_len].shape}")
        # print(f"Cache updated: v_cache.shape = {self.v_cache[:self.cache_len].shape}")

        # Step 5: Repeat for GQA (use cached values)
        K_cached = self.k_cache[:self.cache_len]  # [cache_len, 8, 64]
        V_cached = self.v_cache[:self.cache_len]  # [cache_len, 8, 64]

        
        
        K_repeated = torch.repeat_interleave(K_cached, self.n_heads//self.n_kv_heads, dim=1)  # [cache_len, 32, 64]
        V_repeated = torch.repeat_interleave(V_cached, self.n_heads//self.n_kv_heads, dim=1)  # [cache_len, 32, 64]
        
        # print("K_repeated", K_repeated.shape)
        # print("V_repeated", V_repeated.shape)

        # Step 6: Transpose for attention
        Q = Q.transpose(0, 1)        # [32, seq_len, 64]
        K_repeated = K_repeated.transpose(0, 1)  # [32, cache_len, 64]
        V_repeated = V_repeated.transpose(0, 1)  # [32, cache_len, 64]

        
        output = F.scaled_dot_product_attention(Q, K_repeated, V_repeated, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=1.0/math.sqrt(self.head_dim), enable_gqa=False)
        
        # print("Q transposed", Q.shape)
        # print("K_repeated transposed", K_repeated.shape)

        # Step 7: Compute attention with cache
        # attn_scores = Q @ K_repeated.transpose(1,2) / math.sqrt(self.head_dim)
        # print("attn_scores", attn_scores.shape)  # [32, seq_len, cache_len]

        # Step 8: Apply causal mask (only for new tokens)
        # if mask is not None:
            # attn_scores = attn_scores + mask
            # print("attn_scores with mask", attn_scores.shape)

        # attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(Q)
        # print("attn_probs", attn_probs.shape)

        # Step 9: Attention output
        # output = attn_probs @ V_repeated  # [32, seq_len, 64]
        # print("output", output.shape)

        output = output.transpose(0, 1).contiguous().view(seq_len, -1)  # [seq_len, 2048]
        # print("output", output.shape)
        

        return torch.matmul(output, self.wo.T)
        


        

class FeedForward(nn.Module):
    def __init__(self, w1, w3, w2):
        super().__init__()
        self.w1 = nn.Parameter(w1)
        self.w3 = nn.Parameter(w3)
        self.w2 = nn.Parameter(w2)


    
    def forward(self, x):
        return torch.matmul(
            (F.silu(torch.matmul(x, self.w1.T)) * torch.matmul(x, self.w3.T)),
            self.w2.T
        )



# add attn norm
# add ffn norm
class TransformerBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = MHA(
            model[f"layers.{layer_idx}.attention.wq.weight"], 
            model[f"layers.{layer_idx}.attention.wk.weight"], 
            model[f"layers.{layer_idx}.attention.wv.weight"], 
            model[f"layers.{layer_idx}.attention.wo.weight"]
        )
        self.ffn = FeedForward(
            model[f"layers.{layer_idx}.feed_forward.w1.weight"], 
            model[f"layers.{layer_idx}.feed_forward.w3.weight"], 
            model[f"layers.{layer_idx}.feed_forward.w2.weight"],
        )
        
        self.attn_norm = RMSNorm(model[f"layers.{layer_idx}.attention_norm.weight"])
        self.ffn_norm = RMSNorm(model[f"layers.{layer_idx}.ffn_norm.weight"])

    def forward(self, x, mask, freqs_cos, freqs_sin, start_idx=0):
        # change x to bfloat16
        x = x.to(torch.bfloat16)
        attn_out = x + self.attn(self.attn_norm(x), mask, freqs_cos, freqs_sin, start_idx)
        ffn_out = attn_out + self.ffn(self.ffn_norm(attn_out))
        return ffn_out



    



class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = config.N_LAYERS
        self.vocab_size = config.VOCAB_SIZE
        self.head_dim = config.DIM // config.N_HEADS # 32 heads * 64 dim = 2048 dim
        
        self.tok_emb = nn.Embedding.from_pretrained(
            model["tok_embeddings.weight"], 
            freeze=True
        )
        
        # Final normalization layer
        self.norm = RMSNorm(model["norm.weight"])
        
        # Output projection layer (hidden_dim -> vocab_size)
        self.output_weights = nn.Parameter(model["output.weight"])
        
        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_idx))

        

        
    def forward(self, tokens, start_idx=0):
        x = self.tok_emb(tokens)
        # print("input_emb", x.shape)

        # Create mask for causal attention
        seq_len = x.shape[0]
        current_len = start_idx + seq_len
        
        # For KV cache, we only need to mask future positions beyond current_len
        mask = torch.full((seq_len, current_len), float('-inf'), device=x.device)
        mask = mask.triu(diagonal=start_idx+1)  # Only mask positions after current position
        # print("mask", mask.shape)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, current_len, device=x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cos, freqs_sin, start_idx)
        
        x = self.norm(x)
        # print("after norm", x.shape)
        
        logits = torch.matmul(x, self.output_weights.T).float()
        # print("logits", logits.shape)
        
        return logits

        
        


# Performance measurement
class PerformanceMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.token_times = []
        self.total_tokens_generated = 0
        self.input_tokens = 0
    
    def start_generation(self, input_tokens):
        self.input_tokens = len(input_tokens)
        self.start_time = time.time()
    
    def record_first_token(self):
        self.first_token_time = time.time()
    
    def record_token(self):
        current_time = time.time()
        self.token_times.append(current_time)
        self.total_tokens_generated += 1
    
    def end_generation(self):
        self.end_time = time.time()
    
    def calculate_metrics(self):
        if not all([self.start_time, self.first_token_time, self.end_time]):
            return None
        
        # Time to First Token (TTFT)
        ttft = self.first_token_time - self.start_time
        
        # End-to-End Latency (E2EL)
        e2el = self.end_time - self.start_time
        
        # Token Generation Time (excluding TTFT)
        token_gen_time = e2el - ttft
        
        # Time per Output Token (TPOT)
        output_tokens = self.total_tokens_generated - self.input_tokens
        tpot = token_gen_time / max(output_tokens - 1, 1) if output_tokens > 1 else 0
        
        # Inter-Token Latency (ITL) - average time between consecutive tokens
        itl_times = []
        for i in range(1, len(self.token_times)):
            itl_times.append(self.token_times[i] - self.token_times[i-1])
        avg_itl = sum(itl_times) / len(itl_times) if itl_times else 0
        
        # Tokens per second
        tokens_per_second = output_tokens / token_gen_time if token_gen_time > 0 else 0
        
        # Throughput (total tokens / total time)
        throughput = self.total_tokens_generated / e2el if e2el > 0 else 0
        
        return {
            'TTFT': ttft,
            'E2EL': e2el,
            'Token_Generation_Time': token_gen_time,
            'TPOT': tpot,
            'Average_ITL': avg_itl,
            'Tokens_Per_Second': tokens_per_second,
            'Throughput': throughput,
            'Total_Tokens': self.total_tokens_generated,
            'Output_Tokens': output_tokens,
            'Input_Tokens': self.input_tokens
        }
    
    def print_metrics(self):
        metrics = self.calculate_metrics()
        if not metrics:
            print("No metrics available")
            return
        
        print("\n")
        # print("\n" + "="*60)
        print("[PERFORMANCE METRICS]")
        # print("="*60)
        print(f"Time to First Token (TTFT):     {metrics['TTFT']:.4f} seconds")
        print(f"End-to-End Latency (E2EL):      {metrics['E2EL']:.4f} seconds")
        print(f"Token Generation Time:          {metrics['Token_Generation_Time']:.4f} seconds")
        print(f"Time per Output Token (TPOT):   {metrics['TPOT']:.4f} seconds")
        print(f"Average Inter-Token Latency:    {metrics['Average_ITL']:.4f} seconds")
        print(f"Tokens per Second:              {metrics['Tokens_Per_Second']:.2f} tokens/sec")
        print(f"Throughput:                     {metrics['Throughput']:.2f} tokens/sec")
        print(f"Total Tokens Generated:          {metrics['Total_Tokens']}")
        print(f"Output Tokens:                  {metrics['Output_Tokens']}")
        print(f"Input Tokens:                   {metrics['Input_Tokens']}")
        print("\n")


Transformer = Transformer().to(device)





# Autoregressive generation loop with KV cache
def generate_with_kv_cache(transformer, initial_tokens, max_new_tokens=100):
    """
    Generate text autoregressively using KV cache
    Args:
        transformer: The trained transformer model
        initial_tokens: Starting token sequence
        max_new_tokens: Maximum number of new tokens to generate
    """
    # Initialize performance metrics
    metrics = PerformanceMetrics()
    metrics.start_generation(initial_tokens)
    
    # Step 1: Process initial prompt to populate KV cache
    logits = transformer(initial_tokens, start_idx=0)
    
    # Get prediction for next token
    next_token_logits = logits[-1, :]
    predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
    predicted_token_text = tokenizer.enc.decode([predicted_token_id])
    
    # Initialize generation
    generated_tokens = initial_tokens.clone()
    current_position = len(initial_tokens)
    
    first_token_recorded = False
    
    for step in range(max_new_tokens):
        # Create single token tensor for the predicted token
        new_token = torch.tensor([predicted_token_id], device=initial_tokens.device)
        
        # Record timing for first token
        if not first_token_recorded:
            metrics.record_first_token()
            first_token_recorded = True
        
        # Process new token with KV cache
        new_logits = transformer(new_token, start_idx=current_position)
        
        # Get prediction for next token
        next_token_logits = new_logits[-1, :]
        predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
        predicted_token_text = tokenizer.enc.decode([predicted_token_id])
        
        # Record token generation time
        metrics.record_token()
        
        # Print step output in the specified format
        print(f"Step {step + 1}: Generated token '{predicted_token_text}' (ID: {predicted_token_id})")
        
        # Add the generated token to the sequence
        generated_tokens = torch.cat([generated_tokens, torch.tensor([predicted_token_id], device=generated_tokens.device)])
        current_position += 1
        
        # Optional: Stop if we hit an end token
        if predicted_token_id == 128010:  # <|eot_id|> token
            break
    
    # End generation timing
    metrics.end_generation()
    
    # Print performance metrics
    metrics.print_metrics()
    
    return generated_tokens

# Run full generation loop with KV cache
generated_tokens = generate_with_kv_cache(Transformer, token_ids, max_new_tokens=100)

# Decode the final result
final_text = tokenizer.enc.decode(generated_tokens.tolist())
print("\n" + "="*50)
print("FINAL GENERATED TEXT:")
print("="*50)
print(final_text)




'''
# for later use:
# Single token generation with KV cache for debugging
def debug_single_token_generation(transformer, initial_tokens):
    print(f"Starting single token generation with KV cache")
    print(f"Initial prompt: {tokenizer.enc.decode(initial_tokens.tolist())}")
    print(f"Initial tokens: {initial_tokens.tolist()}")
    
    # First, process the initial prompt to populate KV cache
    print(f"\n=== STEP 1: Process initial prompt ===")
    logits = transformer(initial_tokens, start_idx=0)
    
    # Get prediction for next token
    next_token_logits = logits[-1, :]
    predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
    predicted_token_text = tokenizer.enc.decode([predicted_token_id])
    
    print(f"\n=== STEP 2: Generate next token ===")
    print(f"Predicted next token: '{predicted_token_text}' (ID: {predicted_token_id})")
    
    # Now generate the next token using KV cache
    new_token = torch.tensor([predicted_token_id])
    print(f"\n=== STEP 3: Process new token with KV cache ===")
    print(f"Processing new token: '{predicted_token_text}' at position {len(initial_tokens)}")
    
    # This should use KV cache and only process 1 new token
    new_logits = transformer(new_token, start_idx=len(initial_tokens))
    
    return new_logits, predicted_token_id

'''
