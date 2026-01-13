import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import time

import config
import tokenizer

import time

import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


profiler_schedule = torch.profiler.schedule(
    wait=0,
    warmup=0,   
    active=10,   
    repeat=1
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.load("Llama3.2-1B-Instruct/consolidated.00.pth", map_location=device)

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|><br><br>If mountains could talk<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br>"
encoded = tokenizer.enc.encode(prompt, allowed_special="all")
token_ids = torch.tensor(encoded, device=device)
print(token_ids.shape)


class RMSNorm(nn.Module):
    def __init__(self, weight, eps=config.NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weight)

    def _norm(self, x):
        return torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * x * self.weight


def precompute_freqs_cis(head_dim, end, theta=config.ROPE_THETA, device=None):
    if device is None:
        device=torch.device("cpu")
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, device=device)
    freqs_matrix = torch.outer(t, freq)
    freqs_cos = torch.cos(freqs_matrix)
    freqs_sin = torch.sin(freqs_matrix)
    return freqs_cos, freqs_sin


def apply_RoPE(x, freqs_cos, freqs_sin, n_heads):
    x_even = x[:, :, ::2]
    x_odd = x[:, :, 1::2]
    freqs_cos_reshaped = freqs_cos.unsqueeze(1).expand(-1, n_heads, -1)
    freqs_sin_reshaped = freqs_sin.unsqueeze(1).expand(-1, n_heads, -1)
    x_even_rot = x_even * freqs_cos_reshaped - x_odd * freqs_sin_reshaped
    x_odd_rot = x_even * freqs_sin_reshaped + x_odd * freqs_cos_reshaped
    x_rot = torch.zeros_like(x, device=device)
    x_rot[:, :,::2] = x_even_rot
    x_rot[:, :, 1::2] = x_odd_rot
    return x_rot


class MHA(nn.Module):
    def __init__(self, wq, wk, wv, wo, n_heads=config.N_HEADS, n_kv_heads=config.N_KV_HEADS):
        super().__init__()
        self.q_proj_dim = wq.shape[0]
        self.k_proj_dim = wk.shape[0]
        self.v_proj_dim = wv.shape[0]
        self.wqkv = nn.Parameter(torch.cat([wq, wk, wv], dim=0))
        self.wo = nn.Parameter(wo)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = config.DIM // self.n_heads
        
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

    def forward(self, x, mask, freqs_cos, freqs_sin, start_idx=0):
        seq_len = x.shape[0]
        
        x = x.to(self.wqkv.dtype)
        qkv = torch.matmul(x, self.wqkv.T)
        Q, K, V = torch.split(
            qkv,
            [self.q_proj_dim, self.k_proj_dim, self.v_proj_dim],
            dim=-1,
        )

        Q = Q.view(seq_len, self.n_heads, self.head_dim)
        K = K.view(seq_len, self.n_kv_heads, self.head_dim)
        V = V.view(seq_len, self.n_kv_heads, self.head_dim)

        Q = apply_RoPE(Q, freqs_cos[start_idx:], freqs_sin[start_idx:], self.n_heads)
        K = apply_RoPE(K, freqs_cos[start_idx:], freqs_sin[start_idx:], self.n_kv_heads)

        if self.k_cache is None:
            max_cache_len = 256
            self.k_cache = torch.zeros(max_cache_len, self.n_kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)
            self.v_cache = torch.zeros(max_cache_len, self.n_kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)
            self.cache_len = 0
        
        self.k_cache[start_idx:start_idx+seq_len] = K
        self.v_cache[start_idx:start_idx+seq_len] = V
        self.cache_len = start_idx + seq_len

        K_cached = self.k_cache[:self.cache_len]
        V_cached = self.v_cache[:self.cache_len]
        
        K_repeated = torch.repeat_interleave(K_cached, self.n_heads//self.n_kv_heads, dim=1)
        V_repeated = torch.repeat_interleave(V_cached, self.n_heads//self.n_kv_heads, dim=1)

        Q = Q.transpose(0, 1)
        K_repeated = K_repeated.transpose(0, 1)
        V_repeated = V_repeated.transpose(0, 1)

        output = F.scaled_dot_product_attention(Q, K_repeated, V_repeated, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=1.0/math.sqrt(self.head_dim), enable_gqa=False)
        
        output = output.transpose(0, 1).contiguous().view(seq_len, -1)

        return torch.matmul(output, self.wo.T)
        


        

class FeedForward(nn.Module):
    def __init__(self, w1, w3, w2):
        super().__init__()
        self.w13 = nn.Parameter(torch.cat([w1, w3], dim=0))
        self.w1_dim = w1.shape[0]
        self.w2 = nn.Parameter(w2)

    def forward(self, x):
        fused = torch.matmul(x, self.w13.T)
        gate, up = torch.split(fused, [self.w1_dim, fused.shape[-1] - self.w1_dim], dim=-1)
        return torch.matmul(F.silu(gate) * up, self.w2.T)


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
        x = x.to(torch.bfloat16)
        attn_out = x + self.attn(self.attn_norm(x), mask, freqs_cos, freqs_sin, start_idx)
        ffn_out = attn_out + self.ffn(self.ffn_norm(attn_out))
        return ffn_out


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = config.N_LAYERS
        self.vocab_size = config.VOCAB_SIZE
        self.head_dim = config.DIM // config.N_HEADS
        
        self.tok_emb = nn.Embedding.from_pretrained(
            model["tok_embeddings.weight"], 
            freeze=True
        )
        
        self.norm = RMSNorm(model["norm.weight"])
        self.output_weights = nn.Parameter(model["output.weight"])
        
        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_idx))

        
        

    def forward(self, tokens, start_idx=0):
        x = self.tok_emb(tokens)

        seq_len = x.shape[0]
        current_len = start_idx + seq_len
        
        mask = torch.full((seq_len, current_len), float('-inf'), device=x.device)
        mask = mask.triu(diagonal=start_idx+1)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, current_len, device=x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cos, freqs_sin, start_idx)
        
        x = self.norm(x)
        logits = torch.matmul(x, self.output_weights.T).float()
        
        return logits


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
        
        ttft = self.first_token_time - self.start_time
        e2el = self.end_time - self.start_time
        token_gen_time = e2el - ttft
        
        output_tokens = self.total_tokens_generated - self.input_tokens
        tpot = token_gen_time / max(output_tokens - 1, 1) if output_tokens > 1 else 0
        
        itl_times = []
        for i in range(1, len(self.token_times)):
            itl_times.append(self.token_times[i] - self.token_times[i-1])
        avg_itl = sum(itl_times) / len(itl_times) if itl_times else 0
        
        tokens_per_second = output_tokens / token_gen_time if token_gen_time > 0 else 0
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
        print("[PERFORMANCE METRICS]")
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

import torch._inductor.config as inductor_config
import torch._inductor.utils as inductor_utils

inductor_config.max_autotune_gemm_backends = "TRITON"
inductor_utils.is_big_gpu = lambda _: True
inductor_config.force_fuse_int_mm_with_mul = True
inductor_config.triton.cudagraphs = False

print("\n" + "="*60)
print("Compiling prompt pass with torch.compile...")
print("="*60)

def run_prompt(tokens):
    return Transformer(tokens, start_idx=0)

compiled_prompt = torch.compile(run_prompt, mode="max-autotune-no-cudagraphs", dynamic=True)

if device.type == "cuda":
    _ = compiled_prompt(token_ids)
    torch.cuda.synchronize()




def generate_with_kv_cache(transformer, initial_tokens, max_new_tokens=80, profiler=None):
    metrics = PerformanceMetrics()
    metrics.start_generation(initial_tokens)
    
    logits = compiled_prompt(initial_tokens)
    
    next_token_logits = logits[-1, :]
    predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
    predicted_token_text = tokenizer.enc.decode([predicted_token_id])
    
    generated_tokens = initial_tokens.clone()
    current_position = len(initial_tokens)
    
    first_token_recorded = False
    
    for step in range(max_new_tokens):
        new_token = torch.tensor([predicted_token_id], device=initial_tokens.device)
        
        if not first_token_recorded:
            metrics.record_first_token()
            first_token_recorded = True
        
        new_logits = transformer(new_token, start_idx=current_position)
        
        next_token_logits = new_logits[-1, :]
        predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
        predicted_token_text = tokenizer.enc.decode([predicted_token_id])
        
        metrics.record_token()
        
        print(f"Step {step + 1}: Generated token '{predicted_token_text}' (ID: {predicted_token_id})")
        
        generated_tokens = torch.cat([generated_tokens, torch.tensor([predicted_token_id], device=generated_tokens.device)])
        current_position += 1
        
        if profiler is not None:
            torch.cuda.synchronize()
            profiler.step()

        if predicted_token_id == 128010:
            break
    
    metrics.end_generation()
    metrics.print_metrics()
    
    return generated_tokens




generated_tokens = generate_with_kv_cache(Transformer, token_ids, max_new_tokens=80, profiler=None)

final_text = tokenizer.enc.decode(generated_tokens.tolist())
print("\n" + "="*50)
print("FINAL GENERATED TEXT:")
print("="*50)
print(final_text)
