import torch
import time
from transformers import pipeline
import tiktoken

# Performance measurement class
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
        self.input_tokens = input_tokens
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

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype=torch.bfloat16,
)

messages = [
    {"role": "user", "content": "If mountains could talk"},
]

# Initialize performance metrics
metrics = PerformanceMetrics()

# Count input tokens
input_text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_tokens = len(pipe.tokenizer.encode(input_text))
metrics.start_generation(input_tokens)

print("Starting inference...")

# Generate text and measure timing
outputs = pipe(
    messages,
    max_new_tokens=100,
    do_sample=True,
    return_full_text=False,
)

# Record first token time (approximation)
metrics.record_first_token()

# Record end time
metrics.end_generation()

# Count output tokens
generated_text = outputs[0]['generated_text']
output_tokens = len(pipe.tokenizer.encode(generated_text))
metrics.total_tokens_generated = input_tokens + output_tokens

print(f"Generated text: {generated_text}")

# Print performance metrics
metrics.print_metrics()


