import torch
import time
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

print("Starting inference...")
start_time = time.time()

outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=True,
    return_full_text=False,
)

first_token_time = time.time()
ttft = first_token_time - start_time

print(f"Time to First Token (TTFT): {ttft:.4f} seconds")
print(f"Generated text: {outputs[0]['generated_text']}")


