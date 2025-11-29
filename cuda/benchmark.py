import torch
from torch.utils.cpp_extension import load
import time

# Compile and load the CUDA kernel
print("Compiling CUDA kernel...")
matmul_module = load(
    name="matmul_module",
    sources=["/home/ari/llama3-inference/cuda/matmul.cu"],  
    extra_cuda_cflags=["-O3"],
    verbose=True
)
print("Compilation done!\n")

# Attention dimensions
seq_len = 512
hidden_dim = 2048

print(f"Testing with seq_len={seq_len}, hidden_dim={hidden_dim}")
x = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
wq = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=torch.bfloat16)

# Test correctness
print("\n=== Correctness Test ===")
Q_custom = matmul_module.matmul(x, wq.T)  # [seq_len, 2048]
Q_torch = torch.matmul(x, wq.T)           # [seq_len, 2048]

print(f"Custom output shape: {Q_custom.shape}")
print(f"Torch output shape: {Q_torch.shape}")
print(f"Max difference: {(Q_custom - Q_torch).abs().max().item():.6f}")
print(f"Mean difference: {(Q_custom - Q_torch).abs().mean().item():.6f}")

try:
    torch.testing.assert_close(Q_custom, Q_torch, rtol=1e-2, atol=1e-2)
    print("Matmul is correct!")
except AssertionError as e:
    print("Matmul failed correctness test!")
    print(e)
    print("\nFirst few elements comparison:")
    print("Custom:", Q_custom[0, :5])
    print("Torch: ", Q_torch[0, :5])
    exit(1)

# Benchmark
print("\n=== Performance Benchmark ===")
num_warmup = 10
num_iterations = 100

# Warmup
for _ in range(num_warmup):
    _ = matmul_module.matmul(x, wq.T)
    _ = torch.matmul(x, wq.T)
torch.cuda.synchronize()

# Benchmark custom kernel
start = time.time()
for _ in range(num_iterations):
    Q_custom = matmul_module.matmul(x, wq.T)
torch.cuda.synchronize()
custom_time = (time.time() - start) / num_iterations

# Benchmark PyTorch
start = time.time()
for _ in range(num_iterations):
    Q_torch = torch.matmul(x, wq.T)
torch.cuda.synchronize()
torch_time = (time.time() - start) / num_iterations

print(f"Custom kernel: {custom_time*1000:.3f} ms")
print(f"PyTorch:       {torch_time*1000:.3f} ms")
print(f"Speedup:       {torch_time/custom_time:.2f}x")

print("\nAll tests passed! Ready to use in attention.")