# Llama Inference Optimization

In this repo I iteratively optimize the Llama3.2-1B-Instruct model inference from the ground up. 

## Demo

### 01-naive.py
Baseline implementation with full attention computation for every token.
- **Throughput:** ~0.61 tokens/sec

![Naive Inference Token Generation](outputs/naive_tokens.gif)



### 02-kvcache.py
Adds KV cache optimization to avoid recomputing keys and values.
- **Throughput:** ~12.5 tokens/sec

![KV Cache Token Generation](outputs/kvcache_tokens.gif)



### 03-sdpa.py
Uses PyTorch's optimized Scaled Dot Product Attention (SDPA).
- **Throughput:** ~30.5 tokens/sec *(run to see actual metrics)*

## How to Run

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install torch
   pip install imageio
   ```

2. **Using conda (recommended):**
   ```bash
   conda create -n llama-exp python=3.10
   conda activate llama-exp
   pip install torch imageio
   ```

3. **Download the model:**
   - Place the Llama 3.2 1B Instruct model in `Llama3.2-1B-Instruct/`
   - Required files:
     - `consolidated.00.pth` (model weights)
     - `params.json` (model configuration)
     - `tokenizer.model` (tokenizer)

### Running Inference

```bash
# Naive implementation
python 01-naive.py

# KV cache optimized
python 02-kvcache.py

# SDPA optimized
python 03-sdpa.py
```

## Project Structure

- **`01-naive.py`** - Baseline implementation with full attention computation on every step
- **`02-kvcache.py`** - Adds KV cache optimization to avoid recomputing keys and values
- **`03-sdpa.py`** - Uses PyTorch's optimized Scaled Dot Product Attention (SDPA)
- **`03-fastmuls.py`** - Custom CUDA kernels for faster matrix multiplications
- **`baseline.py`** - Reference implementation
- **`extras/`** - Has scripts to run demo outputs
- **`cuda/`** - Custom cuda kernels to speed up inference
