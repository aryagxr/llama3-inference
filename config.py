import json
import torch

with open("Llama3.2-1B-Instruct/params.json", "r") as f:
    config = json.load(f)

MODEL_CONFIG = config

DIM = config['dim']
N_LAYERS = config['n_layers']
N_HEADS = config['n_heads']
N_KV_HEADS = config['n_kv_heads']
VOCAB_SIZE = config['vocab_size']
FFN_DIM_MULTIPLIER = config['ffn_dim_multiplier']
MULTIPLE_OF = config['multiple_of']
NORM_EPS = config['norm_eps']
ROPE_THETA = config['rope_theta']
USE_SCALED_ROPE = config['use_scaled_rope']

if __name__ == "__main__":
    print("Model Configuration:")
    print(config)

    model = torch.load("Llama3.2-1B-Instruct/consolidated.00.pth", map_location=torch.device('cpu'))
    print(json.dumps(list(model.keys())[:40], indent=4))
    print(model["layers.0.ffn_norm.weight"].shape)
