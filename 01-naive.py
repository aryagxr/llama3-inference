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
# print(emb_layer.weight)


