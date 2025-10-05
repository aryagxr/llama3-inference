import tiktoken
from tiktoken.load import load_tiktoken_bpe

tokenizer_path = "Llama3.2-1B-Instruct/tokenizer.model"
num_reserved_special_tokens = 256
pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens - 5)
        ]

mergeable_ranks = load_tiktoken_bpe(tokenizer_path)

special_tokens = {
    token: len(mergeable_ranks) + i
    for i, token in enumerate(special_tokens)
}


enc = tiktoken.Encoding(
    name="Llama3.2-1B-Instruct",
    pat_str=pat_str,
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens,
)

# print(enc.encode("Hello, world!"))