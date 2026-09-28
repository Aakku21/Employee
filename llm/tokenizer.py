"""Thin wrapper around tiktoken tokenizers.

gpt2 is fine for plain English but wastes tokens on code (it splits indentation
into many pieces). cl100k_base packs code ~1.8x tighter, so it is the default.
"""

import numpy as np
import tiktoken

SUPPORTED = ("cl100k_base", "gpt2", "o200k_base")


class Tokenizer:
    def __init__(self, name: str = "cl100k_base"):
        if name not in SUPPORTED:
            raise ValueError(f"tokenizer must be one of {SUPPORTED}, got {name!r}")
        self.name = name
        self._enc = tiktoken.get_encoding(name)
        self.eot = self._enc.eot_token  # marks the end of each document
        self.vocab_size = self._enc.n_vocab

    @property
    def dtype(self):
        return np.uint16 if self.vocab_size <= 2**16 else np.uint32

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        return self._enc.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
