from __future__ import annotations
import re

class Chunker:
    def __init__(self, chunk_size_tokens: int = 350, overlap_tokens: int = 50):
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be < chunk_size_tokens")

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\S+", text)

    def _detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def chunk(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        tokens = self._tokenize(text)
        n = len(tokens)
        if n == 0:
            return []

        step = self.chunk_size_tokens - self.overlap_tokens
        chunks = []
        start = 0

        while start < n:
            end = min(start + self.chunk_size_tokens, n)
            chunks.append(self._detokenize(tokens[start:end]))

            if end >= n:
                break

            start += step

        return chunks
