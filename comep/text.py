import hashlib
import re

import numpy as np


class TokenCounter:
    _CACHE = {}

    def __init__(self, encoding=None):
        self.encoding = encoding
        self.encoder = None
        if encoding:
            if encoding not in self._CACHE:
                import tiktoken

                self._CACHE[encoding] = tiktoken.get_encoding(encoding)
            self.encoder = self._CACHE[encoding]

    @property
    def label(self):
        return f"tiktoken:{self.encoding}" if self.encoder else "lexical_estimate"

    def count(self, text):
        if self.encoder:
            return len(self.encoder.encode(text, disallowed_special=()))
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class LocalEncoder:
    _MODEL_CACHE = {}
    _TEXT_CACHE: dict[tuple[str | None, int, str], np.ndarray] = {}

    def __init__(self, model=None, dimensions=256):
        self.model_name = model
        self.dimensions = dimensions
        self.model = None
        if model:
            if model not in self._MODEL_CACHE:
                from sentence_transformers import SentenceTransformer

                self._MODEL_CACHE[model] = SentenceTransformer(model, device="cpu", local_files_only=True)
            self.model = self._MODEL_CACHE[model]

    @property
    def label(self):
        return self.model_name or f"lexical_hash:{self.dimensions}"

    def encode(self, text):
        key = (self.model_name, self.dimensions, text)
        if key in self._TEXT_CACHE:
            return self._TEXT_CACHE[key]
        if self.model:
            vec = np.asarray(self.model.encode(text, normalize_embeddings=True, show_progress_bar=False), dtype=float)
        else:
            vector = np.zeros(self.dimensions, dtype=float)
            for word in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
                vector[int.from_bytes(digest, "little") % self.dimensions] += 1.0
            norm = np.linalg.norm(vector)
            vec = vector / norm if norm else vector
        vec.setflags(write=False)
        self._TEXT_CACHE[key] = vec
        return vec


def similarity(first, second):
    return float(np.clip(np.dot(first, second), 0.0, 1.0))
