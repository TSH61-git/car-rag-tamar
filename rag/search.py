"""Search over the chunks: by meaning (vectors), by words (BM25), and the two combined."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.embed import Embedder


def tokenize(text: str) -> list[str]:
    # Keeps codes like "of-450h" and "e-26" as single tokens.
    return re.findall(r"[\w\-]+", text.lower())


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([c.text for c in chunks], "passage")
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def vector_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks closest in meaning to the query, best first."""
        q = self.embedder.embed([query], "query")[0]
        scores = self.vectors @ q
        return [int(i) for i in np.argsort(-scores)[:k]]

    def bm25_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks with the best word match, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """TODO (step 3): Reciprocal Rank Fusion.

    `rankings` is a list of ranked lists of chunk indexes, best first.
    A chunk gets 1 / (k + rank) from every list it appears in (rank starts at 1),
    and the scores are summed. Return all chunk indexes, sorted by score, best first.
    """
    raise NotImplementedError("step 3: write rrf")


def is_current(chunk: Chunk) -> bool:
    """TODO (step 4): False for chunks from documents that were replaced by a newer version.

    Look at the metadata of the documents in data/ and decide which field tells you that.
    """
    raise NotImplementedError("step 4: write is_current")


def search(index: Index, query: str, mode: str = "vector", top_k: int = 5,
           current_only: bool = False) -> list[Chunk]:
    if mode == "vector":
        ranked = index.vector_search(query)
    elif mode == "hybrid":
        ranked = rrf([index.vector_search(query), index.bm25_search(query)])
    else:
        raise ValueError(f"unknown mode: {mode}")
    if current_only:
        ranked = [i for i in ranked if is_current(index.chunks[i])]
    return [index.chunks[i] for i in ranked[:top_k]]
