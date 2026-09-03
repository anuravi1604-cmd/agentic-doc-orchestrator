"""
retrieval_tool.py
------------------
Lightweight hybrid retriever: TF-IDF cosine similarity (lexical) blended
with a simple char-ngram embedding proxy (semantic-ish), fused the same way
ContextIQ fuses BM25 + BGE dense embeddings (weighted linear combination).

Kept dependency-light (sklearn + numpy only, both already available) so the
whole project runs offline with no API keys required for the retrieval path.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchHit:
    doc_id: str
    text: str
    score: float


class HybridRetriever:
    def __init__(self, docs: List[dict]):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        # Lexical channel: word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(stop_words="english")
        self.word_matrix = self.word_vectorizer.fit_transform(self.texts)
        # "Semantic-ish" channel: char n-gram TF-IDF, which captures
        # sub-word / morphological similarity TF-IDF-on-words misses --
        # a cheap, offline proxy for a dense embedding channel.
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.char_matrix = self.char_vectorizer.fit_transform(self.texts)

    @classmethod
    def from_jsonl(cls, path: str) -> "HybridRetriever":
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return cls(docs)

    def search(self, query: str, top_k: int = 3, alpha: float = 0.6) -> List[SearchHit]:
        """
        alpha weights the lexical (word TF-IDF) channel; (1 - alpha) weights
        the char-ngram channel -- same fusion pattern as ContextIQ's
        0.7 * semantic + 0.3 * keyword blend, just with different channels.
        """
        word_q = self.word_vectorizer.transform([query])
        char_q = self.char_vectorizer.transform([query])

        word_scores = cosine_similarity(word_q, self.word_matrix).flatten()
        char_scores = cosine_similarity(char_q, self.char_matrix).flatten()

        fused = alpha * word_scores + (1 - alpha) * char_scores
        ranked_idx = np.argsort(-fused)[:top_k]

        return [
            SearchHit(
                doc_id=self.docs[i].get("id", str(i)),
                text=self.texts[i],
                score=float(fused[i]),
            )
            for i in ranked_idx
        ]
