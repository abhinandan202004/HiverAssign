from __future__ import annotations

import os
import json
import numpy as np
from typing import List, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.models import RetrievedDocument, IntentEnum


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """
    Hybrid Retriever combining lexical BM25 search with dense semantic vector search
    (all-MiniLM-L6-v2) via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        historical_path: str = "data/sample_historical.json",
        embeddings_path: str = "data/historical_embeddings.npz",
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        self.rrf_k = rrf_k
        self.corpus = self._load_corpus(historical_path)
        self.tokenized_corpus = [tokenize(doc["query_text"]) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense model and embedding cache
        self.model = SentenceTransformer(model_name)
        if os.path.exists(embeddings_path):
            data = np.load(embeddings_path)
            self.doc_embeddings = data["embeddings"]
        else:
            texts = [f"{d['query_text']} -> {d['resolution_text']}" for d in self.corpus]
            self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True)

    def _load_corpus(self, path: str) -> List[dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Historical resolutions file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """
        Retrieve top-k historical brand resolutions using Reciprocal Rank Fusion.
        """
        n_docs = len(self.corpus)
        if n_docs == 0:
            return []

        # 1. Lexical BM25 Ranking
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 2. Dense Semantic Cosine Similarity Ranking
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity on normalized vectors is dot product
        dense_scores = np.dot(self.doc_embeddings, query_emb)
        dense_ranked_indices = np.argsort(dense_scores)[::-1]

        # Build rank lookups (1-indexed rank)
        bm25_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_ranked_indices)}
        dense_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(dense_ranked_indices)}

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for doc_idx in range(n_docs):
            r_bm25 = bm25_ranks[doc_idx]
            r_dense = dense_ranks[doc_idx]
            rrf_scores[doc_idx] = (1.0 / (self.rrf_k + r_bm25)) + (1.0 / (self.rrf_k + r_dense))

        top_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]

        results: List[RetrievedDocument] = []
        for idx in top_indices:
            doc = self.corpus[idx]
            results.append(
                RetrievedDocument(
                    tweet_id=doc["tweet_id"],
                    query_text=doc["query_text"],
                    resolution_text=doc["resolution_text"],
                    intent=IntentEnum(doc["intent"]),
                    bm25_score=float(bm25_scores[idx]),
                    dense_score=float(dense_scores[idx]),
                    rrf_score=float(rrf_scores[idx]),
                )
            )

        return results
