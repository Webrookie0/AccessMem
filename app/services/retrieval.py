"""BM25 keyword ranking and reciprocal rank fusion over authorized chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from rank_bm25 import BM25Plus


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    distance: float | None = None
    rrf_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def similarity(self) -> float | None:
        return None if self.distance is None else 1.0 - self.distance


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def keyword_search(chunks: list[RetrievedChunk], query: str, limit: int) -> list[RetrievedChunk]:
    """The caller must authorize the corpus BEFORE it reaches this function."""
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = tokenize(query)
    if not chunks or not terms:
        return []
    corpus = [tokenize(f"{chunk.metadata.get('title', '')}\n{chunk.text}") for chunk in chunks]
    if not any(corpus):
        return []
    scores = BM25Plus(corpus).get_scores(terms)
    query_terms = set(terms)
    # BM25Plus includes a baseline score; require an actual keyword match.
    matches = [
        (float(score), chunk)
        for score, chunk, tokens in zip(scores, chunks, corpus, strict=True)
        if query_terms.intersection(tokens)
    ]
    matches.sort(key=lambda match: (-match[0], match[1].id))
    return [chunk for _, chunk in matches[:limit]]


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    keywords: list[RetrievedChunk],
    *,
    rank_constant: int = 60,
    limit: int = 4,
) -> list[RetrievedChunk]:
    """Fuse by stable chunk ID. Duplicate IDs in a branch count only once."""
    if rank_constant < 1 or limit < 1:
        raise ValueError("rank_constant and limit must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    ranks: list[dict[str, int]] = []
    for ranking in (dense, keywords):
        branch_ranks: dict[str, int] = {}
        for chunk in ranking:
            if chunk.id in branch_ranks:
                continue
            rank = len(branch_ranks) + 1
            branch_ranks[chunk.id] = rank
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rank_constant + rank)
            chunks.setdefault(chunk.id, chunk)
        ranks.append(branch_ranks)
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        replace(
            chunks[chunk_id],
            rrf_score=scores[chunk_id],
            dense_rank=ranks[0].get(chunk_id),
            keyword_rank=ranks[1].get(chunk_id),
        )
        for chunk_id in order[:limit]
    ]
