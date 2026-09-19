from __future__ import annotations

import asyncio
import re
from pathlib import Path

from langchain_pinecone import PineconeVectorStore
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.rag.embeddings import get_embeddings
from app.rag.ingest import load_chunks
from app.rag.schemas import Chunk, SearchHit, SearchQuery, SearchResult

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _key(source: str, page: int, text: str) -> str:
    return f"{source}|{page}|{text[:60]}"


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._chunks: list[Chunk] = load_chunks(Path(self._s.chunks_file))
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks])
        self._store = PineconeVectorStore(
            index_name=self._s.pinecone_index,
            embedding=get_embeddings(),
            namespace=self._s.pinecone_namespace,
            pinecone_api_key=self._s.pinecone_api_key,
        )

    def _bm25_rank(self, query: str, k: int, categoria: str) -> list[tuple[str, Chunk]]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[str, Chunk]] = []
        for i in order:
            c = self._chunks[i]
            if scores[i] <= 0:
                break
            if categoria and c.categoria != categoria:
                continue
            out.append((_key(c.source, c.page, c.text), c))
            if len(out) == k:
                break
        return out

    async def _vector_rank(self, query: str, k: int, categoria: str) -> list[tuple[str, Chunk]]:
        docs = await self._store.asimilarity_search(
            query, k=k, filter={"categoria": categoria} if categoria else None
        )
        out = []
        for d in docs:
            c = Chunk(text=d.page_content, source=d.metadata["source"],
                      categoria=d.metadata["categoria"], page=int(d.metadata["page"]))
            out.append((_key(c.source, c.page, c.text), c))
        return out

    async def search(self, q: SearchQuery) -> SearchResult:
        w_bm25 = self._s.rag_bm25_weight
        w_vec = 1.0 - w_bm25
        bm25_task = asyncio.to_thread(self._bm25_rank, q.query, q.k, q.categoria)
        vec_task = self._vector_rank(q.query, q.k, q.categoria)
        bm25_rank, vec_rank = await asyncio.gather(bm25_task, vec_task)

        fused: dict[str, tuple[float, Chunk]] = {}
        for weight, ranking in ((w_bm25, bm25_rank), (w_vec, vec_rank)):
            for pos, (key, chunk) in enumerate(ranking, start=1):
                prev = fused.get(key, (0.0, chunk))[0]
                fused[key] = (prev + weight / (60 + pos), chunk)  # RRF

        top = sorted(fused.values(), key=lambda x: x[0], reverse=True)[: q.k]
        hits = [
            SearchHit(text=c.text, source=c.source, categoria=c.categoria, page=c.page,
                      score=round(score, 6), citation=f"[{c.source} p.{c.page}]")
            for score, c in top
        ]
        return SearchResult(query=q.query, hits=hits, total=len(hits))
