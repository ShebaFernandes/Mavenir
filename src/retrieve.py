"""
Hybrid retrieval over the 3GPP chunk store: dense vectors + BM25 keyword.

    from retrieve import Retriever
    r = Retriever()
    hits = r.search("What triggers an AMF relocation?", k=6)

Why hybrid. Standards text is acronym-dense ("AMF", "NGAP", "5G-GUTI") and
exact-token-heavy (clause numbers, IE names, cause codes). A dense embedding
model like all-MiniLM captures paraphrase/semantic similarity but routinely
misses a rare acronym that appears verbatim in the query and the target
clause; BM25 nails those exact-token matches but is blind to paraphrase. We
run both and fuse their rankings with Reciprocal Rank Fusion (RRF), which
needs no score calibration between the two very different scoring scales.

Confidence gating. Every result set carries a `confidence` derived from the
best dense relevance score. The generation layer uses it to abstain rather
than answer from weak retrieval -- a core anti-hallucination guard: if the
specs don't actually cover the question, saying so beats guessing.

The BM25 index is built in memory at construction time from the same JSONL
chunks that were embedded (data/processed/chunks/*.jsonl), so the dense and
sparse corpora are guaranteed to be the identical set of chunk_ids.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
PERSIST_DIR = ROOT / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "3gpp_specs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fetch this many candidates from each retriever before fusing. Larger than the
# final k so a chunk ranked mid-pack by one method can still surface if the
# other method rates it highly.
CANDIDATE_K = 20
RRF_K = 60  # standard RRF damping constant; larger = flatter rank weighting

# Below this best-dense-score, treat retrieval as too weak to ground an answer.
# all-MiniLM relevance scores on this corpus run ~0.3-0.6 for good hits; 0.30
# is a deliberately low floor that only trips on genuinely off-topic queries.
CONFIDENCE_FLOOR = 0.30

_token_re = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Keeps '5g', '23.501'->['23','501'],
    acronyms intact -- deliberately simple; BM25 rewards exact token overlap."""
    return _token_re.findall(text.lower())


@dataclass
class Hit:
    document: Document
    dense_score: float | None  # cosine relevance in [0,1], None if BM25-only
    rank: int                  # 1-based fused rank

    @property
    def citation(self) -> str:
        return self.document.metadata.get("citation", "")

    @property
    def text(self) -> str:
        return self.document.page_content


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    confidence: float  # best dense relevance score across hits, in [0,1]

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_FLOOR


class Retriever:
    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        if self._store._collection.count() == 0:
            raise SystemExit(
                f"Vector store at {PERSIST_DIR} is empty. Run src/embed.py first."
            )
        self._docs, self._bm25 = self._build_bm25()

    def _build_bm25(self) -> tuple[list[Document], BM25Okapi]:
        docs: list[Document] = []
        corpus: list[list[str]] = []
        for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    text = record.pop("text")
                    metadata = {k: ("" if v is None else v) for k, v in record.items()}
                    docs.append(Document(page_content=text, metadata=metadata))
                    corpus.append(_tokenize(text))
        if not docs:
            raise SystemExit(f"No chunks found in {CHUNKS_DIR}. Run src/ingest.py first.")
        return docs, BM25Okapi(corpus)

    def _dense(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def _sparse(self, query: str, k: int) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 6) -> RetrievalResult:
        dense = self._dense(query, CANDIDATE_K)
        sparse = self._sparse(query, CANDIDATE_K)

        # Reciprocal Rank Fusion: each list contributes 1/(RRF_K + rank) to a
        # chunk's fused score. A chunk found by both retrievers gets both terms.
        fused: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        by_id: dict[str, Document] = {}

        for rank, (doc, score) in enumerate(dense, start=1):
            cid = doc.metadata["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            dense_scores[cid] = score
            by_id[cid] = doc
        for rank, doc in enumerate(sparse, start=1):
            cid = doc.metadata["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(cid, doc)

        ordered = sorted(fused, key=lambda c: fused[c], reverse=True)[:k]
        hits = [
            Hit(document=by_id[cid], dense_score=dense_scores.get(cid), rank=i)
            for i, cid in enumerate(ordered, start=1)
        ]
        confidence = max((s for s in dense_scores.values()), default=0.0)
        return RetrievalResult(query=query, hits=hits, confidence=confidence)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the role of the AMF in the 5G core?"
    r = Retriever()
    result = r.search(q)
    print(f"Query: {q}")
    print(f"Confidence: {result.confidence:.3f}  (confident={result.is_confident})\n")
    for hit in result.hits:
        ds = f"{hit.dense_score:.3f}" if hit.dense_score is not None else "  -  "
        print(f"[{hit.rank}] dense={ds}  {hit.citation}")
        snippet = hit.text.replace("\n", " ")[:160]
        print(f"     {snippet}...\n")
