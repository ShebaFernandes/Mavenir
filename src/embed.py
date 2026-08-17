"""
Embedding pipeline: chunked JSONL -> persisted Chroma vector store.

    python src/embed.py

Reads every data/processed/chunks/{spec}.jsonl (produced by src/ingest.py),
rebuilds each line as a LangChain Document (page_content=text, the rest as
metadata), embeds it with a local sentence-transformers model, and persists
the vectors + metadata to a Chroma collection on disk. This stage is
deliberately decoupled from ingestion: re-embedding (e.g. to try a different
model) never requires re-parsing the .docx files.

Model: sentence-transformers/all-MiniLM-L6-v2 -- runs locally (no API key,
no per-chunk cost), 384-dim, fast enough on CPU for ~17k chunks. General-
purpose, not telecom-tuned; swap EMBED_MODEL below if retrieval quality on
acronym-heavy text needs it later. Embeddings are stored in Chroma, so
switching models means re-running this script into a fresh persist_directory
-- never mix vectors from two different models in one collection.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
PERSIST_DIR = ROOT / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "3gpp_specs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500  # stay under Chroma's sqlite max-batch-size limit


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.pop("text")
                # Chroma metadata values must be str/int/float/bool -- None
                # (e.g. clause_number for unnumbered sections) isn't allowed.
                metadata = {k: ("" if v is None else v) for k, v in record.items()}
                documents.append(Document(page_content=text, metadata=metadata))
    return documents


def embed_all() -> None:
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No chunks found in {CHUNKS_DIR}. Run src/ingest.py first.")
    print(f"Loaded {len(documents)} chunks from {CHUNKS_DIR}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        ids = [doc.metadata["chunk_id"] for doc in batch]
        store.add_documents(batch, ids=ids)
        done = min(start + BATCH_SIZE, len(documents))
        print(f"  embedded {done}/{len(documents)}")

    print(f"\nDone. {store._collection.count()} vectors in collection '{COLLECTION_NAME}'")
    print(f"Persisted to: {PERSIST_DIR}")


if __name__ == "__main__":
    embed_all()
