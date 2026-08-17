"""
Ingestion pipeline entrypoint: raw .docx -> clause-tagged, citation-ready chunks.

    python src/ingest.py

Reads every .docx in data/raw/docx/, parses clause structure, chunks it, and
writes one JSONL file per spec to data/processed/chunks/, plus a manifest.json
summarizing what was produced. This is the hand-off point to the next stage
(embedding + vector store), which is not implemented yet by design -- get
ingestion right and verified before building on top of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking.chunker import chunk_spec
from parsing.docx_parser import parse_docx

ROOT = Path(__file__).resolve().parent.parent
RAW_DOCX_DIR = ROOT / "data" / "raw" / "docx"
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"


def ingest_all() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    docx_files = sorted(RAW_DOCX_DIR.glob("*.docx"))
    if not docx_files:
        raise SystemExit(f"No .docx files found in {RAW_DOCX_DIR}. Run scripts/download_specs.py first.")

    manifest = []
    total_chunks = 0

    for path in docx_files:
        spec_id = path.stem
        print(f"[parse] {spec_id}")
        spec = parse_docx(path, spec_id)

        print(f"[chunk] {spec_id}: {len(spec.sections)} sections")
        chunks = chunk_spec(spec)

        out_path = CHUNKS_DIR / f"{spec_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in chunks:
                record = {"text": doc.page_content, **doc.metadata}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        avg_len = sum(len(doc.page_content) for doc in chunks) / len(chunks) if chunks else 0
        print(f"        -> {len(chunks)} chunks, avg {avg_len:.0f} chars -> {out_path.name}")

        if spec.release != "17":
            print(f"        [WARN] {spec_id}: expected Release 17, got Release {spec.release}")

        manifest.append(
            {
                "spec_id": spec_id,
                "title": spec.title,
                "release": spec.release,
                "version": spec.version,
                "sections": len(spec.sections),
                "chunks": len(chunks),
            }
        )
        total_chunks += len(chunks)

    manifest_path = ROOT / "data" / "processed" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. {total_chunks} chunks across {len(docx_files)} specs.")
    print(f"Manifest: {manifest_path}")
    print(f"Chunks:   {CHUNKS_DIR}")


if __name__ == "__main__":
    ingest_all()
