"""
Turns parsed clause Sections into retrieval-ready LangChain Documents.

Design choice driving hallucination reduction: chunk boundaries follow the
document's own clause/table/note structure (never split mid-block), and every
chunk carries a ready-made citation string plus the full heading breadcrumb.
That means the retriever's output is never "a stray sentence from somewhere
in a 900-page document" -- it's always "this is clause 4.2.8.1 of TS 23.501,
under Architecture model and concepts > ...", which the generation layer can
surface to the user and which a verification step can later check claims
against.

Packing uses LangChain's RecursiveCharacterTextSplitter: within a section, a
table is always its own unsplit chunk (a half-table is worse than one
oversized chunk); everything else is concatenated on block boundaries ("\n\n")
and handed to the splitter, which packs greedily up to MAX_CHARS and only
falls back to sentence/word splits for a single paragraph that alone exceeds
the cap.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from parsing.docx_parser import Block, ParsedSpec, Section

MAX_CHARS = 1500  # ~350-400 tokens; keeps a chunk to roughly one focused idea
CHUNK_OVERLAP = 0  # clause boundaries already isolate topics; overlap would blur citations

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _block_text(block: Block) -> str:
    if block.type != "table":
        return block.text
    cap = f"{block.table_caption}\n" if block.table_caption else ""
    rows_txt = "\n".join(" | ".join(row) for row in block.table_rows)
    return f"{cap}{rows_txt}"


def _citation(spec: ParsedSpec, section: Section) -> str:
    where = f"clause {section.clause_number}" if section.clause_number else section.title
    return f'3GPP TS {spec.spec_id} V{spec.version} (Release {spec.release}), {where} "{section.title}"'


def _section_body_chunks(section: Section) -> list[str]:
    """Split a section's blocks into <=MAX_CHARS pieces, keeping every table
    intact regardless of size."""
    segments: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            segments.extend(_splitter.split_text("\n\n".join(buffer)))
            buffer.clear()

    for block in section.blocks:
        text = _block_text(block).strip()
        if not text:
            continue
        if block.type == "table":
            flush()
            segments.append(text)
        else:
            buffer.append(text)
    flush()
    return segments


def chunk_section(spec: ParsedSpec, section: Section) -> list[Document]:
    if not section.blocks:
        return []

    body_chunks = _section_body_chunks(section)
    if not body_chunks:
        return []

    citation = _citation(spec, section)
    documents: list[Document] = []
    for i, body in enumerate(body_chunks):
        header = f"[{section.heading_path}]\n" if section.heading_path else ""
        chunk_id = f"{spec.spec_id}::{section.clause_number or section.title}::{i}"
        documents.append(
            Document(
                page_content=f"{header}{body}",
                metadata={
                    "chunk_id": chunk_id,
                    "spec_id": spec.spec_id,
                    "release": spec.release,
                    "version": spec.version,
                    "clause_number": section.clause_number,
                    "section_title": section.title,
                    "heading_path": section.heading_path,
                    "citation": citation,
                    "part_index": i,
                    "part_count": len(body_chunks),
                },
            )
        )
    return documents


def chunk_spec(spec: ParsedSpec) -> list[Document]:
    documents: list[Document] = []
    seen_keys: dict[str, int] = {}
    for section in spec.sections:
        key = section.clause_number or section.title
        seen_keys[key] = seen_keys.get(key, 0) + 1
        occurrence = seen_keys[key]

        section_docs = chunk_section(spec, section)
        if occurrence > 1:
            # Some specs reuse a clause number for two distinct sub-clauses
            # (e.g. TS 24.501 has two "8.2.6.35" headings under 8.2.6) --
            # disambiguate so chunk_id stays unique across the whole spec.
            for doc in section_docs:
                doc.metadata["chunk_id"] = doc.metadata["chunk_id"].replace(
                    f"{key}::", f"{key}~{occurrence}::", 1
                )
        documents.extend(section_docs)
    return documents
