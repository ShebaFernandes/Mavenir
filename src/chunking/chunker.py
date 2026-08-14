"""
Turns parsed clause Sections into retrieval-ready Chunks.

Design choice driving hallucination reduction: chunk boundaries follow the
document's own clause/table/note structure (never split mid-block), and every
chunk carries a ready-made citation string plus the full heading breadcrumb.
That means the retriever's output is never "a stray sentence from somewhere
in a 900-page document" -- it's always "this is clause 4.2.8.1 of TS 23.501,
under Architecture model and concepts > ...", which the generation layer can
surface to the user and which a verification step can later check claims
against.

A section's content is packed greedily into chunks up to MAX_CHARS. Tables
and single very long paragraphs are never split internally (a half-table or
half-sentence is worse than one oversized chunk); everything else packs at
block boundaries so a NOTE never gets orphaned from the requirement it
qualifies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from parsing.docx_parser import Block, ParsedSpec, Section

MAX_CHARS = 1500  # ~350-400 tokens; keeps a chunk to roughly one focused idea
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+")


@dataclass
class Chunk:
    chunk_id: str
    spec_id: str
    release: str
    version: str
    clause_number: str | None
    section_title: str
    heading_path: str
    citation: str
    text: str
    part_index: int
    part_count: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "spec_id": self.spec_id,
            "release": self.release,
            "version": self.version,
            "clause_number": self.clause_number,
            "section_title": self.section_title,
            "heading_path": self.heading_path,
            "citation": self.citation,
            "text": self.text,
            "part_index": self.part_index,
            "part_count": self.part_count,
        }


def _block_text(block: Block) -> str:
    if block.type != "table":
        return block.text
    cap = f"{block.table_caption}\n" if block.table_caption else ""
    rows_txt = "\n".join(" | ".join(row) for row in block.table_rows)
    return f"{cap}{rows_txt}"


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Fallback for a single block that alone exceeds max_chars: split on
    sentence boundaries and repack, never mid-word."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [text]


def _pack_pieces(pieces: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if len(candidate) > max_chars and current:
            packed.append(current)
            current = piece
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _citation(spec: ParsedSpec, section: Section) -> str:
    where = f"clause {section.clause_number}" if section.clause_number else section.title
    return f'3GPP TS {spec.spec_id} V{spec.version} (Release {spec.release}), {where} "{section.title}"'


def chunk_section(spec: ParsedSpec, section: Section) -> list[Chunk]:
    if not section.blocks:
        return []

    atomic_pieces: list[str] = []
    for block in section.blocks:
        text = _block_text(block).strip()
        if not text:
            continue
        if block.type != "table" and len(text) > MAX_CHARS:
            atomic_pieces.extend(_split_long_text(text, MAX_CHARS))
        else:
            atomic_pieces.append(text)

    body_chunks = _pack_pieces(atomic_pieces, MAX_CHARS)
    if not body_chunks:
        return []

    citation = _citation(spec, section)
    chunks: list[Chunk] = []
    for i, body in enumerate(body_chunks):
        header = f"[{section.heading_path}]\n" if section.heading_path else ""
        chunk_id = f"{spec.spec_id}::{section.clause_number or section.title}::{i}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                spec_id=spec.spec_id,
                release=spec.release,
                version=spec.version,
                clause_number=section.clause_number,
                section_title=section.title,
                heading_path=section.heading_path,
                citation=citation,
                text=f"{header}{body}",
                part_index=i,
                part_count=len(body_chunks),
            )
        )
    return chunks


def chunk_spec(spec: ParsedSpec) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in spec.sections:
        chunks.extend(chunk_section(spec, section))
    return chunks
