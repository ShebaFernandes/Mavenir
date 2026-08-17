# 3GPP RAG Chatbot

A Retrieval-Augmented Generation chatbot over 3GPP telecom standards, designed for
near-zero hallucination. This repo currently covers **ingestion** (step 1 of the
pipeline) -- turning raw 3GPP specification documents into clause-tagged,
citation-ready chunks.

## Why this design

3GPP specs are the ground truth, but they're huge and cross-referenced. The
hallucination risk in a naive RAG setup comes from chunking that loses clause
structure: retrieve a fragment with no traceable source, and the model has to
guess or paraphrase. This pipeline keeps every chunk tied to its exact clause
number and heading path (e.g. `TS 23.501 clause 4.2.8.1`), so retrieval results
are always citable and verifiable against the source spec -- not paraphrased.

## Data source

12 curated Release 17 5G Standalone core specifications, downloaded directly
from the official 3GPP FTP server (not a secondary/scraped source):

| Spec | Covers |
|---|---|
| TS 23.501 | 5G System architecture |
| TS 23.502 | 5G System procedures |
| TS 23.503 | Policy and charging control |
| TS 24.501 | NAS protocol (5G) |
| TS 24.301 | NAS protocol (LTE/EPS) |
| TS 38.300 | NR / NG-RAN overall description |
| TS 38.331 | RRC protocol |
| TS 33.501 | 5G security architecture |
| TS 29.500 | Service-based architecture |
| TS 29.518 | AMF services |
| TS 38.401 | NG-RAN architecture |
| TS 38.413 | NGAP |

Every downloaded file's release is verified against its own cover-page version
string (`3GPP TS x V17.y.z`), not just trusted from the FTP filename -- the
filename's release-letter encoding is easy to misread.

## Pipeline

```
scripts/download_specs.py   raw .docx   -->  data/raw/docx/
src/parsing/docx_parser.py  .docx       -->  clause-tagged Sections (in memory)
src/chunking/chunker.py     Sections    -->  citation-ready Chunks
src/ingest.py                           -->  data/processed/chunks/{spec}.jsonl
src/embed.py                chunks      -->  data/vectorstore/chroma  (16.8k vectors)
src/retrieve.py             query       -->  fused dense+BM25 clause hits
src/generate.py             hits+query  -->  citation-grounded answer (or abstain)
src/chat.py  /  app.py                  -->  CLI + Streamlit chat interfaces
```

- **Parsing** walks the document body in original order (paragraphs and tables
  interleaved) and uses the spec's own Word heading styles (`Heading 1`..`Heading 9`)
  to reconstruct the clause hierarchy, including annex sub-clauses (`D.1`,
  `K.2.2.3`) and letter-suffixed clauses (`4.2.5a`). Tables stay attached to the
  clause they appear in and are never split across chunks.
- **Chunking** packs a clause's content into ~1500-character chunks at block
  boundaries (never mid-table, mid-sentence only as a last-resort fallback for
  an unusually long single paragraph). Every chunk carries its full heading
  breadcrumb and a ready-made citation string.

## Running it

Build the data once (the `data/` directory is gitignored -- it's fully
reproducible from these commands, so it isn't committed to keep the repo lean):

```bash
pip install -r requirements.txt
python scripts/download_specs.py   # -> data/raw/docx/*.docx (~62 MB)
python src/ingest.py               # -> data/processed/chunks/*.jsonl
python src/embed.py                # -> data/vectorstore/chroma  (~17k vectors)
```

Then chat, either in the terminal or the browser:

```bash
python src/chat.py                 # CLI REPL
streamlit run app.py               # web UI at http://localhost:8501
```

By default answers run in **extractive** mode (no API key needed): the bot
quotes the single most relevant clause verbatim and lists the others, so it
cannot hallucinate. Set an API key to switch on **LLM synthesis** -- fluent
answers composed strictly from the retrieved clauses, with inline `[1][2]`
citations. Either way the bot **abstains** when retrieval confidence is below a
floor, rather than guessing.

The backend is chosen by which key is set (Gemini takes priority over Claude).
Put your key in a `.env` file at the repo root -- it's loaded automatically at
startup and is gitignored, so the key is never committed:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...  (or ANTHROPIC_API_KEY=sk-ant-...)
```

`.env` contents:

```
GEMINI_API_KEY=your-gemini-api-key-here
# GEMINI_MODEL=gemini-1.5-flash        # optional override, e.g. gemini-1.5-pro
```

(A shell `export GEMINI_API_KEY=...` still works too and takes precedence over
`.env`.)

Each line of `data/processed/chunks/{spec}.jsonl` is one chunk:

```json
{
  "chunk_id": "23.501::4.2.8.1::0",
  "spec_id": "23.501",
  "release": "17",
  "version": "17.15.0",
  "clause_number": "4.2.8.1",
  "section_title": "General Concepts to Support Trusted and Untrusted Non-3GPP Access",
  "heading_path": "4 Architecture model and concepts > 4.2 Architecture reference model > ...",
  "citation": "3GPP TS 23.501 V17.15.0 (Release 17), clause 4.2.8.1 \"...\"",
  "text": "[heading breadcrumb]\n...clause text...",
  "part_index": 0,
  "part_count": 1
}
```

Current output: **~16,800 chunks** across the 12 specs.

## The RAG layer

Once chunks are embedded into Chroma, three modules turn them into a chatbot:

- **Retrieval (`src/retrieve.py`)** runs *hybrid* search: a dense
  `all-MiniLM-L6-v2` vector search (good at paraphrase/semantics) and a BM25
  keyword search (good at the exact acronyms and clause numbers that saturate
  standards text), fused with Reciprocal Rank Fusion so a chunk strong in
  either method surfaces. The best dense score becomes a **confidence** used
  downstream to decide whether to answer at all.
- **Generation (`src/generate.py`)** hands the numbered clause excerpts to the
  answer backend under a strict *ground-or-abstain* prompt. It uses **Gemini**
  (`GEMINI_API_KEY`) or **Claude** (`ANTHROPIC_API_KEY`) when a key is set;
  without one it falls back to an **extractive** answer that quotes the top
  clause verbatim (hallucination-proof by construction). If retrieval
  confidence is below the floor it **abstains** before any model runs -- weak
  retrieval never gets paraphrased into a confident-sounding guess.
- **Interfaces (`src/chat.py`, `app.py`)** expose it as a terminal REPL and a
  Streamlit web app. Both show every answer's mode, confidence, and the exact
  clause citations backing it.

## Not yet built (next steps)

- Answer verification pass (check each generated claim against its cited chunk)
- Cross-reference resolution ("see clause 5.4.2") to pull in referenced clauses
- A telecom-tuned or larger embedding model, re-embedded into a fresh Chroma
  collection, if retrieval quality on acronym-dense queries needs it
