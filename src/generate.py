"""
Citation-grounded answer generation over retrieved 3GPP clauses.

    from retrieve import Retriever
    from generate import answer
    result = answer("What is the AMF?", Retriever())
    print(result.text)

Two design guarantees carried over from ingestion/retrieval:

1. Ground-or-abstain. The model is instructed to answer ONLY from the
   retrieved clauses and to say it cannot answer when they don't cover the
   question. If retrieval confidence is below the floor (see retrieve.py), we
   abstain *before* calling any model at all -- weak retrieval never gets a
   chance to be paraphrased into a confident-sounding guess.

2. Every answer is traceable. The context handed to the model is numbered
   [1], [2], ... and the model is told to cite those markers inline; the exact
   clause citation for each marker is returned alongside the answer so the UI
   can render verifiable references.

Backends, in priority order:
  1. GEMINI  -- if GEMINI_API_KEY (or GOOGLE_API_KEY) is set and the
     `google-generativeai` package is installed. Answers are synthesized by
     Google Gemini under a strict grounding system prompt.
  2. CLAUDE  -- if ANTHROPIC_API_KEY is set and `anthropic` is installed.
  3. EXTRACTIVE (no key) -- returns the top retrieved clause(s) verbatim with
     citations. Extractive answers cannot hallucinate (they are quotes), so
     the project is useful and honest out of the box, and gains fluent
     synthesis the moment a key is provided.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from retrieve import Hit, RetrievalResult, Retriever

# Load API keys from a project-root .env if present. Pointing at an explicit
# path (rather than relying on cwd) means keys load whether you run from the
# repo root, from src/, or via `streamlit run app.py`. Real env vars already
# set in the shell win -- load_dotenv does not override them by default.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_CONTEXT_CHUNKS = 6
MAX_TOKENS = 1024

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a precise assistant for 3GPP telecommunications standards. You
    answer strictly and only from the numbered clause excerpts provided in
    each question. These excerpts are verbatim text from Release 17 3GPP
    specifications.

    Rules:
    - Use ONLY information present in the provided excerpts. Never rely on
      outside knowledge, even if you think you know the answer.
    - Cite the excerpt number(s) you used inline, like [1] or [2][4], next to
      each claim.
    - If the excerpts do not contain enough information to answer, say so
      plainly: "The retrieved 3GPP clauses do not cover this." Do not guess.
    - Be concise and technical. Prefer the specification's own terminology.
    - Do not invent clause numbers, IE names, or procedures not in the text.
    """
)


@dataclass
class Answer:
    query: str
    text: str
    citations: list[str]          # citation strings, index-aligned to [1],[2],...
    hits: list[Hit] = field(default_factory=list)
    mode: str = "extractive"      # "gemini" | "claude" | "extractive" | "abstained"
    confidence: float = 0.0

    @property
    def abstained(self) -> bool:
        return self.mode == "abstained"


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.citation}\n{hit.text.strip()}")
    return "\n\n".join(blocks)


def _gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _using_gemini() -> bool:
    if not _gemini_key():
        return False
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        return False
    return True


def _gemini_answer(query: str, hits: list[Hit]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=_gemini_key())
    context = _format_context(hits)
    # Gemini has no dedicated system role in this SDK; fold the grounding rules
    # into system_instruction so they still bind the model's behavior.
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    user = (
        f"Retrieved 3GPP clause excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the excerpts above, citing excerpt numbers inline."
    )
    resp = model.generate_content(
        user,
        generation_config={"max_output_tokens": MAX_TOKENS, "temperature": 0.0},
    )
    return resp.text.strip()


def _using_claude() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _claude_answer(query: str, hits: list[Hit]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    context = _format_context(hits)
    user = (
        f"Retrieved 3GPP clause excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the excerpts above, citing excerpt numbers inline."
    )
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def _extractive_answer(query: str, hits: list[Hit]) -> str:
    """No-LLM fallback: quote the single most relevant clause, then list the
    other retrieved clauses as further reading. Honest and hallucination-proof
    -- it never paraphrases, it quotes."""
    top = hits[0]
    body = top.text.strip()
    # Drop the leading "[heading breadcrumb]" line the chunker prepends, so the
    # quote reads as clause prose.
    lines = body.split("\n")
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        body = "\n".join(lines[1:]).strip()

    out = [
        "No language model is configured (ANTHROPIC_API_KEY unset), so here is "
        "the most relevant clause verbatim rather than a synthesized answer:",
        "",
        f"From {top.citation} [1]:",
        f'"{body}"',
    ]
    if len(hits) > 1:
        out.append("")
        out.append("Other potentially relevant clauses:")
        for i, hit in enumerate(hits[1:], start=2):
            out.append(f"  [{i}] {hit.citation}")
    return "\n".join(out)


def active_backend() -> str:
    """Human-readable name of the backend that answer() will use, for UIs."""
    if _using_gemini():
        return f"Gemini ({GEMINI_MODEL})"
    if _using_claude():
        return f"Claude ({ANTHROPIC_MODEL})"
    return "Extractive (no API key)"


def answer(query: str, retriever: Retriever, k: int = MAX_CONTEXT_CHUNKS) -> Answer:
    result: RetrievalResult = retriever.search(query, k=k)
    citations = [h.citation for h in result.hits]

    if not result.is_confident:
        return Answer(
            query=query,
            text=(
                "I can't answer that from the 3GPP specifications I have indexed. "
                "The closest clauses retrieved were not a strong enough match to "
                "answer reliably, so I'm abstaining rather than guessing.\n\n"
                "Closest clauses (for reference):\n"
                + "\n".join(f"  - {c}" for c in citations[:3])
            ),
            citations=citations,
            hits=result.hits,
            mode="abstained",
            confidence=result.confidence,
        )

    if _using_gemini():
        text = _gemini_answer(query, result.hits)
        mode = "gemini"
    elif _using_claude():
        text = _claude_answer(query, result.hits)
        mode = "claude"
    else:
        text = _extractive_answer(query, result.hits)
        mode = "extractive"

    return Answer(
        query=query,
        text=text,
        citations=citations,
        hits=result.hits,
        mode=mode,
        confidence=result.confidence,
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the role of the AMF in the 5G core?"
    r = Retriever()
    a = answer(q, r)
    print(f"Q: {q}")
    print(f"[mode={a.mode}  confidence={a.confidence:.3f}]\n")
    print(a.text)
    print("\nCitations:")
    for i, c in enumerate(a.citations, start=1):
        print(f"  [{i}] {c}")
