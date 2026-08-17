"""
Command-line chatbot over the 3GPP specs.

    python src/chat.py

Loads the retriever once (embedding model + BM25 index), then runs a REPL:
type a question, get a citation-grounded answer. Each answer shows the mode
(claude / extractive / abstained), the retrieval confidence, and the numbered
clause citations so every claim is traceable to a specific 3GPP clause.

Commands: 'exit' / 'quit' to leave, ':k N' to change how many clauses are
retrieved for the next questions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import MAX_CONTEXT_CHUNKS, active_backend, answer
from retrieve import Retriever

BANNER = r"""
  3GPP RAG Chatbot  --  ask about Release 17 5G specs
  Answers are grounded in retrieved clauses and always cited.
  Type 'exit' to quit, ':k N' to change how many clauses are retrieved.
"""


def main() -> None:
    print(BANNER)
    print(f"  Backend: {active_backend()}\n")
    print("  Loading retriever (embedding model + BM25 index)...")
    retriever = Retriever()
    print("  Ready.\n")

    k = MAX_CONTEXT_CHUNKS
    while True:
        try:
            query = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        if query.startswith(":k "):
            try:
                k = max(1, int(query[3:].strip()))
                print(f"  [retrieving {k} clauses per question]\n")
            except ValueError:
                print("  usage: :k 8\n")
            continue

        result = answer(query, retriever, k=k)
        print()
        print(f"bot [{result.mode}, confidence {result.confidence:.2f}] >")
        print(result.text)
        print("\n  Sources:")
        for i, citation in enumerate(result.citations, start=1):
            print(f"    [{i}] {citation}")
        print()


if __name__ == "__main__":
    main()
