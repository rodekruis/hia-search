"""Time each stage of a chat turn against the real services (prints durations only).

Usage:
    uv run python scripts/profile_latency.py [--sheet ID] [--message TEXT] [--follow-up TEXT]

Side effects: a few LLM/embedding calls and one throwaway thread in the checkpoint DB.
Never prints configuration values or prompts.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import uuid
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)  # .env must win over stale shell values

from langchain.messages import HumanMessage  # noqa: E402

import agents.rag_agent as rag_agent  # noqa: E402
from routes.chat import chat  # noqa: E402
from utils.prompt_loader import get_system_prompt  # noqa: E402
from utils.translator import detect_language, translate  # noqa: E402
from utils.vector_store import get_vector_store  # noqa: E402

DEFAULT_SHEET = "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04"
rows: list[tuple[str, float]] = []


def timed(label: str, fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    rows.append((label, (perf_counter() - start) * 1000))
    return result


def repeat_ms(fn, n: int = 3) -> str:
    samples = []
    for _ in range(n):
        start = perf_counter()
        fn()
        samples.append((perf_counter() - start) * 1000)
    return f"min {min(samples):.0f} / median {statistics.median(samples):.0f} ms"


def graph_turn(label: str, graph, message: str, config: dict) -> None:
    """Per-node timings via stream_mode='updates' (includes checkpoint writes)."""
    start = perf_counter()
    last = start
    for update in graph.stream(
        {"messages": [HumanMessage(message)]}, config, stream_mode="updates"
    ):
        now = perf_counter()
        for node in update:
            rows.append((f"{label}: node {node}", (now - last) * 1000))
        last = now
    rows.append((f"{label}: graph total", (perf_counter() - start) * 1000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--message", default="Where can I find a doctor?")
    parser.add_argument("--follow-up", default="And what if it is an emergency?")
    parser.add_argument("--foreign", default="Waar kan ik een dokter vinden?")
    args = parser.parse_args()

    print("== startup ==")
    timed("init_rag_agent (pool + setup + compile)", rag_agent.init_rag_agent)
    graph = rag_agent.get_rag_agent()

    def pg_ping():
        with rag_agent._checkpointer_pool.connection() as conn:
            conn.execute("SELECT 1")

    print(f"postgres SELECT 1: {repeat_ms(pg_ping)}")

    print("\n== per-dependency ==")
    store = timed(
        "get_vector_store cold (construct + count)",
        get_vector_store, args.sheet, check_if_exists=True,
    )
    timed("get_vector_store warm (cache hit)", get_vector_store, args.sheet, check_if_exists=True)
    print(f"azure search count_documents: {repeat_ms(store.count_documents)}")
    print(f"azure openai embed_query: {repeat_ms(lambda: store.embedder.embed_query('doctor'))}")
    timed("similarity_search k=10 (embed + search)", store.similarity_search, args.message, k=10)
    timed("detect_language (en text)", detect_language, args.message)
    lang = timed("detect_language (nl text)", detect_language, args.foreign)
    timed(f"translate {lang}->en", translate, from_lang=lang, to_lang="en", text=args.foreign)
    timed("get_system_prompt cold (Google Sheets)", get_system_prompt, args.sheet)
    timed("get_system_prompt warm (cache hit)", get_system_prompt, args.sheet)
    timed(
        "LLM baseline (short prompt, no tools)",
        rag_agent._get_llm().invoke, [HumanMessage("Reply with the single word: ok")],
    )

    print("\n== graph, per node (fresh thread) ==")
    thread = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread,
            "googleSheetId": args.sheet,
            "system_prompt": get_system_prompt(args.sheet),
        }
    }
    graph_turn("turn 1 (retrieval)", graph, args.message, config)
    graph_turn("turn 2 (follow-up)", graph, args.follow_up, config)
    graph_turn("turn 3 (small talk)", graph, "Thank you, that was helpful!", config)

    print("\n== end-to-end chat() (warm caches) ==")
    thread = str(uuid.uuid4())
    timed("chat() english turn", chat, thread, args.sheet, args.message)
    timed("chat() dutch turn (detect + 2x translate)", chat, thread, args.sheet, args.foreign)

    width = max(len(label) for label, _ in rows)
    print("\n== summary (ms) ==")
    for label, ms in rows:
        print(f"{label:<{width}}  {ms:8.0f}")

    rag_agent.close_rag_agent()


if __name__ == "__main__":
    main()
