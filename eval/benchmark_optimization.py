
import json
import statistics
import time
from pathlib import Path

from src.rag import (
    load_reranker,
    retrieve,
    retrieve_reranked,
)

questions = [
    json.loads(line)
    for line in Path("eval/questions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip() and json.loads(line)["split"] == "test"
]

# Use the same 20 test questions for both modes.
questions = questions[:80]

print("Loading reranker...")
load_reranker()

# Warm up both retrieval paths.
warmup = questions[0]["question"]
retrieve(warmup, mode="hybrid")
retrieve_reranked(warmup)

results = {}

for mode in ["hybrid_rerank", "hybrid"]:
    durations = []
    hits = 0

    for item in questions:
        start = time.perf_counter()

        if mode == "hybrid_rerank":
            passages = retrieve_reranked(
                item["question"], k=5, candidate_k=20
            )
        else:
            passages = retrieve(
                item["question"], k=5, mode="hybrid"
            )

        durations.append(
            (time.perf_counter() - start) * 1000
        )

        found = bool(
            {p["id"] for p in passages}
            & set(item["gold_chunk_ids"])
        )
        hits += int(found)

    durations.sort()
    results[mode] = {
        "p50": statistics.median(durations),
        "p95": durations[
            int(0.95 * (len(durations) - 1))
        ],
        "hits": hits,
    }

    print(
        f"{mode}: Hit@5={hits}/{len(questions)}, "
        f"p50={results[mode]['p50']:.1f}ms, "
        f"p95={results[mode]['p95']:.1f}ms"
    )

before = results["hybrid_rerank"]["p95"]
after = results["hybrid"]["p95"]

improvement = (before - after) / before * 100

print(f"\np95 reduction: {improvement:.1f}%")
