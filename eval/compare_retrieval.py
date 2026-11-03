
import json
import time
from pathlib import Path

from src.rag import retrieve, retrieve_reranked

questions = [
    json.loads(line)
    for line in Path("eval/questions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

modes = ["bm25", "vector", "hybrid", "hybrid_rerank"]

print(f"Evaluating {len(questions)} questions...\n")

for mode in modes:
    hits = {"dev": 0, "test": 0}
    totals = {"dev": 0, "test": 0}
    durations = []

    for question in questions:
        start = time.perf_counter()

        if mode == "hybrid_rerank":
            passages = retrieve_reranked(
                question["question"], k=5, candidate_k=20
            )
        else:
            passages = retrieve(
                question["question"], k=5, mode=mode
            )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        durations.append(elapsed_ms)

        split = question["split"]
        totals[split] += 1

        found = bool(
            set(p["id"] for p in passages)
            & set(question["gold_chunk_ids"])
        )

        hits[split] += int(found)

    total_hits = sum(hits.values())
    total_questions = sum(totals.values())

    durations.sort()
    n = len(durations)

    p50 = durations[int(0.50 * (n - 1))]
    p95 = durations[int(0.95 * (n - 1))]

    print(f"MODE: {mode}")
    print(
        f"  Overall Hit@5: "
        f"{total_hits}/{total_questions} "
        f"= {total_hits / total_questions:.1%}"
    )
    print(
        f"  Dev Hit@5: "
        f"{hits['dev']}/{totals['dev']}"
    )
    print(
        f"  Test Hit@5: "
        f"{hits['test']}/{totals['test']}"
    )
    print(f"  p50 latency: {p50:.1f} ms")
    print(f"  p95 latency: {p95:.1f} ms")
    print()
