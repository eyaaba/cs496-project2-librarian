
import json
from pathlib import Path

from src.rag import retrieve, retrieve_reranked

path = Path("eval/questions.jsonl")

questions = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and json.loads(line)["answerable"]
]

modes = ["bm25", "vector", "hybrid", "hybrid_rerank"]
results = {}

for mode in modes:
    hits = 0

    print(f"\n=== {mode.upper()} ===")

    for item in questions:
        if mode == "hybrid_rerank":
            passages = retrieve_reranked(
                item["question"],
                k=5,
                candidate_k=20
            )
        else:
            passages = retrieve(
                item["question"],
                k=5,
                mode=mode
            )

        retrieved_ids = {p["id"] for p in passages}
        gold_ids = set(item["gold_chunk_ids"])

        hit = bool(retrieved_ids & gold_ids)
        hits += int(hit)

        print(f"{item['id']}: {'HIT' if hit else 'MISS'}")

    results[mode] = hits / len(questions)
    print(f"Hit@5: {hits}/{len(questions)} = {results[mode]:.1%}")

print("\n=== SUMMARY ===")

for mode, score in results.items():
    print(f"{mode:15} {score:.1%}")
