
import json
from pathlib import Path

from src.rag import retrieve

dataset = Path("eval/questions.jsonl")

questions = [
    json.loads(line)
    for line in dataset.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

hits = 0

for item in questions:
    retrieved = retrieve(
        item["question"],
        k=5,
        mode="hybrid"
    )

    retrieved_ids = {p["id"] for p in retrieved}
    gold_ids = set(item["gold_chunk_ids"])

    found = bool(retrieved_ids & gold_ids)

    if found:
        hits += 1

    print(
        f"{item['id']}: "
        f"{'HIT' if found else 'MISS'}"
    )

print()
print(f"Recall@5 (question-level hit rate): {hits}/{len(questions)} = {hits/len(questions):.1%}")
