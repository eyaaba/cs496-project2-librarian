
import json
from pathlib import Path

questions_path = Path("eval/questions.jsonl")
index_path = Path("data/index.json")

questions = [
    json.loads(line)
    for line in questions_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

chunks = json.loads(index_path.read_text(encoding="utf-8"))
valid_chunk_ids = {chunk["id"] for chunk in chunks}

required_fields = {
    "id", "question", "reference_answer", "gold_chunk_ids",
    "topic", "difficulty", "answerable", "split"
}

errors = []
seen_ids = set()

for q in questions:
    qid = q.get("id", "UNKNOWN")

    missing = required_fields - q.keys()
    if missing:
        errors.append(f"{qid}: Missing fields: {sorted(missing)}")

    if qid in seen_ids:
        errors.append(f"{qid}: Duplicate question ID")
    seen_ids.add(qid)

    if q.get("split") not in ("dev", "test"):
        errors.append(f"{qid}: Invalid split")

    if q.get("difficulty") not in ("easy", "medium", "hard"):
        errors.append(f"{qid}: Invalid difficulty")

    if not isinstance(q.get("answerable"), bool):
        errors.append(f"{qid}: answerable must be true or false")

    gold_ids = q.get("gold_chunk_ids", [])
    if not isinstance(gold_ids, list):
        errors.append(f"{qid}: gold_chunk_ids must be a list")
        continue

    if q.get("answerable") is True and not gold_ids:
        errors.append(f"{qid}: Answerable question has no gold evidence")

    for chunk_id in gold_ids:
        if chunk_id not in valid_chunk_ids:
            errors.append(f"{qid}: Unknown chunk ID: {chunk_id}")

print(f"Questions: {len(questions)}")
print(f"Index chunks: {len(chunks)}")
print(f"Errors: {len(errors)}")

for error in errors:
    print("ERROR:", error)
