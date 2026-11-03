
import json
from pathlib import Path

questions_path = Path("eval/questions.jsonl")
candidates_path = Path("eval/candidates.jsonl")

existing = [
    json.loads(line)
    for line in questions_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

candidates = [
    json.loads(line)
    for line in candidates_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

existing_sources = {
    (q["question"].strip().lower(), tuple(q["gold_chunk_ids"]))
    for q in existing
}

next_id = max(int(q["id"][1:]) for q in existing) + 1
added = 0

with questions_path.open("a", encoding="utf-8") as out:
    for candidate in candidates:
        key = (
            candidate["question"].strip().lower(),
            (candidate["source_chunk_id"],),
        )

        if key in existing_sources:
            continue

        question = {
            "id": f"Q{next_id:03d}",
            "question": candidate["question"],
            "reference_answer": candidate["reference_answer"],
            "gold_chunk_ids": [candidate["source_chunk_id"]],
            "topic": candidate["topic"],
            "difficulty": candidate["difficulty"],
            "answerable": True,
            "split": "dev" if next_id <= 20 else "test",
        }

        out.write(json.dumps(question, ensure_ascii=False) + "\n")

        existing_sources.add(key)
        next_id += 1
        added += 1

print(f"Added {added} questions")
print(f"Total questions: {len(existing) + added}")
