
import json
from pathlib import Path

dataset = Path("eval/questions.jsonl")
candidates_file = Path("eval/candidates_batch.jsonl")

questions = [
    json.loads(line)
    for line in dataset.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

candidates = [
    json.loads(line)
    for line in candidates_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# Only the first 10 candidates have been reviewed.
approved = {1, 2, 3, 4, 5, 6, 7, 8, 10}

corrections = {
    2: {
        "question": "What does inspect.get_annotations() do?",
    },
    4: {
        "question": "What is the data type of the faulthandler attribute in the Python configuration table?",
    },
    5: {
        "question": (
            "When Connection.autocommit is LEGACY_TRANSACTION_CONTROL, "
            "what happens if Connection.isolation_level is None?"
        ),
    },
}

existing_questions = {
    q["question"].strip().lower() for q in questions
}
next_id = max(int(q["id"][1:]) for q in questions) + 1

added = 0

with dataset.open("a", encoding="utf-8") as out:
    for number in sorted(approved):
        candidate = candidates[number - 1].copy()
        candidate.update(corrections.get(number, {}))

        question_text = candidate["question"].strip()

        if question_text.lower() in existing_questions:
            continue

        question = {
            "id": f"Q{next_id:03d}",
            "question": question_text,
            "reference_answer": candidate["reference_answer"],
            "gold_chunk_ids": [candidate["source_chunk_id"]],
            "topic": candidate["topic"],
            "difficulty": candidate["difficulty"].lower(),
            "answerable": True,
            "split": "dev" if next_id <= 20 else "test",
        }

        out.write(json.dumps(question, ensure_ascii=False) + "\n")

        existing_questions.add(question_text.lower())
        next_id += 1
        added += 1

print(f"Added: {added}")
print(f"Total: {len(questions) + added}")
