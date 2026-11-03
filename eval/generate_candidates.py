
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INDEX = Path("data/index.json")
DATASET = Path("eval/questions.jsonl")
OUTPUT = Path("eval/candidates_batch.jsonl")
TARGET = 90

chunks = json.loads(INDEX.read_text(encoding="utf-8"))

existing = [
    json.loads(line)
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

used_ids = {
    cid
    for q in existing
    for cid in q.get("gold_chunk_ids", [])
}

previous = []
if OUTPUT.exists():
    previous = [
        json.loads(line)
        for line in OUTPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

used_ids.update(
    q["source_chunk_id"] for q in previous
)

existing_questions = {
    re.sub(r"\W+", " ", q["question"].lower()).strip()
    for q in existing + previous
}

eligible = [
    c for c in chunks
    if c["id"] not in used_ids
    and len(c["text"].split()) >= 100
]

random.Random(43).shuffle(eligible)

api = OpenAI()
added = len(previous)
failures = 0

with OUTPUT.open("a", encoding="utf-8") as out:
    for chunk in eligible:
        if added >= TARGET:
            break

        try:
            response = api.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate one source-grounded Python "
                            "documentation evaluation question. "
                            "Return only JSON containing question, "
                            "reference_answer, topic, difficulty. "
                            "The answer must be explicitly supported "
                            "by the supplied passage. "
                            "Difficulty must be easy, medium, or hard."
                        ),
                    },
                    {
                        "role": "user",
                        "content": chunk["text"],
                    },
                ],
            )

            item = json.loads(
                response.choices[0].message.content
            )

            question = item["question"].strip()
            answer = item["reference_answer"].strip()
            difficulty = item["difficulty"].strip().lower()
            normalized = re.sub(
                r"\W+", " ", question.lower()
            ).strip()

            if (
                not question
                or not answer
                or normalized in existing_questions
                or difficulty not in ("easy", "medium", "hard")
            ):
                continue

            item["difficulty"] = difficulty
            item["source_chunk_id"] = chunk["id"]
            item["source"] = chunk["source"]
            item["passage_excerpt"] = chunk["text"]

            out.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )
            out.flush()

            existing_questions.add(normalized)
            added += 1
            print(f"Generated {added}/{TARGET}", flush=True)

        except Exception as exc:
            failures += 1
            print(f"Skipped: {type(exc).__name__}: {exc}")

            if failures >= 5:
                print("Stopping after repeated errors.")
                break

print(f"Saved {added} candidates to {OUTPUT}")
