
import csv
import json
import time
from pathlib import Path

from src.rag import answer, validate_citations, extract_citations
dataset = [
    json.loads(line)
    for line in Path("eval/questions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

# Small initial sample to control API usage.
supported = [q for q in dataset if q["split"] == "test"][:10]

unsupported = [
    "What was the population of Tunisia in 2025?",
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Australia?",
    "What is the recommended treatment for fictional Zorvax syndrome?",
    "What was Apple's total revenue in 2024?",
]


def check_answer_citations(response, passages):
    valid_ids, invalid_ids = validate_citations(response, passages)

    is_refusal = (
        response.strip().lower()
        == "i don't know based on the provided documents."
    )

    has_citations = bool(extract_citations(response))

    is_error = response.startswith("I couldn't")

    return (
        valid_ids
        and not invalid_ids
        and (is_refusal or (has_citations and not is_error))
    )


rows = []

for item in supported:
    question = item["question"]
    start = time.perf_counter()
    response, passages = answer(question, mode="hybrid")
    latency = (time.perf_counter() - start) * 1000

    valid = check_answer_citations(response, passages)

    rows.append({
        "id": item["id"],
        "question": question,
        "reference_answer": item["reference_answer"],
        "generated_answer": response,
        "expected_answerable": True,
        "citation_ids_valid": valid,
        "cited_passage_ids": "; ".join(
            p["id"] for p in passages
        ),
        "latency_ms": round(latency, 1),
        "human_correct": "",
        "human_faithful": "",
        "human_citations_supported": "",
    })

    print(f"Evaluated {item['id']}", flush=True)

for i, question in enumerate(unsupported, 1):
    start = time.perf_counter()
    response, passages = answer(question, mode="hybrid")
    latency = (time.perf_counter() - start) * 1000


    valid = check_answer_citations(response, passages)
    rows.append({
        "id": f"U{i:03d}",
        "question": question,
        "reference_answer": "I don't know based on the provided documents.",
        "generated_answer": response,
        "expected_answerable": False,
        "citation_ids_valid": valid,
        "cited_passage_ids": "; ".join(
            p["id"] for p in passages
        ),
        "latency_ms": round(latency, 1),
        "human_correct": "",
        "human_faithful": "",
        "human_citations_supported": "",
    })

    print(f"Evaluated U{i:03d}", flush=True)

output = Path("eval/answer_review.csv")

with output.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} results to {output}")
