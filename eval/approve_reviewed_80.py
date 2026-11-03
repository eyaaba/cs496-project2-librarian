"""Approve manually reviewed candidates 11-90 (4 excluded).
Run from the repository root: python eval/approve_reviewed_80.py
No OpenAI API calls. Safe to re-run; maintains a review manifest.
"""
import json
import re
from pathlib import Path

ROOT = Path('eval')
DATASET = ROOT / 'questions.jsonl'
CANDIDATES = ROOT / 'candidates_batch.jsonl'
INDEX = Path('data/index.json')
MANIFEST = ROOT / 'reviewed_candidates_11_90.json'

# Explicit exclusions after review of full source passages:
# 54: grammar excerpt doesn't establish claimed semantics of with.
# 68: source only contains a contents heading for doctest.
# 70: conflicting SQLite version updates in changelog.
# 79: near-duplicate of an already-approved timeit question.
REJECTED = {54, 68, 70, 79}

# Corrections remove ambiguities, incorrect claims and version confusion.
CORRECTIONS = {
    21: {'reference_answer': 'random.seed() initializes the pseudo-random number generator. If the seed is omitted or None, it uses operating-system randomness when available, or system time otherwise. An integer seed is used directly.'},
    24: {
        'question': 'What can the XML-RPC client proxy object described in the documentation be used for?',
        'reference_answer': 'Its methods can invoke the corresponding RPC calls on a remote server. If supported, it can also query introspection information.'
    },
    40: {'question': 'According to the Python 2.2 release notes, what changed about three-argument pow() with floating-point values?'},
    44: {
        'question': 'What replacement did the Python deprecation notes recommend for pathlib.Path.link_to()?',
        'reference_answer': 'Use pathlib.Path.hardlink_to() instead of pathlib.Path.link_to().'
    },
    49: {
        'reference_answer': 'The first syslog() call automatically calls openlog() if it has not been called already.'
    },
    51: {
        'reference_answer': 'It returns the JSON string \'{"__complex__": true, "real": 1.0, "imag": 2.0}\'. The custom_json function must be defined as in the source example.'
    },
    56: {'question': 'According to the Python deprecation notes, which re function was scheduled for removal in Python 3.13?'},
    62: {
        'reference_answer': 'It returns a list of address-information tuples suitable for TCP connections to example.org on port 80. The documented example contains both IPv6 and IPv4 results, but actual results depend on the system.'
    },
    69: {
        'question': 'What should an import finder set the ModuleSpec.name attribute to?',
        'reference_answer': 'A non-empty string giving the module fully qualified name.'
    },
    77: {
        'question': 'According to the historical Python release note, which file stores interactive interpreter history when readline support is enabled?'
    },
    83: {
        'reference_answer': 'An IndentationError is raised, with a message like "expected an indented block after if statement".'
    },
    89: {
        'question': 'Why are two recv loops needed when receiving a length-prefixed network message?',
        'reference_answer': 'One loop reads the complete length prefix, which may arrive in multiple pieces; the second loop reads the message payload until the stated length is received.'
    },
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def normalize(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


if not CANDIDATES.exists() or not DATASET.exists() or not INDEX.exists():
    raise SystemExit('Missing candidates, questions, or index file; run from repository root.')

candidates = load_jsonl(CANDIDATES)
questions = load_jsonl(DATASET)
if len(candidates) < 90:
    raise SystemExit(f'Expected at least 90 candidates; found {len(candidates)}. No changes made.')

valid_chunks = {chunk['id'] for chunk in json.loads(INDEX.read_text(encoding='utf-8'))}
existing = {normalize(q['question']) for q in questions}
next_id = 1 + max((int(q['id'][1:]) for q in questions), default=0)
new = []
skipped = []

for number in range(11, 91):
    if number in REJECTED:
        skipped.append([number, 'review rejected'])
        continue
    q = dict(candidates[number - 1])
    q.update(CORRECTIONS.get(number, {}))
    question = q['question'].strip()
    answer = q['reference_answer'].strip()
    chunk_id = q['source_chunk_id']
    difficulty = q['difficulty'].strip().lower()
    if chunk_id not in valid_chunks:
        raise SystemExit(f'Candidate {number} has unknown evidence {chunk_id}; no changes made.')
    if not question or not answer or difficulty not in {'easy', 'medium', 'hard'}:
        raise SystemExit(f'Candidate {number} has invalid fields; no changes made.')
    norm = normalize(question)
    if norm in existing:
        skipped.append([number, 'question already present'])
        continue
    existing.add(norm)
    new.append({
        'id': f'Q{next_id:03d}',
        'question': question,
        'reference_answer': answer,
        'gold_chunk_ids': [chunk_id],
        'topic': q.get('topic') or 'python',
        'difficulty': difficulty,
        'answerable': True,
        'split': 'dev' if next_id <= 20 else 'test',
    })
    next_id += 1

if new:
    with DATASET.open('a', encoding='utf-8') as file:
        for q in new:
            file.write(json.dumps(q, ensure_ascii=False) + '\n')

MANIFEST.write_text(json.dumps({
    'reviewed_candidate_numbers': list(range(11, 91)),
    'rejected': sorted(REJECTED),
    'corrected': sorted(CORRECTIONS),
    'approved_and_added': len(new),
    'skipped': skipped,
    'note': 'Evidence checked against user-provided full passage review. Structural validation is separate from held-out evaluation.',
}, indent=2), encoding='utf-8')

print(f'Added: {len(new)}')
print(f'Total questions: {len(questions) + len(new)}')
print(f'Skipped: {len(skipped)}')
print('Run: python eval/validate_dataset.py')
