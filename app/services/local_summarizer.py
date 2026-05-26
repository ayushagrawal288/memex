"""
Extractive memory summariser — pure Python, zero dependencies, zero API calls.

Algorithm (modified Luhn frequency scoring):
  1. Split all memory strings into sentences
  2. Deduplicate sentences that share >= 70% vocabulary overlap — handles the
     common case where agents write near-identical memories across sessions
  3. Score each unique sentence by mean word frequency across the full corpus —
     sentences that use words appearing across many memories are more "central"
  4. Take the top-N by score, restored to original document order for coherence
  5. Join into a single paragraph

Why extractive rather than abstractive:
  - Abstractive needs an LLM (API call or heavy local model).
  - Extractive preserves exact wording of stored facts — no hallucination risk.
  - For memory consolidation, factual precision matters more than prose quality.
"""

import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def summarize(memories: list[str], max_sentences: int = 5) -> str:
    """
    Condense a list of raw memory strings into a compact paragraph.
    Returns a single string joining the top-N most representative sentences.
    """
    # Split each memory into sentences; drop very short fragments
    sentences: list[str] = []
    for mem in memories:
        parts = re.split(r"(?<=[.!?])\s+", mem.strip())
        sentences.extend(p.strip() for p in parts if len(p.strip()) > 20)

    if not sentences:
        return ". ".join(memories[:max_sentences])

    # Deduplicate: keep first occurrence; skip if >= 70% Jaccard overlap with any keeper
    tok_per_sent = [_tokens(s) for s in sentences]
    kept_indices: list[int] = []
    for i, toks in enumerate(tok_per_sent):
        if all(_jaccard(toks, tok_per_sent[j]) < 0.7 for j in kept_indices):
            kept_indices.append(i)

    unique = [(sentences[i], tok_per_sent[i]) for i in kept_indices]

    if len(unique) <= max_sentences:
        return " ".join(s for s, _ in unique)

    # Frequency-score: words that appear across many sentences are "central"
    freq: Counter = Counter(tok for _, toks in unique for tok in toks)

    def score(toks: list[str]) -> float:
        return sum(freq[t] for t in toks) / max(len(toks), 1)

    # Pick top-N, then restore original order for readability
    scored = sorted(enumerate(unique), key=lambda x: score(x[1][1]), reverse=True)
    top_indices = sorted(i for i, _ in scored[:max_sentences])
    return " ".join(unique[i][0] for i in top_indices)
