"""Correctness and hallucination rates for the template Q&A corpus.

Task 2 asks for "response correctness and hallucination rate using clearly
defined evaluation criteria". Every reference answer in the synthesized corpus
(synthesize_corpus.py) states exactly one checkable fact: a state capital, or
a sum spoken as a Telugu number word. That makes a stricter test than word-
overlap F1 possible, computed post hoc from evaluate.py's report.json files
(no model inference):

* key fact: the capital name (audio file ``question_cap-*``) or the sum word
  (``question_sum-*``), taken from the human reference answer.
* correct: the normalized response contains the key fact as a substring (for
  sums, the digit form such as "5" also counts, as a whole number).
* abstained: not correct, but the response contains an uncertainty marker.
* hallucinated: neither correct nor abstained, i.e. a non-answer or wrong
  answer asserted without hedging, despite the system prompt's "ask for
  clarification, do not invent facts" instruction.

This is a lexical heuristic, not a semantic judge: a correct answer phrased
with a different inflection of the key fact is miscounted as hallucinated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Sequence

from evaluate import normalize_text
from synthesize_corpus import NUMBER_WORDS

# "I don't know", "sorry", "not clear" -- honest non-answers, not hallucinations.
UNCERTAINTY_MARKERS = ("తెలియదు", "క్షమించండి", "స్పష్టంగా లేదు")
SUM_SUFFIX = "వస్తుంది"


def extract_key_fact(audio_path: str, ground_truth_response: str) -> str:
    """Return the single checkable fact in a template reference answer."""
    words = normalize_text(ground_truth_response).rstrip(".").split()
    stem = Path(audio_path.replace("\\", "/")).name
    if stem.startswith("question_cap-"):
        return words[-1]
    if stem.startswith("question_sum-"):
        if len(words) < 2 or words[-1] != SUM_SUFFIX:
            raise ValueError(f"Unexpected sum answer template: {ground_truth_response!r}")
        return words[-2]
    raise ValueError(f"No key-fact rule for audio file {stem!r}")


def accepted_forms(audio_path: str, key_fact: str) -> tuple[str, ...]:
    """Surface forms that count as stating the fact: sums also accept digits."""
    forms = (normalize_text(key_fact),)
    if Path(audio_path.replace("\\", "/")).name.startswith("question_sum-"):
        value = {word: number for number, word in NUMBER_WORDS.items()}.get(forms[0])
        if value is not None:
            forms += (str(value),)
    return forms


def classify_response(response: str, key_fact: str | Sequence[str]) -> str:
    """Label a response ``correct``, ``abstained``, or ``hallucinated``.

    Digit-only forms must match a whole number ("5" must not match inside "15").
    """
    text = normalize_text(response)
    forms = (key_fact,) if isinstance(key_fact, str) else tuple(key_fact)
    for form in map(normalize_text, forms):
        if form.isdigit():
            if re.search(rf"(?<!\d){form}(?!\d)", text):
                return "correct"
        elif form in text:
            return "correct"
    if any(marker in text for marker in UNCERTAINTY_MARKERS):
        return "abstained"
    return "hallucinated"


def fact_rates(samples: Sequence[dict[str, Any]], response_field: str) -> dict[str, Any]:
    """Count labels for one response field of a report's per-sample rows."""
    counts = {"correct": 0, "abstained": 0, "hallucinated": 0}
    for row in samples:
        fact = extract_key_fact(row["audio_path"], row["ground_truth_response"])
        forms = accepted_forms(row["audio_path"], fact)
        counts[classify_response(row[response_field], forms)] += 1
    total = sum(counts.values())
    return {
        "samples": total,
        **counts,
        "correct_rate": counts["correct"] / total if total else None,
        "hallucination_rate": counts["hallucinated"] / total if total else None,
    }


def analyze_report(report_path: str | Path) -> dict[str, Any]:
    """Rates for the ASR->LLM answers and the human-transcript->LLM answers."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    samples = report["samples"]
    return {
        "asr_transcript": fact_rates(samples, "asr_response"),
        "human_transcript": fact_rates(samples, "clean_transcript_response"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="evaluate.py report.json files; the parent folder names each condition")
    parser.add_argument("--output", type=Path, help="Optional UTF-8 JSON destination")
    args = parser.parse_args()
    results = {path.parent.name: analyze_report(path) for path in args.reports}
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
