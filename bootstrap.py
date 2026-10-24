"""Paired bootstrap confidence intervals for corpus-level WER/CER differences.

evaluate.py reports point estimates only. This resamples per-question edit
counts (rows paired by sample id, so each question contributes the same index
to both systems) to give a 95% interval for ``corpus_rate(B) - corpus_rate(A)``.
An interval containing 0 means the difference is not distinguishable from
sampling noise over this question set; it says nothing about speaker or
recording variation the question set does not contain.

Usage:
    python bootstrap.py --a outputs/evaluation/noisy_baseline/report.json \
        --b outputs/evaluation/noisy_denoised/report.json --metric wer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

FIELDS = {
    "wer": ("word_edits", "reference_words"),
    "cer": ("character_edits", "reference_characters"),
}


def paired_bootstrap(
    edits_a: Sequence[int], units_a: Sequence[int],
    edits_b: Sequence[int], units_b: Sequence[int],
    *, resamples: int = 10000, seed: int = 0, level: float = 0.95,
) -> dict[str, float]:
    """Point difference and percentile interval of corpus_rate(B) - corpus_rate(A)."""
    n = len(edits_a)
    if not n or not (n == len(units_a) == len(edits_b) == len(units_b)):
        raise ValueError("Systems must have the same nonzero number of paired rows")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")

    def diff(indices: Sequence[int]) -> float:
        rate_a = sum(edits_a[i] for i in indices) / sum(units_a[i] for i in indices)
        rate_b = sum(edits_b[i] for i in indices) / sum(units_b[i] for i in indices)
        return rate_b - rate_a

    rng = random.Random(seed)
    draws = sorted(diff([rng.randrange(n) for _ in range(n)]) for _ in range(resamples))
    tail = (1 - level) / 2
    return {
        "difference": diff(range(n)),
        "ci_low": draws[int(tail * resamples)],
        "ci_high": draws[min(resamples - 1, int((1 - tail) * resamples))],
        "samples": n,
    }


def load_rows(report_path: str | Path, metric: str) -> dict[str, tuple[int, int]]:
    edits_key, units_key = FIELDS[metric]
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {row["id"]: (row[edits_key], row[units_key]) for row in report["samples"]}


def compare_reports(path_a: str | Path, path_b: str | Path, metric: str, **kwargs) -> dict[str, float]:
    rows_a, rows_b = load_rows(path_a, metric), load_rows(path_b, metric)
    if rows_a.keys() != rows_b.keys():
        raise ValueError("Reports must cover the same sample ids")
    ids = sorted(rows_a)
    return paired_bootstrap(
        [rows_a[i][0] for i in ids], [rows_a[i][1] for i in ids],
        [rows_b[i][0] for i in ids], [rows_b[i][1] for i in ids], **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Reference-system report.json")
    parser.add_argument("--b", type=Path, required=True, help="Comparison-system report.json")
    parser.add_argument("--metric", choices=sorted(FIELDS), default="wer")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = compare_reports(args.a, args.b, args.metric, resamples=args.resamples, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
