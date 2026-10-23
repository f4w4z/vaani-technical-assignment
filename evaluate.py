"""Offline evaluation for the Telugu speech pipeline.

Run ``python evaluate.py --dataset data/test.json --output-dir outputs/evaluation``.
Metric/report utilities use only the standard library. Audio duration uses
soundfile; model imports occur only in main(). See README.md for dataset format.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import platform
from statistics import mean
from typing import Any, Callable, Protocol, Sequence, TypeVar
import unicodedata

LOGGER = logging.getLogger(__name__)
Token = TypeVar("Token")
QualityScorer = Callable[[str, str], float]


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> tuple[str, float]: ...


class SpeechRunner(Protocol):
    """The interface implemented by pipeline.SpeechPipeline."""

    llm: LanguageModel

    def run(self, audio_path: str | Path) -> Any: ...


def normalize_text(text: str) -> str:
    """NFC-normalize Unicode and collapse whitespace; retain case/punctuation.

    Telugu vowel signs and combining marks are preserved. CER counts Unicode
    code points, not user-perceived grapheme clusters, and excludes whitespace.
    The same policy is applied to every reference and hypothesis.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def levenshtein_distance(reference: Sequence[Token], hypothesis: Sequence[Token]) -> int:
    """Compute unit-cost insertions, deletions, and substitutions from scratch.

    Two-row dynamic programming: O(N*M) time, O(min(N, M)) memory.
    Tokens may be words or individual Unicode code points.
    """
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, start=1):
        current = [i]
        for j, observed in enumerate(hypothesis, start=1):
            current.append(min(
                current[j - 1] + 1,             # insertion
                previous[j] + 1,                # deletion
                previous[j - 1] + (expected != observed),  # substitution
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ErrorRate:
    """Edit count and reference length; rate is undefined for an empty reference."""

    edits: int
    reference_units: int

    @property
    def rate(self) -> float | None:
        """Return a fraction (possibly greater than 1), or None if undefined."""
        return self.edits / self.reference_units if self.reference_units else None


def transcript_errors(reference: str, hypothesis: str) -> dict[str, ErrorRate]:
    """Return word and character edit counts using identical normalization."""
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    ref_chars, hyp_chars = "".join(ref_words), "".join(hyp_words)
    return {
        "wer": ErrorRate(levenshtein_distance(ref_words, hyp_words), len(ref_words)),
        "cer": ErrorRate(levenshtein_distance(ref_chars, hyp_chars), len(ref_chars)),
    }


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """WER = word edit distance / reference words; None for an empty reference."""
    return transcript_errors(reference, hypothesis)["wer"].rate


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """CER = code-point edit distance / reference code points, excluding spaces."""
    return transcript_errors(reference, hypothesis)["cer"].rate


def response_token_f1(reference: str, hypothesis: str) -> float:
    """Multiset word-overlap F1 in [0, 1]; this is a lexical quality proxy.

    Repeated words count only up to their reference frequency. Order is ignored.
    Both empty strings score 1; exactly one empty string scores 0. This does not
    measure factual correctness or credit semantically equivalent paraphrases.
    """
    expected = normalize_text(reference).split()
    observed = normalize_text(hypothesis).split()
    if not expected or not observed:
        return float(expected == observed)
    overlap = sum((Counter(expected) & Counter(observed)).values())
    return 2.0 * overlap / (len(expected) + len(observed))


def real_time_factor(total_seconds: float, audio_seconds: float) -> float:
    """End-to-end processing seconds divided by positive audio duration."""
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise ValueError("Processing time must be finite and nonnegative")
    if not math.isfinite(audio_seconds) or audio_seconds <= 0:
        raise ValueError("Audio duration must be finite and positive")
    return total_seconds / audio_seconds


def audio_duration(audio_path: str | Path) -> float:
    """Read actual duration from audio metadata without loading all samples."""
    import soundfile as sf

    info = sf.info(str(audio_path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"Empty or invalid audio: {audio_path}")
    return info.frames / info.samplerate


@dataclass(frozen=True)
class EvaluationSample:
    """One human-labeled evaluation item; the response is the scoring target."""

    id: str
    audio_path: Path
    ground_truth_transcript: str
    ground_truth_response: str


def load_dataset(dataset_path: str | Path) -> list[EvaluationSample]:
    """Load a nonempty JSON array; resolve audio paths from the JSON directory.

    Required keys: audio_path, ground_truth_transcript, ground_truth_response.
    IDs default to 1-based row numbers and must be unique. Empty references are
    rejected for paired evaluation, although standalone edit metrics allow them.
    Missing/unreadable audio is handled during evaluation, sample by sample.
    """
    path = Path(dataset_path).expanduser().resolve()
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array")
    samples: list[EvaluationSample] = []
    identifiers: set[str] = set()
    required = ("audio_path", "ground_truth_transcript", "ground_truth_response")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index} must be an object")
        for key in required:
            if not isinstance(row.get(key), str) or not normalize_text(row[key]):
                raise ValueError(f"Dataset row {index}: {key} must be a nonempty string")
        identifier = row.get("id", str(index))
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError(f"Dataset row {index}: id must be a unique nonempty string")
        identifiers.add(identifier)
        audio = Path(row["audio_path"]).expanduser()
        samples.append(EvaluationSample(
            identifier, (path.parent / audio).resolve(),
            row["ground_truth_transcript"], row["ground_truth_response"],
        ))
    return samples


def analyze_error_propagation(
    pipeline: SpeechRunner,
    sample: EvaluationSample,
    *,
    quality_scorer: QualityScorer = response_token_f1,
) -> dict[str, Any]:
    """Compare two generated answers against the SAME human ground truth.

    Noisy arm: audio -> ASR -> LLM. Clean arm: human transcript -> same LLM.
    Positive quality_drop means the ASR arm scored worse; negative means better.
    The clean-arm generated answer is a baseline, not the scoring ground truth.
    Use deterministic decoding and a stateless LLM to isolate the ASR effect.
    Custom scorers accept (reference, hypothesis) and return finite [0, 1] values.
    """
    duration = audio_duration(sample.audio_path)
    result = pipeline.run(sample.audio_path)
    clean_response, clean_llm_seconds = pipeline.llm.generate(sample.ground_truth_transcript)
    errors = transcript_errors(sample.ground_truth_transcript, result.transcript)
    noisy_quality = float(quality_scorer(sample.ground_truth_response, result.response))
    clean_quality = float(quality_scorer(sample.ground_truth_response, clean_response))
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in (noisy_quality, clean_quality)):
        raise ValueError("Quality scorer must return finite values between 0 and 1")
    timings = (result.asr_seconds, result.llm_seconds, result.total_seconds, clean_llm_seconds)
    if any(not math.isfinite(t) or t < 0 for t in timings):
        raise ValueError("Pipeline returned invalid timing values")
    return {
        "id": sample.id,
        "audio_path": str(sample.audio_path),
        "ground_truth_transcript": sample.ground_truth_transcript,
        "ground_truth_response": sample.ground_truth_response,
        "asr_transcript": result.transcript,
        "asr_response": result.response,
        "clean_transcript_response": clean_response,
        "wer": errors["wer"].rate,
        "word_edits": errors["wer"].edits,
        "reference_words": errors["wer"].reference_units,
        "cer": errors["cer"].rate,
        "character_edits": errors["cer"].edits,
        "reference_characters": errors["cer"].reference_units,
        "audio_seconds": duration,
        "asr_seconds": result.asr_seconds,
        "llm_seconds": result.llm_seconds,
        "total_seconds": result.total_seconds,
        "rtf": real_time_factor(result.total_seconds, duration),
        "asr_rtf": real_time_factor(result.asr_seconds, duration),
        "clean_llm_seconds": clean_llm_seconds,
        "asr_response_quality": noisy_quality,
        "clean_response_quality": clean_quality,
        "quality_drop": clean_quality - noisy_quality,
        "asr_response_exact_match": normalize_text(sample.ground_truth_response) == normalize_text(result.response),
        "clean_response_exact_match": normalize_text(sample.ground_truth_response) == normalize_text(clean_response),
        "response_agreement_f1": response_token_f1(clean_response, result.response),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    """Mean and linearly interpolated percentiles; nulls for no observations."""
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"mean": mean(values), "p50": percentile(0.5), "p95": percentile(0.95)}


def aggregate_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate completed pairs only, with corpus-weighted WER, CER, and RTF."""
    def ratio(numerator: str, denominator: str) -> float | None:
        divisor = sum(row[denominator] for row in rows)
        return sum(row[numerator] for row in rows) / divisor if divisor else None

    result: dict[str, Any] = {
        "corpus_wer": ratio("word_edits", "reference_words"),
        "corpus_cer": ratio("character_edits", "reference_characters"),
        "corpus_rtf": ratio("total_seconds", "audio_seconds"),
        "corpus_asr_rtf": ratio("asr_seconds", "audio_seconds"),
        "total_audio_seconds": sum(row["audio_seconds"] for row in rows),
        "total_pipeline_seconds": sum(row["total_seconds"] for row in rows),
        "total_clean_llm_seconds": sum(row["clean_llm_seconds"] for row in rows),
    }
    for key in (
        "wer", "cer", "asr_seconds", "llm_seconds", "total_seconds", "rtf",
        "clean_llm_seconds", "asr_response_quality", "clean_response_quality", "quality_drop",
    ):
        result[key] = _distribution([row[key] for row in rows if row[key] is not None])
    result["degraded_fraction"] = mean(row["quality_drop"] > 1e-12 for row in rows) if rows else None
    result["improved_fraction"] = mean(row["quality_drop"] < -1e-12 for row in rows) if rows else None
    for key in ("asr_response_exact_match", "clean_response_exact_match"):
        result[key + "_rate"] = mean(row[key] for row in rows) if rows else None
    return result


def evaluate_dataset(
    pipeline: SpeechRunner,
    samples: Sequence[EvaluationSample],
    *,
    quality_scorer: QualityScorer = response_token_f1,
    quality_name: str = "word_overlap_f1",
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Evaluate any dataset size sequentially, reusing the loaded model instances.

    Failed pairs are explicitly reported and excluded from ALL aggregate metrics.
    In particular, failure rate must be considered alongside quality/latency:
    successful-only averages can be optimistic. Model loading, metadata reads,
    scoring, and clean-arm generation are excluded from pipeline latency/RTF.
    """
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sample in enumerate(samples, start=1):
        LOGGER.info("Evaluating %s (%d/%d)", sample.id, index, len(samples))
        try:
            rows.append(analyze_error_propagation(pipeline, sample, quality_scorer=quality_scorer))
        except Exception as exc:
            if fail_fast:
                raise
            LOGGER.warning("Sample %s failed: %s", sample.id, exc)
            failures.append({
                "id": sample.id, "audio_path": str(sample.audio_path),
                "error_type": type(exc).__name__, "error": str(exc),
            })
    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "quality_metric": quality_name,
            "quality_range": [0, 1],
            "quality_drop_definition": "clean_response_quality - asr_response_quality",
            "normalization": "NFC; collapse whitespace; retain case/punctuation; CER excludes whitespace",
            "cer_unit": "Unicode code point (not grapheme cluster)",
            "aggregation_population": "successful complete pairs only",
            "rtf_definition": "pipeline total seconds / audio seconds; excludes clean arm",
        },
        "summary": {
            "samples_total": len(samples), "samples_succeeded": len(rows),
            "samples_failed": len(failures),
            "failure_rate": len(failures) / len(samples) if samples else None,
            **aggregate_results(rows),
        },
        "samples": rows,
        "failures": failures,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Save full JSON, sample CSV, failure CSV, and ready-to-plot paired points.

    Plot CSV columns include WER (x), noisy/clean quality (y), and quality drop.
    Unicode CSVs use a BOM for spreadsheet compatibility; JSON is strict UTF-8.
    """
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = report["samples"]
    fields = list(rows[0]) if rows else ["id", "audio_path", "wer", "cer", "rtf", "quality_drop"]
    _write_csv(directory / "samples.csv", rows, fields)
    _write_csv(directory / "failures.csv", report["failures"], ["id", "audio_path", "error_type", "error"])
    _write_csv(directory / "wer_vs_quality.csv", rows, [
        "id", "wer", "asr_response_quality", "clean_response_quality", "quality_drop",
    ])


def main() -> None:
    """Load models once, optionally warm up, evaluate the dataset, and save reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--warmup", type=int, default=0, help="Unmeasured paired runs on the first sample")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first sample failure")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Keep metric imports lightweight and usable without torch/transformers.
    from pipeline import SpeechPipeline, load_config

    config = load_config(args.config)
    samples = load_dataset(args.dataset)
    if config.get("llm", {}).get("generation", {}).get("do_sample", False):
        parser.error("Set llm.generation.do_sample to false for paired error-propagation evaluation")
    pipeline = SpeechPipeline.from_config(config)
    for _ in range(args.warmup):
        analyze_error_propagation(pipeline, samples[0])
    report = evaluate_dataset(pipeline, samples, fail_fast=args.fail_fast)
    report["metadata"].update({
        "config": config, "dataset": str(args.dataset.resolve()),
        "warmup_runs": args.warmup,
    })
    save_report(report, args.output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    LOGGER.info("Reports saved to %s", args.output_dir.resolve())
    if report["failures"]:
        raise SystemExit(1)  # Partial reports are still saved before failure exit.


if __name__ == "__main__":
    main()
