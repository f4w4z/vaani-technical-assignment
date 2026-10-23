"""Offline evaluation checks using deterministic substitute models."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluate import (
    EvaluationSample,
    aggregate_results,
    analyze_error_propagation,
    character_error_rate,
    evaluate_dataset,
    levenshtein_distance,
    load_dataset,
    normalize_text,
    real_time_factor,
    response_token_f1,
    save_report,
    transcript_errors,
    word_error_rate,
)


class EvaluationChecks(unittest.TestCase):
    def test_edit_distance_and_unicode_policy(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance([], ["a", "b"]), 2)
        self.assertEqual(levenshtein_distance(["a", "b"], []), 2)
        self.assertAlmostEqual(word_error_rate("a b c", "a d c x"), 2 / 3)
        self.assertEqual(word_error_rate("a", "a b c"), 2)
        self.assertEqual(word_error_rate("a", ""), 1)
        self.assertIsNone(word_error_rate("", "inserted"))
        self.assertIsNone(character_error_rate("", ""))
        self.assertEqual(transcript_errors("", "a b")["wer"].edits, 2)
        self.assertEqual(character_error_rate("కా", "క"), 0.5)
        self.assertEqual(character_error_rate("కా  క", "కాక"), 0)
        self.assertEqual(character_error_rate("e\u0301", "é"), 0)
        self.assertEqual(normalize_text("  నమస్కారం\n ప్రపంచం  "), "నమస్కారం ప్రపంచం")

    def test_quality_and_rtf(self) -> None:
        self.assertAlmostEqual(response_token_f1("a a b", "a b b"), 2 / 3)
        self.assertEqual(response_token_f1("", ""), 1)
        self.assertEqual(response_token_f1("a", ""), 0)
        self.assertEqual(real_time_factor(3, 2), 1.5)
        for processing, duration in [(0, 0), (-1, 2), (1, float("inf"))]:
            with self.assertRaises(ValueError):
                real_time_factor(processing, duration)

    def test_paired_comparison_and_corpus_weighting(self) -> None:
        pipeline = Mock()
        pipeline.run.return_value = SimpleNamespace(
            transcript="a x", response="wrong", asr_seconds=0.2,
            llm_seconds=0.3, total_seconds=0.6,
        )
        pipeline.llm.generate.return_value = ("correct", 0.4)
        sample = EvaluationSample("one", Path("one.wav"), "a b", "correct")
        with patch("evaluate.audio_duration", return_value=2):
            row = analyze_error_propagation(pipeline, sample)
        pipeline.llm.generate.assert_called_once_with("a b")
        self.assertEqual(row["quality_drop"], 1)
        self.assertEqual(row["wer"], 0.5)
        self.assertEqual(row["rtf"], 0.3)  # Clean LLM time is excluded.
        second = dict(row, word_edits=0, reference_words=8, wer=0,
                      total_seconds=0.4, audio_seconds=8, rtf=0.05)
        summary = aggregate_results([row, second])
        self.assertEqual(summary["corpus_wer"], 0.1)
        self.assertEqual(summary["wer"]["mean"], 0.25)
        self.assertEqual(summary["corpus_rtf"], 0.1)
        self.assertEqual(summary["rtf"]["mean"], 0.175)
        with patch("evaluate.audio_duration", return_value=2):
            pipeline.run.return_value.response = "correct"
            pipeline.llm.generate.return_value = ("wrong", 0.4)
            self.assertEqual(analyze_error_propagation(pipeline, sample)["quality_drop"], -1)
            with self.assertRaisesRegex(ValueError, "Quality scorer"):
                analyze_error_propagation(pipeline, sample, quality_scorer=lambda r, h: float("nan"))

    def test_batch_reporting_and_failures(self) -> None:
        samples = [EvaluationSample(str(i), Path(f"{i}.wav"), "a", "yes") for i in range(101)]
        pipeline = Mock()
        pipeline.run.return_value = SimpleNamespace(
            transcript="a", response="yes", asr_seconds=0.1,
            llm_seconds=0.2, total_seconds=0.3,
        )
        pipeline.llm.generate.return_value = ("yes", 0.2)
        with patch("evaluate.audio_duration", return_value=1):
            report = evaluate_dataset(pipeline, samples)
        self.assertEqual(report["summary"]["samples_succeeded"], 101)
        self.assertEqual(report["summary"]["corpus_wer"], 0)
        self.assertEqual(report["summary"]["asr_response_quality"]["mean"], 1)
        with tempfile.TemporaryDirectory() as directory:
            save_report(report, directory)
            restored = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(restored["samples"]), 101)
            self.assertEqual(len((Path(directory) / "wer_vs_quality.csv").read_text(encoding="utf-8-sig").splitlines()), 102)
            self.assertTrue((Path(directory) / "failures.csv").is_file())
        pipeline.run.side_effect = RuntimeError("ASR failure")
        with patch("evaluate.audio_duration", return_value=1), self.assertLogs("evaluate", level="WARNING"):
            failed = evaluate_dataset(pipeline, samples[:1])
        self.assertEqual(failed["summary"]["failure_rate"], 1)
        self.assertIsNone(failed["summary"]["corpus_wer"])
        self.assertEqual(failed["failures"][0]["error_type"], "RuntimeError")
        with patch("evaluate.audio_duration", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "ASR failure"):
                evaluate_dataset(pipeline, samples[:1], fail_fast=True)

    def test_dataset_paths_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            row = {
                "id": "te-001", "audio_path": "audio/one.wav",
                "ground_truth_transcript": "నమస్కారం", "ground_truth_response": "నమస్తే",
            }
            path.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")
            sample = load_dataset(path)[0]
            self.assertEqual(sample.audio_path, Path(directory) / "audio" / "one.wav")
            self.assertEqual(sample.ground_truth_response, "నమస్తే")
            for invalid in [[], {}, [row, row], [dict(row, ground_truth_response=" ")]]:
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset(path)


if __name__ == "__main__":
    unittest.main()
