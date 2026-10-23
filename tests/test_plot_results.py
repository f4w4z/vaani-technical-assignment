"""Headless plotting checks using synthetic evaluator-shaped reports."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from evaluate import save_report
from plot_results import create_plots, read_results


def example_report() -> dict:
    rows = [
        {
            "id": f"te-{i}", "wer": wer, "asr_response_quality": quality,
            "clean_response_quality": 0.9, "quality_drop": 0.9 - quality,
            "asr_seconds": 0.2, "llm_seconds": 0.5,
            "total_seconds": 0.75, "clean_llm_seconds": 0.45,
        }
        for i, (wer, quality) in enumerate([(0.0, 0.9), (0.4, 0.5), (1.4, 0.0)])
    ]
    return {
        "metadata": {"quality_metric": "word_overlap_f1"},
        "summary": {"samples_failed": 1}, "samples": rows, "failures": [],
    }


class PlotChecks(unittest.TestCase):
    def test_evaluator_json_and_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_report(example_report(), root)
            rows, metric, failed = read_results(root / "report.json")
            self.assertEqual((len(rows), metric, failed), (3, "word_overlap_f1", 1))
            for name, expected in [("report.json", 2), ("samples.csv", 2), ("wer_vs_quality.csv", 1)]:
                paths = create_plots(root / name, root / name.replace(".", "_"))
                self.assertEqual(len(paths), expected)
                for path in paths:
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertGreater(path.stat().st_size, 1000)

    def test_invalid_data_is_not_silently_plotted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "report.json"
            for field, value, message in [
                ("wer", -1, "nonnegative"),
                ("asr_response_quality", 2, "in \\[0, 1\\]"),
                ("total_seconds", "NaN", "nonfinite"),
                ("total_seconds", 0.1, "less than stage sum"),
                ("llm_seconds", None, "requires ASR"),
                ("clean_llm_seconds", None, "all samples or none"),
            ]:
                report = example_report()
                report["samples"][0][field] = value
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    create_plots(path, root / "plots")
            path.write_text('{"samples": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no successful"):
                read_results(path)

    def test_missing_scatter_pairs_and_optional_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "points.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["wer", "asr_response_quality"])
                writer.writeheader()
                writer.writerow({"wer": "", "asr_response_quality": 0.5})
            with self.assertRaisesRegex(ValueError, "No valid WER"):
                create_plots(path, root / "plots")


if __name__ == "__main__":
    unittest.main()
