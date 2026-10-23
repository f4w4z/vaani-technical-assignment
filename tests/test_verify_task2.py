"""Verification checks: arithmetic proof, failed-readiness reporting, and config guards."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from verify_task2 import (
    Check,
    ROOT,
    check_dependencies,
    check_directories,
    markdown_summary,
    run_mock_evaluation,
    validate_config,
    verify_metric_primitives,
    verify_task2,
)


class VerificationChecks(unittest.TestCase):
    def test_three_real_evaluator_pairs_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with patch("asr.AutoModelForSpeechSeq2Seq.from_pretrained", side_effect=AssertionError("No model downloads")), patch(
                "llm.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("No model downloads")
            ):
                report = run_mock_evaluation(work)
            self.assertEqual(report["summary"]["samples_succeeded"], 3)
            self.assertAlmostEqual(report["summary"]["corpus_wer"], 3 / 7)
            self.assertAlmostEqual(report["summary"]["corpus_cer"], 5 / 14)
            self.assertAlmostEqual(report["summary"]["corpus_rtf"], 0.3)
            self.assertTrue(report["metadata"]["mock"])
            self.assertTrue((work / "samples.csv").is_file())
            self.assertEqual(len(json.loads((work / "test.json").read_text(encoding="utf-8"))), 3)
            markdown = markdown_summary([Check("Dependency | test", "FAIL", "wrong\nversion")], report)
            self.assertIn("**NOT READY**", markdown)
            self.assertIn("Dependency \\| test", markdown)
            self.assertIn("not research accuracy or performance results", markdown)
            self.assertIn("| substitution | 0.5000 | 0.2500", markdown)
            self.assertIn("| Corpus CER (5/14) | 0.3571 |", markdown)

    def test_config_guards(self) -> None:
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        validate_config(config)
        for section, key, value in [
            ("runtime", "device", "cuda:invalid"), ("runtime", "dtype", "int8"),
            ("asr", "batch_size", True), ("llm", "max_input_tokens", 0),
            ("asr", "language", "english"), ("paths", "output", ""),
        ]:
            invalid = copy.deepcopy(config)
            invalid[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(invalid)
        for generation in ({"do_sample": True}, {"max_new_tokens": 1.5}, {"top_p": 1.5}):
            invalid = copy.deepcopy(config)
            invalid["llm"]["generation"] = generation
            with self.assertRaises(ValueError):
                validate_config(invalid)

    def test_dependency_version_constraints_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text("numpy>=1,<2\n", encoding="utf-8")
            with patch("verify_task2.metadata.version", return_value="2.0"), patch("verify_task2.importlib.import_module"):
                checks = check_dependencies(requirements)
            self.assertEqual(checks[-1].status, "FAIL")
            self.assertIn("Installed 2.0", checks[-1].detail)
            with patch("verify_task2.metadata.version", return_value="1.26"), patch("verify_task2.importlib.import_module"):
                self.assertEqual(check_dependencies(requirements)[-1].status, "PASS")

    def test_directory_failures_and_independent_mock_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "a_file"
            blocked.write_text("not a directory", encoding="utf-8")
            checks = check_directories(None, root / "config.yaml", blocked)
            self.assertEqual(checks[0].status, "FAIL")
            with patch("verify_task2.check_repository", return_value=[]), patch(
                "verify_task2.check_dependencies", return_value=[Check("Dependency", "FAIL", "Missing")]
            ), patch("verify_task2.check_configuration", return_value=(None, [])), patch(
                "verify_task2.check_directories", return_value=[]
            ), patch("verify_task2.run_mock_evaluation", return_value={"mock": True}) as run:
                checks, report = verify_task2(root / "config.yaml", root / "out")
            run.assert_called_once()
            self.assertTrue(report["mock"])
            self.assertEqual(checks[0].status, "FAIL")
            self.assertEqual(checks[-1].status, "PASS")

    def test_wrong_metric_results_raise_instead_of_passing(self) -> None:
        verify_metric_primitives()
        with patch("evaluate.word_error_rate", return_value=0):
            with self.assertRaisesRegex(AssertionError, "WER insertions"):
                verify_metric_primitives()
        self.assertIn("results unavailable", markdown_summary([], None))


if __name__ == "__main__":
    unittest.main()
