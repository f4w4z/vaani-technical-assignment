"""Preparation integration checks with tiny generated WAV files, no models."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import wave

from evaluate import load_dataset
from prepare_data import PLACEHOLDER_RESPONSE, prepare_dataset


def write_audio(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)


class PreparationChecks(unittest.TestCase):
    def test_folder_round_trip_and_subset_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "corpus"
            folder.mkdir()
            for i in range(5):
                write_audio(folder / f"{i}.wav")
                (folder / f"{i}.txt").write_text("  తెలంగాణ రాజధాని ఏది?\n", encoding="utf-8")
                (folder / f"{i}.response.txt").write_text("హైదరాబాద్.", encoding="utf-8")
            output = root / "subset" / "test.json"
            metadata = prepare_dataset(folder, output, limit=3, seed=7)
            samples = load_dataset(output)
            self.assertEqual(len(samples), 3)
            self.assertEqual(metadata["placeholder_responses"], 0)
            self.assertTrue(all(sample.audio_path.is_file() for sample in samples))
            self.assertEqual(samples[0].ground_truth_transcript, "తెలంగాణ రాజధాని ఏది?")
            original = output.read_text(encoding="utf-8")
            prepare_dataset(folder, output, limit=3, seed=7, overwrite=True)
            self.assertEqual(output.read_text(encoding="utf-8"), original)
            self.assertTrue(output.with_suffix(".metadata.json").is_file())
            with self.assertRaises(FileExistsError):
                prepare_dataset(folder, output)

    def test_manifest_mapping_and_explicit_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_audio(root / "one.wav")
            manifest = root / "labels.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "text"])
                writer.writeheader()
                writer.writerow({"path": "one.wav", "text": "నమస్కారం"})
            output = root / "prepared" / "test.json"
            with self.assertRaisesRegex(ValueError, "missing reference response"):
                prepare_dataset(manifest, output, audio_column="path", transcript_column="text")
            self.assertFalse(output.exists())
            metadata = prepare_dataset(
                manifest, output, audio_column="path", transcript_column="text",
                response_mode="placeholder", limit=0,
            )
            self.assertTrue(metadata["response_quality_is_test_only"])
            self.assertEqual(metadata["placeholder_responses"], 1)
            sample = load_dataset(output)[0]
            self.assertEqual(sample.ground_truth_response, PLACEHOLDER_RESPONSE)
            self.assertEqual(sample.audio_path, root / "one.wav")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["response_source"], "placeholder")

    def test_invalid_sources_and_duplicate_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_audio(root / "one.wav")
            manifest = root / "labels.json"
            row = {"audio_path": "one.wav", "ground_truth_transcript": "నమస్కారం", "ground_truth_response": "నమస్తే"}
            output = root / "test.json"
            for rows, expected in [
                ([row, row], "Duplicate audio"),
                ([dict(row, audio_path="missing.wav")], "audio not found"),
                ([dict(row, ground_truth_transcript=" ")], "reference transcript"),
                ([], "No source samples"),
            ]:
                manifest.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
                    prepare_dataset(manifest, output, limit=0)
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "overwrite the source"):
                prepare_dataset(manifest, manifest, overwrite=True)


if __name__ == "__main__":
    unittest.main()
