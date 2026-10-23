"""Prepare a local Telugu evaluation subset without downloading a corpus.

Accepts an audio folder with matching .txt transcripts / .response.txt answers,
or a CSV/TSV/JSON manifest with configurable field names. Outputs test.json in
evaluate.py's schema, plus provenance in test.metadata.json. See README.md.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Literal

from evaluate import audio_duration, normalize_text

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}
PLACEHOLDER_RESPONSE = "పరీక్ష కోసం తాత్కాలిక సమాధానం."


def read_source(source: Path) -> list[dict[str, Any]]:
    """Read manifest rows or discover folder audio in a stable path order.

    Manifest audio paths resolve from the manifest's directory. Directory rows
    defer reading sidecars until sampling, avoiding reading an entire corpus.
    """
    if source.is_dir():
        return [
            {"audio_path": str(path)}
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
    if source.suffix.lower() == ".json":
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
    elif source.suffix.lower() in {".csv", ".tsv"}:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t" if source.suffix.lower() == ".tsv" else ","))
    else:
        raise ValueError("Source must be an audio directory or a CSV/TSV/JSON manifest")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Manifest must contain an array of objects or tabular rows")
    return rows


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def prepare_dataset(
    source: str | Path,
    output: str | Path = "data/test.json",
    *,
    limit: int = 100,
    seed: int = 42,
    response_mode: Literal["require", "placeholder"] = "require",
    audio_column: str = "audio_path",
    transcript_column: str = "ground_truth_transcript",
    response_column: str = "ground_truth_response",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select a reproducible subset, validate audio/labels, and save UTF-8 JSON.

    ``limit=0`` selects all rows. A fixed seed reproduces the selection for an
    unchanged source listing. Audio remains in its source location; JSON paths
    are relative to the output directory (absolute if Windows drives differ).
    Validation errors stop preparation instead of silently dropping samples.
    """
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    if limit < 0:
        raise ValueError("limit must be nonnegative (0 means all samples)")
    if response_mode not in {"require", "placeholder"}:
        raise ValueError("response_mode must be require or placeholder")
    if output_path.suffix.lower() != ".json":
        raise ValueError("Output must be a .json file")
    if source_path in {output_path, metadata_path}:
        raise ValueError("Output cannot overwrite the source manifest")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError("Output or metadata exists; choose another path or use --overwrite")
    candidates = read_source(source_path)
    if not candidates:
        raise ValueError("No source samples found")
    count = min(limit, len(candidates)) if limit else len(candidates)
    indices = sorted(random.Random(seed).sample(range(len(candidates)), count))
    base = source_path if source_path.is_dir() else source_path.parent
    prepared: list[dict[str, str]] = []
    identifiers: set[str] = set()
    audio_paths: set[Path] = set()
    placeholder_count = 0
    for index in indices:
        row = candidates[index]
        sidecars = source_path.is_dir()
        audio_value = row.get("audio_path" if sidecars else audio_column)
        if not isinstance(audio_value, str) or not audio_value.strip():
            raise ValueError(f"Source row {index + 1}: missing audio path")
        audio = (base / Path(audio_value).expanduser()).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"Source row {index + 1}: audio not found: {audio}")
        # Header inspection catches empty/undecodable files before expensive inference.
        audio_duration(audio)
        if sidecars:
            transcript = _read_text(audio.with_suffix(".txt"))
            response_path = audio.with_suffix(".response.txt")
            response = _read_text(response_path) if response_path.is_file() else None
        else:
            transcript = row.get(transcript_column)
            response = row.get(response_column)
        if not isinstance(transcript, str) or not normalize_text(transcript):
            raise ValueError(f"Source row {index + 1}: missing/empty reference transcript")
        # Do not coerce corrupt numeric/list labels into strings or placeholders.
        if response is not None and not isinstance(response, str):
            raise ValueError(f"Source row {index + 1}: response must be text")
        response_source = "provided"
        if response is None or not normalize_text(response):
            if response_mode == "require":
                raise ValueError(
                    f"Source row {index + 1}: missing reference response; "
                    "supply a human label or explicitly select placeholder mode"
                )
            response = PLACEHOLDER_RESPONSE
            response_source = "placeholder"
            placeholder_count += 1
        identifier = row.get("id")
        if identifier is None or identifier == "":
            identifier = f"te-{index + 1:06d}"
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError(f"Source row {index + 1}: duplicate/invalid id: {identifier!r}")
        if audio in audio_paths:
            raise ValueError(f"Duplicate audio in selected subset: {audio}")
        identifiers.add(identifier)
        audio_paths.add(audio)
        try:
            stored_path = Path(os.path.relpath(audio, output_path.parent)).as_posix()
        except ValueError:  # Relative paths cannot span Windows drive letters.
            stored_path = audio.as_posix()
        prepared.append({
            "id": identifier,
            "audio_path": stored_path,
            "ground_truth_transcript": normalize_text(transcript),
            "ground_truth_response": normalize_text(response),
            "response_source": response_source,
        })
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "source_samples": len(candidates),
        "selected_samples": len(prepared),
        "limit": limit, "seed": seed, "response_mode": response_mode,
        "placeholder_responses": placeholder_count,
        "response_quality_is_test_only": placeholder_count > 0,
        "columns": {"audio": audio_column, "transcript": transcript_column, "response": response_column},
        "normalization": "NFC and collapsed whitespace; punctuation/case retained",
    }
    # Finish validation before creating any output. Never overwrite input audio/labels.
    protected = audio_paths | {
        path.with_suffix(suffix)
        for path in audio_paths for suffix in (".txt", ".response.txt")
    }
    if output_path in protected or metadata_path in protected:
        raise ValueError("Output paths conflict with source audio or labels")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/test.json"))
    parser.add_argument("--limit", type=int, default=100, help="Subset size; 0 selects all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--response-mode", choices=["require", "placeholder"], default="require")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--transcript-column", default="ground_truth_transcript")
    parser.add_argument("--response-column", default="ground_truth_response")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = prepare_dataset(
            args.source, args.output, limit=args.limit, seed=args.seed,
            response_mode=args.response_mode, audio_column=args.audio_column,
            transcript_column=args.transcript_column, response_column=args.response_column,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if metadata["placeholder_responses"]:
        print("Placeholder labels are for plumbing tests; response-quality scores are not valid benchmarks.")


if __name__ == "__main__":
    main()
