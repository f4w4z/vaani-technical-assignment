"""Verify Task 2 setup and evaluator arithmetic without downloading models.

Run ``python verify_task2.py`` from the repository root. Writes a UTF-8 log,
machine-readable checks, and a Markdown summary to outputs/verification.
Dummy WAVs contain silence; model outputs AND latency values are synthetic.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import logging
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Literal
from unittest.mock import patch
import wave

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
Status = Literal["PASS", "FAIL", "WARN", "UNAVAILABLE"]


@dataclass(frozen=True)
class Check:
    """One independently reportable check, with actionable failure details."""

    name: str
    status: Status
    detail: str


def check_dependencies(requirements_path: Path) -> list[Check]:
    """Check installed versions AND importability against requirements.txt.

    packaging handles version constraints, including prereleases and local build
    labels. No packages are installed and no network requests are made.
    """
    checks = [Check("Python", "PASS" if sys.version_info >= (3, 10) else "FAIL", sys.version.split()[0])]
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:
        return checks + [Check("Dependency constraints", "UNAVAILABLE", f"Install packaging: {exc}")]
    modules = {"pyyaml": "yaml"}
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return checks + [Check("Dependencies", "UNAVAILABLE", str(exc))]
    for line in lines:
        declaration = line.split("#", 1)[0].strip()
        if not declaration:
            continue
        try:
            requirement = Requirement(declaration)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            version = metadata.version(requirement.name)
            name = requirement.name.lower().replace("-", "_")
            importlib.import_module(modules.get(name, name))
            compatible = requirement.specifier.contains(version)
            checks.append(Check(
                f"Dependency: {requirement.name}", "PASS" if compatible else "FAIL",
                f"Installed {version}; required {requirement.specifier or 'any version'}",
            ))
        except Exception as exc:
            checks.append(Check(f"Dependency: {declaration}", "FAIL", f"{type(exc).__name__}: {exc}"))
    return checks


def check_repository(root: Path) -> list[Check]:
    """Confirm required source files exist and parse without executing them."""
    checks = []
    for name in ("config.yaml", "asr.py", "llm.py", "pipeline.py", "evaluate.py",
                 "prepare_data.py", "plot_results.py", "requirements.txt"):
        path = root / name
        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".py":
                ast.parse(content, filename=str(path))
            checks.append(Check(f"File: {name}", "PASS", "Present; Python syntax valid" if name.endswith(".py") else "Present"))
        except (OSError, SyntaxError, UnicodeError) as exc:
            checks.append(Check(f"File: {name}", "FAIL", str(exc)))
    return checks


def validate_config(config: Any) -> None:
    """Validate parameters needed by the existing wrappers and paired evaluation.

    Model IDs are checked for nonempty text only: availability, chat-template
    compatibility, context limits, licenses, and model quality require real runs.
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def positive(value: Any, *, integer: bool = False) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, int if integer else (int, float))
            and math.isfinite(value) and value > 0
        )

    require(isinstance(config, dict), "Configuration must be a mapping")
    for section in ("paths", "runtime", "asr", "llm"):
        require(isinstance(config.get(section), dict), f"{section} must be a mapping")
    paths, runtime = config["paths"], config["runtime"]
    for name in ("audio", "output", "cache_dir"):
        value = paths.get(name)
        require(isinstance(value, str) and bool(value.strip()), f"paths.{name} must be a nonempty path")
    device, dtype = runtime.get("device", "auto"), runtime.get("dtype", "auto")
    require(isinstance(device, str) and re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", device) is not None,
            "runtime.device must be auto, cpu, mps, or cuda[:index]")
    require(isinstance(dtype, str) and dtype in {"auto", "float32", "float16", "bfloat16"}, "Invalid runtime.dtype")
    require(not (device == "cpu" and dtype == "float16"), "CPU float16 is unsupported by this wrapper")
    for section in ("asr", "llm"):
        settings = config[section]
        for name in ("model_name", "revision"):
            value = settings.get(name)
            require(isinstance(value, str) and bool(value.strip()), f"{section}.{name} must be nonempty text")
        generation = settings.get("generation", {})
        require(isinstance(generation, dict), f"{section}.generation must be a mapping")
        require(generation.get("do_sample", False) is False, f"{section}.generation.do_sample must be false for paired evaluation")
        for name in ("max_new_tokens", "num_beams"):
            if name in generation:
                require(positive(generation[name], integer=True), f"{section}.generation.{name} must be a positive integer")
        for name in ("temperature", "repetition_penalty"):
            if name in generation:
                require(positive(generation[name]), f"{section}.generation.{name} must be positive")
        if "top_p" in generation:
            require(positive(generation["top_p"]) and generation["top_p"] <= 1, f"{section}.generation.top_p must be in (0, 1]")
    asr, llm = config["asr"], config["llm"]
    asr_language = asr.get("language")
    asr_model = str(asr.get("model_name", "")).lower()
    require(
        asr_language in ("telugu", "te")
        or (asr_language is None and "telugu" in asr_model),
        "asr.language must target Telugu (telugu or te), or be null only for a Telugu-specific checkpoint",
    )
    asr_task = asr.get("generation", {}).get("task", "transcribe")
    require(asr_task in ("transcribe", None), "ASR task must be transcribe (or unset for a single-language checkpoint)")
    require(positive(asr.get("chunk_length_s", 30)), "asr.chunk_length_s must be positive")
    require(positive(asr.get("batch_size", 1), integer=True), "asr.batch_size must be a positive integer")
    require(positive(llm.get("max_input_tokens", 2048), integer=True), "llm.max_input_tokens must be a positive integer")
    require(isinstance(llm.get("system_prompt"), str) and bool(llm["system_prompt"].strip()), "llm.system_prompt must be nonempty")


def check_configuration(path: Path) -> tuple[dict[str, Any] | None, list[Check]]:
    """Parse YAML independently of heavy model imports, then validate fields."""
    try:
        import yaml
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_config(config)
    except Exception as exc:
        return None, [Check("Configuration", "FAIL", f"{type(exc).__name__}: {exc}")]
    checks = [Check("Configuration", "PASS", f"Validated {path}")]
    for section in ("asr", "llm"):
        if config[section]["revision"] == "main":
            checks.append(Check(f"Revision: {section}", "WARN", "Pin a model commit before reproducible research runs"))
    audio = (path.parent / Path(config["paths"]["audio"]).expanduser()).resolve()
    checks.append(Check("Configured input audio", "PASS" if audio.is_file() else "WARN",
                        f"{audio}" + ("" if audio.is_file() else " is absent; not needed for the dummy run")))
    try:
        from asr import resolve_runtime
        device, dtype = resolve_runtime(config["runtime"].get("device", "auto"), config["runtime"].get("dtype", "auto"))
        if device.type == "cuda":
            import torch
            torch.cuda.get_device_properties(device)  # Also detects invalid CUDA indices.
        checks.append(Check("Runtime device", "PASS", f"Resolved {device}, {dtype}; model memory is not tested"))
    except (ImportError, ModuleNotFoundError) as exc:
        checks.append(Check("Runtime device", "UNAVAILABLE", str(exc)))
    except Exception as exc:
        checks.append(Check("Runtime device", "FAIL", str(exc)))
    return config, checks


def check_directories(config: dict[str, Any] | None, config_path: Path, output_dir: Path) -> list[Check]:
    """Create/check output and model-cache directories with real UTF-8 writes.

    Evaluation/plot defaults follow the working directory, matching their CLIs;
    configured pipeline/cache paths follow the configuration file's directory.
    """
    targets = {"Verification output": output_dir, "Prepared dataset output": Path("data"),
               "Evaluation output": Path("outputs/evaluation"),
               "Plot output": Path("outputs/plots")}
    if config:
        paths = config["paths"]
        targets["Pipeline output"] = (config_path.parent / Path(paths["output"]).expanduser()).parent
        targets["Model cache"] = config_path.parent / Path(paths["cache_dir"]).expanduser()
    checks = []
    for label, directory in targets.items():
        try:
            if label == "Pipeline output" and config:
                target_file = config_path.parent / Path(config["paths"]["output"]).expanduser()
                if target_file.exists() and not target_file.is_file():
                    raise OSError(f"Pipeline output must be a file path: {target_file}")
            directory = directory.resolve()
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", dir=directory, suffix=".tmp") as handle:
                handle.write("తెలుగు")
                handle.flush()
                handle.seek(0)
                if handle.read() != "తెలుగు":
                    raise OSError("UTF-8 write/read mismatch")
            checks.append(Check(label, "PASS", f"Writable: {directory}"))
        except OSError as exc:
            checks.append(Check(label, "FAIL", str(exc)))
    return checks


def _expect(actual: Any, expected: Any, label: str) -> None:
    """Assertions remain active under python -O and identify incorrect metrics."""
    matches = math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9) if isinstance(expected, float) else actual == expected
    if not matches:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def verify_metric_primitives() -> None:
    """Check known DP answers, insertions/deletions, and Telugu code-point CER."""
    from evaluate import character_error_rate, levenshtein_distance, real_time_factor, word_error_rate

    _expect(levenshtein_distance("kitten", "sitting"), 3, "Levenshtein")
    _expect(word_error_rate("కా", "కా కి కు"), 2.0, "WER insertions > 1")
    _expect(word_error_rate("కా కి", ""), 1.0, "WER deletions")
    _expect(word_error_rate("", "కా"), None, "Empty-reference WER")
    _expect(character_error_rate("కా కి", "కా కు"), 0.25, "Telugu CER substitution")
    _expect(character_error_rate("e\u0301", "é"), 0.0, "NFC normalization")
    _expect(real_time_factor(0.6, 2), 0.3, "RTF")


def run_mock_evaluation(work_dir: Path) -> dict[str, Any]:
    """Exercise the REAL pipeline/evaluator with three silent WAV fixtures.

    Dependency-injected ASR/LLM stand-ins provide fixed strings and stage times;
    only the pipeline clock is patched. Thus pipeline handoff, real WAV duration
    reading, evaluator scoring, corpus aggregation, and report writing all run.
    This function is sequential and the patched clock must not be used by other
    threads. No model constructors, GPU inference, sleeps, or downloads occur.
    """
    from evaluate import EvaluationSample, evaluate_dataset, load_dataset, save_report
    from pipeline import SpeechPipeline

    work_dir.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("perfect", "కా కి", "కా కి", 1, 0.1, 0.2, 0.3),
        ("substitution", "కా కి", "కా కు", 2, 0.2, 0.4, 0.6),
        ("deletion", "కా కి కు", "కా", 4, 0.4, 0.8, 1.2),
    ]
    expected_response = "సరైన సమాధానం"
    records = []
    transcripts = {}
    for name, reference, hypothesis, duration, asr_time, _, _ in fixtures:
        path = work_dir / f"{name}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * (16000 * duration))
        transcripts[path.name] = (hypothesis, asr_time)
        records.append({"id": name, "audio_path": path.name,
                        "ground_truth_transcript": reference, "ground_truth_response": expected_response})
    dataset_path = work_dir / "test.json"
    dataset_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples: list[EvaluationSample] = load_dataset(dataset_path)

    class MockASR:
        def transcribe(self, audio_path: str | Path) -> tuple[str, float]:
            return transcripts[Path(audio_path).name]

    class MockLLM:
        def generate(self, prompt: str) -> tuple[str, float]:
            if prompt == "కా కు":
                return "తప్పు సమాధానం", 0.4
            if prompt == "కా":
                return "", 0.8
            if prompt in {"కా కి", "కా కి కు"}:
                return expected_response, 0.2
            raise ValueError(f"Unexpected mock prompt: {prompt!r}")

    pipeline = SpeechPipeline(MockASR(), MockLLM())
    # Exactly two clock reads per run; times are simulated, not measured speed.
    clock = [0.0, 0.3, 10.0, 10.6, 20.0, 21.2]
    with patch("pipeline.perf_counter", side_effect=clock):
        report = evaluate_dataset(pipeline, samples, fail_fast=True)
    _expect(report["summary"]["samples_succeeded"], 3, "Completed pairs")
    _expect(report["summary"]["samples_failed"], 0, "Failed pairs")
    for row, fixture, wer, cer, quality in zip(
        report["samples"], fixtures, (0.0, 0.5, 2 / 3), (0.0, 0.25, 2 / 3), (1.0, 0.5, 0.0),
    ):
        for key, expected in {
            "wer": wer, "cer": cer, "asr_seconds": fixture[4], "llm_seconds": fixture[5],
            "total_seconds": fixture[6], "audio_seconds": float(fixture[3]), "rtf": 0.3,
            "clean_llm_seconds": 0.2, "clean_response_quality": 1.0,
            "asr_response_quality": quality, "quality_drop": 1.0 - quality,
        }.items():
            _expect(row[key], expected, f"{row['id']}.{key}")
    for key, expected in {"corpus_wer": 3 / 7, "corpus_cer": 5 / 14, "corpus_rtf": 0.3,
                          "degraded_fraction": 2 / 3, "total_pipeline_seconds": 2.1}.items():
        _expect(report["summary"][key], expected, key)
    _expect(report["summary"]["quality_drop"]["mean"], 0.5, "Mean quality drop")
    report["metadata"].update({"mock": True, "timings": "synthetic; not measured performance",
                               "audio": "generated silence; not Telugu speech"})
    save_report(report, work_dir)
    restored = json.loads((work_dir / "report.json").read_text(encoding="utf-8"))
    _expect(len(restored["samples"]), 3, "JSON report round trip")
    return report


def markdown_summary(checks: list[Check], report: dict[str, Any] | None) -> str:
    """Render copyable setup and metric tables with unambiguous mock labeling."""
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    ready = not any(check.status in {"FAIL", "UNAVAILABLE"} for check in checks)
    lines = ["# Task 2 verification", "", f"Setup checks: **{'PASS' if ready else 'NOT READY'}**.", "",
             "**Synthetic verification only. Outputs and timings below are mocked; these are not research accuracy or performance results.**", "",
             "| Check | Status | Detail |", "| --- | --- | --- |"]
    lines.extend(f"| {escape(check.name)} | {check.status} | {escape(check.detail)} |" for check in checks)
    if report is None:
        lines.extend(["", "Mock evaluation results unavailable; see the checks and verification log."])
        return "\n".join(lines) + "\n"
    lines.extend(["", "## Three-sample mock evaluation", "",
                  "WER/CER/RTF are ratios. Quality is word-overlap F1. Positive drop = clean quality − ASR quality.", "",
                  "| Dummy case | WER | CER | Audio s | ASR s* | LLM s* | Total s* | RTF* | ASR quality | Clean quality | Drop |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in report["samples"]:
        fields = ("wer", "cer", "audio_seconds", "asr_seconds", "llm_seconds", "total_seconds", "rtf",
                  "asr_response_quality", "clean_response_quality", "quality_drop")
        lines.append("| " + escape(row["id"]) + " | " + " | ".join(f"{row[field]:.4f}" for field in fields) + " |")
    summary = report["summary"]
    lines.extend(["", "| Aggregate mock metric | Value |", "| --- | ---: |",
                  f"| Corpus WER (3/7) | {summary['corpus_wer']:.4f} |",
                  f"| Corpus CER (5/14) | {summary['corpus_cer']:.4f} |",
                  f"| Corpus RTF* | {summary['corpus_rtf']:.4f} |",
                  f"| Mean quality drop | {summary['quality_drop']['mean']:.4f} |",
                  f"| Degraded pairs | {summary['degraded_fraction']:.4f} |", "",
                  "*All timing-derived figures use synthetic clock/stage values. CER counts NFC Unicode code points without whitespace.",
                  "The clean-transcript LLM call is excluded from pipeline latency/RTF. No real model quality, model availability, or memory capacity was verified."])
    return "\n".join(lines) + "\n"


def verify_task2(config_path: Path, output_dir: Path) -> tuple[list[Check], dict[str, Any] | None]:
    """Run independent checks even when environment readiness fails elsewhere."""
    checks = check_repository(ROOT) + check_dependencies(ROOT / "requirements.txt")
    config, config_checks = check_configuration(config_path)
    checks.extend(config_checks)
    checks.extend(check_directories(config, config_path, output_dir))
    report = None
    for name, operation in (("Dynamic-programming metric answers", verify_metric_primitives),
                            ("Three-sample mock evaluation", lambda: run_mock_evaluation(output_dir / "mock"))):
        try:
            result = operation()
            if name == "Three-sample mock evaluation":
                report = result
            checks.append(Check(name, "PASS", "All fixed expected values match"))
        except ImportError as exc:
            checks.append(Check(name, "UNAVAILABLE", f"Missing/broken dependency: {exc}"))
        except Exception as exc:
            LOGGER.exception("%s failed", name)
            checks.append(Check(name, "FAIL", f"{type(exc).__name__}: {exc}"))
    return checks, report


def main() -> None:
    """Persist diagnostics even if setup checks fail; exit 1 for any failed/unavailable check."""
    # Windows redirected streams may default to cp1252; tables/paths use Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verification"))
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "verification.log", mode="w", encoding="utf-8")])
        LOGGER.info("Starting offline Task 2 verification; no models will be loaded")
        checks, report = verify_task2(args.config.expanduser().resolve(), output_dir)
        for check in checks:
            level = logging.ERROR if check.status in {"FAIL", "UNAVAILABLE"} else (
                logging.WARNING if check.status == "WARN" else logging.INFO
            )
            LOGGER.log(level, "%s %s: %s", check.status, check.name, check.detail)
        summary = markdown_summary(checks, report)
        (output_dir / "summary.md").write_text(summary, encoding="utf-8")
        ready = not any(check.status in {"FAIL", "UNAVAILABLE"} for check in checks)
        (output_dir / "checks.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "ready": ready,
            "python_executable": sys.executable, "config_path": str(args.config.resolve()),
            "mock_evaluation_passed": report is not None,
            "checks": [asdict(check) for check in checks],
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        LOGGER.info("Summary saved to %s", output_dir / "summary.md")
        print(summary)
    except OSError as exc:
        parser.exit(1, f"Cannot write verification outputs: {exc}\n")
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
