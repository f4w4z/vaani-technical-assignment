"""Telugu Audio -> ASR -> LLM pipeline and JSON command-line entry point.

Example::

    python pipeline.py --config config.yaml --audio data/question.wav

Model loading is excluded from all reported inference timings. Both models
remain resident, so the selected device must have room for both checkpoints.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from asr import ASRWrapper
from llm import LLMWrapper

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """One record for downstream latency and reference-based evaluation."""

    audio_path: str
    transcript: str
    response: str
    asr_seconds: float
    llm_seconds: float
    total_seconds: float


class SpeechPipeline:
    """Compose reusable model instances; accept substitutes for testing."""

    def __init__(self, asr: ASRWrapper, llm: LLMWrapper) -> None:
        self.asr = asr
        self.llm = llm

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SpeechPipeline":
        """Load model instances from a configuration returned by load_config."""
        runtime = config.get("runtime", {})
        shared = {
            "device": runtime.get("device", "auto"),
            "dtype": runtime.get("dtype", "auto"),
            "cache_dir": config.get("paths", {}).get("cache_dir"),
        }
        LOGGER.info("Loading ASR model")
        asr = ASRWrapper(**shared, **config.get("asr", {}))
        LOGGER.info("Loading LLM")
        llm = LLMWrapper(**shared, **config.get("llm", {}))
        return cls(asr, llm)

    def run(self, audio_path: str | Path) -> PipelineResult:
        """Run both stages, rejecting empty transcriptions before generation."""
        path = Path(audio_path).expanduser().resolve()
        started = perf_counter()
        transcript, asr_seconds = self.asr.transcribe(path)
        if not transcript.strip():
            raise ValueError("ASR returned an empty transcript; LLM was not called")
        response, llm_seconds = self.llm.generate(transcript)
        return PipelineResult(
            audio_path=str(path), transcript=transcript, response=response,
            asr_seconds=asr_seconds, llm_seconds=llm_seconds,
            total_seconds=perf_counter() - started,
        )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read safe YAML and resolve configured paths relative to its directory."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("paths", "runtime", "asr", "llm"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
    for section in ("asr", "llm"):
        generation = config.get(section, {}).get("generation")
        if generation is not None and not isinstance(generation, dict):
            raise ValueError(f"{section}.generation must be a mapping")
    paths = config.setdefault("paths", {})
    for key in ("audio", "output", "cache_dir"):
        value = paths.get(key)
        if value is not None:
            resolved = Path(value).expanduser()
            paths[key] = str((path.parent / resolved).resolve())
    return config


def main() -> None:
    """Read configuration, run one audio file, and print/save UTF-8 JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--audio", type=Path, help="Override audio path (relative to current directory)")
    parser.add_argument("--output", type=Path, help="Override JSON path (relative to current directory)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config)
        audio = args.audio or config["paths"].get("audio")
        if not audio or not Path(audio).expanduser().is_file():
            raise FileNotFoundError(f"Provide an existing audio file via --audio: {audio}")
        pipeline = SpeechPipeline.from_config(config)
        result = pipeline.run(audio)
        payload = json.dumps(asdict(result), ensure_ascii=False, indent=2)
        output = args.output or config["paths"].get("output")
        if output:
            destination = Path(output).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    except (OSError, ValueError, RuntimeError, TypeError, yaml.YAMLError) as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
