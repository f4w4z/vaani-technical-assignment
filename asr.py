"""Whisper-compatible ASR with local audio decoding and synchronized timing.

Inference time includes audio decoding, resampling, and transcription, but
excludes model loading. Inputs supported by libsndfile (such as WAV/FLAC) work
without an FFmpeg installation. Model weights are downloaded on first use.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import math

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def resolve_runtime(device: str, dtype: str) -> tuple[torch.device, torch.dtype]:
    """Resolve an explicit device and precision shared by both wrappers."""
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    target = torch.device(device)
    if target.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda[:index], mps, or auto")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if dtype == "auto":
        dtype = "float16" if target.type == "cuda" else "float32"
    precisions = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in precisions:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if target.type == "cpu" and dtype == "float16":
        raise ValueError("Use float32 or bfloat16 for CPU inference")
    return target, precisions[dtype]


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class ASRWrapper:
    """Load one Whisper-compatible checkpoint and reuse it for audio files.

    This wrapper intentionally targets Whisper architectures. Checkpoints with
    other architectures require their own preprocessing and decoding adapter.
    Each instance processes one request at a time.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        *,
        device: str = "auto",
        dtype: str = "auto",
        cache_dir: str | None = None,
        revision: str = "main",
        language: str = "telugu",
        chunk_length_s: float = 30,
        batch_size: int = 1,
        generation: Mapping[str, Any] | None = None,
    ) -> None:
        if chunk_length_s <= 0 or batch_size < 1:
            raise ValueError("chunk_length_s and batch_size must be positive")
        self.device, precision = resolve_runtime(device, dtype)
        self.generation = {"task": "transcribe", "do_sample": False}
        self.generation.update(generation or {})
        self.generation["language"] = language
        processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir, revision=revision,
            trust_remote_code=False,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, cache_dir=cache_dir, revision=revision,
            torch_dtype=precision, use_safetensors=True, trust_remote_code=False,
        ).to(self.device).eval()
        if model.config.model_type != "whisper":
            raise ValueError("ASRWrapper requires a Whisper-compatible checkpoint")
        self.sample_rate = int(processor.feature_extractor.sampling_rate)
        self.recognizer = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
        )

    def transcribe(self, audio_path: str | Path) -> tuple[str, float]:
        """Return ``(transcript, elapsed_seconds)`` for a local audio file.

        Stereo channels are averaged to mono. Audio is resampled to the model's
        actual expected sample rate; changing a sample-rate label is insufficient.
        """
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        synchronize(self.device)
        started = perf_counter()
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[0] == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite, nonempty samples")
        audio = audio.mean(axis=1)
        if sample_rate != self.sample_rate:
            divisor = math.gcd(sample_rate, self.sample_rate)
            audio = resample_poly(
                audio, self.sample_rate // divisor, sample_rate // divisor,
            ).astype(np.float32)
        with torch.inference_mode():
            result = self.recognizer(
                {"array": audio, "sampling_rate": self.sample_rate},
                generate_kwargs=dict(self.generation),
            )
        synchronize(self.device)
        return str(result["text"]).strip(), perf_counter() - started
