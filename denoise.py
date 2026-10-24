"""Spectral-subtraction noise reduction: the Task 2 research-extension fix.

evaluate.py on corpus_noisy/ exposes a measurable bottleneck: additive noise
raises ASR WER relative to corpus_clean/ (see reports/task2.md for the actual
numbers). This module is the "one improvement" the assignment asks for: a
magnitude spectral-subtraction filter applied before ASR, switched on with
``asr.denoise: true`` in config.yaml. It estimates a static noise magnitude
profile from the quietest frames of the signal itself (no separate noise-only
reference clip is required), subtracts it from every STFT frame with a floor
to limit musical-noise artifacts, then reconstructs the waveform by inverse
STFT using the original (unmodified) phase.

This is a standard, well-understood technique (Boll, 1979), not a novel
contribution; what is evaluated is whether it measurably narrows the
clean/noisy WER gap for this pipeline, at what latency cost, on this dataset.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


def spectral_subtract_denoise(
    audio: np.ndarray,
    sample_rate: int,
    *,
    noise_percentile: float = 10.0,
    oversubtraction: float = 1.5,
    floor: float = 0.05,
) -> np.ndarray:
    """Return a denoised copy of a 1-D mono float32 waveform.

    ``noise_percentile`` selects which per-bin magnitude percentile across
    time is treated as the noise floor (lower = closer to the quietest
    frames). ``oversubtraction`` scales how aggressively that floor is
    subtracted; ``floor`` keeps at least that fraction of the original
    magnitude per bin so subtraction cannot invert the spectrum to negative
    values or fully zero out frames (the usual "musical noise" trade-off).
    """
    if audio.ndim != 1:
        raise ValueError("audio must be a 1-D mono array")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("audio must contain finite, nonempty samples")
    if not 0 <= noise_percentile <= 100:
        raise ValueError("noise_percentile must be between 0 and 100")
    if oversubtraction < 0 or not 0 <= floor <= 1:
        raise ValueError("oversubtraction must be nonnegative and floor must be in [0, 1]")

    nperseg = min(1024, audio.size)
    if nperseg < 8:
        return audio.astype(np.float32)  # Too short to estimate a spectrum usefully.

    _, _, spectrum = stft(audio, fs=sample_rate, nperseg=nperseg)
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)
    noise_profile = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)
    subtracted = magnitude - oversubtraction * noise_profile
    cleaned_magnitude = np.maximum(subtracted, floor * magnitude)
    cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
    _, denoised = istft(cleaned_spectrum, fs=sample_rate, nperseg=nperseg)

    denoised = denoised[: audio.size].astype(np.float32)
    if denoised.size < audio.size:  # istft can return a few samples short at edges.
        denoised = np.pad(denoised, (0, audio.size - denoised.size))
    peak = np.max(np.abs(denoised))
    if peak > 1.0:
        denoised = denoised / peak
    return denoised
