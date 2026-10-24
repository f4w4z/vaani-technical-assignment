"""Offline checks for the spectral-subtraction research-extension filter."""

import unittest

import numpy as np

from denoise import spectral_subtract_denoise


def _bursty_tone(frequency: float, seconds: float, sample_rate: int) -> np.ndarray:
    """A tone with silent gaps, like speech: spectral subtraction needs quiet
    frames somewhere in each frequency bin to estimate a noise floor from."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    envelope = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(np.float32)  # 2 Hz on/off
    return (envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class DenoiseChecks(unittest.TestCase):
    def test_reduces_noise_power_while_keeping_tone_energy(self) -> None:
        sample_rate = 16000
        rng = np.random.default_rng(0)
        tone = _bursty_tone(440.0, 1.0, sample_rate)
        noise = rng.normal(0.0, 0.2, size=tone.shape).astype(np.float32)
        noisy = tone + noise

        denoised = spectral_subtract_denoise(noisy, sample_rate)

        self.assertEqual(denoised.shape, noisy.shape)
        self.assertTrue(np.isfinite(denoised).all())
        error_before = np.mean((noisy - tone) ** 2)
        error_after = np.mean((denoised - tone) ** 2)
        self.assertLess(error_after, error_before)

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-D"):
            spectral_subtract_denoise(np.zeros((4, 2), dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "finite, nonempty"):
            spectral_subtract_denoise(np.array([], dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "finite, nonempty"):
            spectral_subtract_denoise(np.array([float("nan")], dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "noise_percentile"):
            spectral_subtract_denoise(np.ones(100, dtype=np.float32), 16000, noise_percentile=150)
        with self.assertRaisesRegex(ValueError, "oversubtraction"):
            spectral_subtract_denoise(np.ones(100, dtype=np.float32), 16000, oversubtraction=-1)

    def test_short_signal_passes_through_unchanged(self) -> None:
        short = np.ones(4, dtype=np.float32)
        result = spectral_subtract_denoise(short, 16000)
        np.testing.assert_array_equal(result, short)


if __name__ == "__main__":
    unittest.main()
