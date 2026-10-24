"""Synthesize a reproducible Telugu Q&A speech corpus for Task 2 evaluation.

No recorded Telugu corpus ships with this repository (see README.md). This
script builds one instead: a fixed bank of Telugu factual question/answer text
pairs (state capitals and small addition problems, both template-generated so
every fact is checkable) is synthesized to speech with an open Telugu TTS
model, then written twice: once as clean audio and once with additive white
noise at a fixed SNR. The two folders are evaluate.py's two "linguistic or
acoustic conditions" and the noisy one is also the bottleneck target for the
Task 2 research extension (see denoise.py).

Usage:
    python synthesize_corpus.py --clean-out corpus_clean --noisy-out corpus_noisy

Each output folder matches prepare_data.py's documented sidecar format:
    question_001.wav / question_001.txt / question_001.response.txt

This script is a one-time data-generation utility, not part of the evaluated
pipeline. Its outputs are synthetic speech, not real human recordings; see
reports/task2.md for why that limits what the resulting numbers can claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

TTS_MODEL_NAME = "facebook/mms-tts-tel"
TARGET_SNR_DB = 8.0
NOISE_SEED = 42

NUMBER_WORDS = {
    1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు", 5: "ఐదు",
    6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది", 10: "పది",
    11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పదనాలుగు", 15: "పదిహేను",
    16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పందొమ్మిది", 20: "ఇరవై",
}

CAPITALS = [
    ("తెలంగాణ", "హైదరాబాద్"), ("ఆంధ్రప్రదేశ్", "అమరావతి"), ("తమిళనాడు", "చెన్నై"),
    ("కర్ణాటక", "బెంగళూరు"), ("కేరళ", "తిరువనంతపురం"), ("మహారాష్ట్ర", "ముంబై"),
    ("గుజరాత్", "గాంధీనగర్"), ("రాజస్థాన్", "జైపూర్"), ("పంజాబ్", "చండీగఢ్"),
    ("పశ్చిమ బెంగాల్", "కోల్‌కతా"), ("బీహార్", "పాట్నా"), ("ఒడిశా", "భువనేశ్వర్"),
    ("మధ్యప్రదేశ్", "భోపాల్"), ("ఉత్తరప్రదేశ్", "లక్నో"), ("హర్యానా", "చండీగఢ్"),
    ("అస్సాం", "దిస్పూర్"), ("ఛత్తీస్‌గఢ్", "రాయ్‌పూర్"), ("జార్ఖండ్", "రాంచీ"),
    ("గోవా", "పనాజీ"), ("సిక్కిం", "గ్యాంగ్‌టక్"), ("మణిపూర్", "ఇంఫాల్"),
    ("మేఘాలయ", "షిల్లాంగ్"), ("నాగాలాండ్", "కోహిమా"), ("త్రిపుర", "అగర్తలా"),
    ("మిజోరం", "ఐజ్వాల్"), ("అరుణాచల్ ప్రదేశ్", "ఇటానగర్"), ("ఉత్తరాఖండ్", "డెహ్రాడూన్"),
    ("హిమాచల్ ప్రదేశ్", "షిమ్లా"), ("భారతదేశ", "న్యూఢిల్లీ"),
]


def build_qa_pairs() -> list[tuple[str, str, str]]:
    """Return (id, question, answer) triples: capitals, then addition facts."""
    pairs: list[tuple[str, str, str]] = []
    for index, (place, capital) in enumerate(CAPITALS, start=1):
        question = f"{place} రాజధాని ఏది?"
        answer = f"{place} రాజధాని {capital}."
        pairs.append((f"cap-{index:03d}", question, answer))
    index = 0
    for a in range(1, 11):
        for b in range(1, 11):
            index += 1
            total = a + b
            question = f"{NUMBER_WORDS[a]} మరియు {NUMBER_WORDS[b]} కలిపితే ఎంత వస్తుంది?"
            answer = f"{NUMBER_WORDS[a]} మరియు {NUMBER_WORDS[b]} కలిపితే {NUMBER_WORDS[total]} వస్తుంది."
            pairs.append((f"sum-{index:03d}", question, answer))
    return pairs


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise at a fixed signal-to-noise ratio."""
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    if signal_power <= 0:
        raise ValueError("Cannot add noise to a silent signal")
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    noisy = signal + noise
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-out", type=Path, default=Path("corpus_clean"))
    parser.add_argument("--noisy-out", type=Path, default=Path("corpus_noisy"))
    parser.add_argument("--snr-db", type=float, default=TARGET_SNR_DB)
    parser.add_argument("--limit", type=int, default=0, help="Synthesize only the first N pairs (0 = all)")
    parser.add_argument("--skip-clean", action="store_true", help="Resynthesize audio but only write the noisy copy")
    args = parser.parse_args()

    import torch
    from transformers import VitsModel, AutoTokenizer

    pairs = build_qa_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Synthesizing {len(pairs)} Telugu Q&A pairs with {TTS_MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitsModel.from_pretrained(TTS_MODEL_NAME).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
    sample_rate = model.config.sampling_rate
    rng = np.random.default_rng(NOISE_SEED)

    if not args.skip_clean:
        args.clean_out.mkdir(parents=True, exist_ok=True)
    args.noisy_out.mkdir(parents=True, exist_ok=True)

    for position, (sample_id, question, answer) in enumerate(pairs, start=1):
        inputs = tokenizer(question, return_tensors="pt").to(device)
        with torch.no_grad():
            waveform = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)

        if not args.skip_clean:
            clean_path = args.clean_out / f"question_{sample_id}.wav"
            sf.write(clean_path, waveform, sample_rate)
            (args.clean_out / f"question_{sample_id}.txt").write_text(question, encoding="utf-8")
            (args.clean_out / f"question_{sample_id}.response.txt").write_text(answer, encoding="utf-8")

        noisy_waveform = add_noise(waveform, args.snr_db, rng)
        noisy_path = args.noisy_out / f"question_{sample_id}.wav"
        sf.write(noisy_path, noisy_waveform, sample_rate)
        (args.noisy_out / f"question_{sample_id}.txt").write_text(question, encoding="utf-8")
        (args.noisy_out / f"question_{sample_id}.response.txt").write_text(answer, encoding="utf-8")

        if position % 20 == 0 or position == len(pairs):
            print(f"  {position}/{len(pairs)} synthesized")

    print(f"Clean corpus: {args.clean_out.resolve()}")
    print(f"Noisy corpus ({args.snr_db:.1f} dB SNR): {args.noisy_out.resolve()}")


if __name__ == "__main__":
    main()
