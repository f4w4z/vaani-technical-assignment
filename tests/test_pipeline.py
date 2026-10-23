"""Small offline regression checks; no model weights are downloaded."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from asr import ASRWrapper
from llm import LLMWrapper
from pipeline import SpeechPipeline, load_config


class PipelineChecks(unittest.TestCase):
    def test_stage_handoff_and_empty_transcript(self) -> None:
        asr = Mock()
        llm = Mock()
        asr.transcribe.return_value = ("నమస్కారం", 0.1)
        llm.generate.return_value = ("నమస్తే!", 0.2)
        pipeline = SpeechPipeline(asr, llm)
        result = pipeline.run("question.wav")
        llm.generate.assert_called_once_with("నమస్కారం")
        self.assertEqual(result.response, "నమస్తే!")
        self.assertEqual((result.asr_seconds, result.llm_seconds), (0.1, 0.2))
        asr.transcribe.return_value = ("  ", 0.1)
        llm.reset_mock()
        with self.assertRaisesRegex(ValueError, "empty transcript"):
            pipeline.run("question.wav")
        llm.generate.assert_not_called()

    def test_config_paths_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("paths:\n  audio: data/question.wav\n", encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(
                Path(config["paths"]["audio"]),
                Path(directory) / "data" / "question.wav",
            )
            config_path.write_text("llm: []", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a mapping"):
                load_config(config_path)

    def test_audio_mono_resampling_and_missing_file(self) -> None:
        with patch("asr.AutoProcessor.from_pretrained") as processor, patch(
            "asr.AutoModelForSpeechSeq2Seq.from_pretrained"
        ) as model, patch("asr.pipeline") as pipeline_factory:
            processor.return_value.feature_extractor.sampling_rate = 16000
            model.return_value.to.return_value.eval.return_value.config.model_type = "whisper"
            recognizer = pipeline_factory.return_value
            recognizer.return_value = {"text": " నమస్కారం "}
            wrapper = ASRWrapper(device="cpu")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "stereo.wav"
                sf.write(path, np.ones((8000, 2), dtype=np.float32) * 0.1, 8000)
                transcript, elapsed = wrapper.transcribe(path)
                self.assertEqual(transcript, "నమస్కారం")
                self.assertGreaterEqual(elapsed, 0)
                supplied = recognizer.call_args.args[0]
                self.assertEqual(supplied["sampling_rate"], 16000)
                self.assertEqual(supplied["array"].shape, (16000,))
                self.assertEqual(recognizer.call_args.kwargs["generate_kwargs"]["language"], "telugu")
                with self.assertRaises(FileNotFoundError):
                    wrapper.transcribe(Path(directory) / "missing.wav")

    def test_llm_returns_only_continuation_and_checks_input_budget(self) -> None:
        class Inputs(dict):
            def to(self, device: torch.device) -> "Inputs":
                return self

        with patch("llm.AutoTokenizer.from_pretrained") as tokenizer_factory, patch(
            "llm.AutoModelForCausalLM.from_pretrained"
        ) as model_factory:
            tokenizer = tokenizer_factory.return_value
            tokenizer.chat_template = "chat template"
            tokenizer.pad_token_id = 0
            tokenizer.apply_chat_template.return_value = Inputs(
                input_ids=torch.tensor([[11, 12, 13]]),
                attention_mask=torch.ones((1, 3), dtype=torch.long),
            )
            tokenizer.decode.return_value = " నమస్తే! "
            model = model_factory.return_value.to.return_value.eval.return_value
            model.config = SimpleNamespace(max_position_embeddings=4096)
            model.generate.return_value = torch.tensor([[11, 12, 13, 21, 22]])
            wrapper = LLMWrapper(device="cpu", generation={"temperature": 0.7})
            response, elapsed = wrapper.generate("నమస్కారం")
            self.assertEqual(response, "నమస్తే!")
            self.assertGreaterEqual(elapsed, 0)
            self.assertEqual(tokenizer.decode.call_args.args[0].tolist(), [21, 22])
            self.assertNotIn("temperature", model.generate.call_args.kwargs)
            model.generate.reset_mock()
            wrapper.max_input_tokens = 2
            with self.assertRaisesRegex(ValueError, "limit"):
                wrapper.generate("నమస్కారం")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                wrapper.generate(" ")
            model.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
