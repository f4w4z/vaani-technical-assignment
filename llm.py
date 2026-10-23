"""Local causal-language-model inference using Hugging Face Transformers."""

from time import perf_counter
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr import resolve_runtime, synchronize


class LLMWrapper:
    """Load a chat checkpoint once and return only newly generated text.

    The checkpoint must provide a chat template supporting system/user roles.
    Qwen2.5-Instruct is the default. Base models such as Sarvam-1 may need a
    model-specific prompt format and should not be substituted blindly.
    Timing includes prompt tokenization, generation, and response decoding,
    excluding model loading. Each instance processes one request at a time.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        device: str = "auto",
        dtype: str = "auto",
        cache_dir: str | None = None,
        revision: str = "main",
        system_prompt: str = "Answer clearly and concisely in Telugu script.",
        max_input_tokens: int = 2048,
        generation: Mapping[str, Any] | None = None,
    ) -> None:
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        self.device, precision = resolve_runtime(device, dtype)
        self.system_prompt = system_prompt
        self.max_input_tokens = max_input_tokens
        self.generation = {"max_new_tokens": 256, "do_sample": False}
        self.generation.update(generation or {})
        if int(self.generation["max_new_tokens"]) < 1:
            raise ValueError("max_new_tokens must be positive")
        if not self.generation.get("do_sample", False):
            self.generation.pop("temperature", None)
            self.generation.pop("top_p", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, revision=revision,
            trust_remote_code=False,
        )
        if not self.tokenizer.chat_template:
            raise ValueError("Choose an instruction model with a chat template")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, revision=revision,
            torch_dtype=precision, use_safetensors=True, trust_remote_code=False,
        ).to(self.device).eval()

    def generate(self, prompt: str) -> tuple[str, float]:
        """Return ``(response, elapsed_seconds)`` without echoing the prompt.

        Reject oversized inputs instead of silently truncating Telugu requests
        or removing chat-control tokens. No history is retained between calls.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        synchronize(self.device)
        started = perf_counter()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt.strip()},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        input_length = inputs["input_ids"].shape[-1]
        if input_length > self.max_input_tokens:
            raise ValueError(
                f"Prompt has {input_length} tokens; limit is {self.max_input_tokens}"
            )
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and input_length + self.generation["max_new_tokens"] > context_limit:
            raise ValueError("Prompt plus response budget exceeds model context")
        generation = dict(self.generation)
        if "pad_token_id" not in generation:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is not None:
                generation["pad_token_id"] = pad_id
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        response = self.tokenizer.decode(
            output[0, input_length:], skip_special_tokens=True,
        ).strip()
        synchronize(self.device)
        return response, perf_counter() - started
