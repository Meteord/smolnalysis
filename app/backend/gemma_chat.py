from __future__ import annotations

import argparse
from functools import lru_cache
import logging
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .adapter_registry import DEFAULT_ADAPTER, get_adapter, list_adapters
except ImportError:
    from adapter_registry import DEFAULT_ADAPTER, get_adapter, list_adapters

try:
    from ..hf_tracing import huggingface_span
except ImportError:
    try:
        from hf_tracing import huggingface_span
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def huggingface_span(name: str, attributes: dict[str, Any] | None = None):
            yield None


BASE_MODEL_ID = "google/gemma-4-E4B-it"
BASE_ADAPTER_NAMES = {"base", "none", "no_adapter", "no-adapter"}
AUTO_ADAPTER_NAMES = {"auto", "router"}
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("SMOLNALYSIS_GEMMA_MAX_NEW_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("SMOLNALYSIS_GEMMA_TEMPERATURE", "1.0"))
DEFAULT_TOP_P = float(os.getenv("SMOLNALYSIS_GEMMA_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("SMOLNALYSIS_GEMMA_TOP_K", "64"))
logger = logging.getLogger(__name__)


def get_input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


class GemmaAdapterModel:
    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        initial_adapter: str | None = DEFAULT_ADAPTER,
    ) -> None:
        self.base_model_id = base_model_id
        self.tokenizer: Any = None
        self.model: Any = None
        self.loaded_adapters: set[str] = set()
        self.active_adapter: str | None = None
        self.auto_adapter = False

        logger.info("initializing Gemma runtime: base_model=%s initial_adapter=%s", base_model_id, initial_adapter or "base")
        self._load_tokenizer(initial_adapter)
        self._load_base_model()
        if self._is_auto_adapter(initial_adapter):
            self.auto_adapter = True
            self.active_adapter = None
        elif not self._is_base_adapter(initial_adapter):
            self.set_adapter(initial_adapter)
        else:
            self.active_adapter = None

    @staticmethod
    def _is_base_adapter(adapter_name: str | None) -> bool:
        return adapter_name is None or adapter_name.lower() in BASE_ADAPTER_NAMES

    @staticmethod
    def _is_auto_adapter(adapter_name: str | None) -> bool:
        return adapter_name is not None and adapter_name.lower() in AUTO_ADAPTER_NAMES

    def _load_tokenizer(self, adapter_name: str | None) -> None:
        if self._is_base_adapter(adapter_name) or self._is_auto_adapter(adapter_name):
            tokenizer_path = self.base_model_id
        else:
            assert adapter_name is not None
            adapter = get_adapter(adapter_name)
            tokenizer_path = adapter.path if adapter.exists else self.base_model_id

        with huggingface_span(
            "tokenizer.load",
            {
                "gen_ai.system": "huggingface",
                "gen_ai.request.model": str(tokenizer_path),
                "smolnalysis.hf.base_model": self.base_model_id,
                "smolnalysis.hf.initial_adapter": adapter_name or "base",
            },
        ):
            logger.info("loading tokenizer: path=%s initial_adapter=%s", tokenizer_path, adapter_name or "base")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_base_model(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        with huggingface_span(
            "model.load",
            {
                "gen_ai.system": "huggingface",
                "gen_ai.request.model": self.base_model_id,
                "smolnalysis.hf.quantization": "4bit-nf4",
                "smolnalysis.hf.device_map": "auto",
                "smolnalysis.hf.torch_dtype": "bfloat16",
            },
        ):
            logger.info("loading base model: model=%s quantization=4bit-nf4 device_map=auto", self.base_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

    def set_adapter(self, adapter_name: str | None) -> None:
        previous = "auto" if self.auto_adapter else self.active_adapter or "base"
        if self._is_auto_adapter(adapter_name):
            self.auto_adapter = True
            self.active_adapter = None
            self.model.eval()
            logger.info("adapter mode changed: previous=%s requested=%s active=auto", previous, adapter_name or "auto")
            return

        self.auto_adapter = False

        if self._is_base_adapter(adapter_name):
            self.active_adapter = None
            self.model.eval()
            logger.info("adapter mode changed: previous=%s requested=%s active=base", previous, adapter_name or "base")
            return

        assert adapter_name is not None
        adapter = get_adapter(adapter_name)
        if not adapter.exists:
            logger.error("adapter missing: requested=%s path=%s", adapter.name, adapter.path)
            raise FileNotFoundError(
                f"Adapter '{adapter_name}' is registered but not trained at {adapter.path}"
            )

        with huggingface_span(
            "adapter.load",
            {
                "gen_ai.system": "huggingface",
                "gen_ai.request.model": self.base_model_id,
                "smolnalysis.hf.adapter": adapter.name,
                "smolnalysis.hf.adapter_path": str(adapter.path),
                "smolnalysis.hf.adapter_already_loaded": adapter.name in self.loaded_adapters,
            },
        ):
            logger.info(
                "loading adapter: requested=%s path=%s already_loaded=%s",
                adapter.name,
                adapter.path,
                adapter.name in self.loaded_adapters,
            )
            if not self.loaded_adapters:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter.path,
                    adapter_name=adapter.name,
                )
                self.loaded_adapters.add(adapter.name)
            elif adapter.name not in self.loaded_adapters:
                self.model.load_adapter(adapter.path, adapter_name=adapter.name)
                self.loaded_adapters.add(adapter.name)

        self.model.set_adapter(adapter.name)
        self.model.eval()
        self.active_adapter = adapter.name
        logger.info("adapter mode changed: previous=%s requested=%s active=%s", previous, adapter_name, self.active_adapter)

    def route_adapter(self, user_text: str) -> str:
        prompt = (
            "Choose the adapter for the next user message.\n"
            "Return only JSON with this shape: {\"adapter\":\"base\"} or {\"adapter\":\"retrieval\"}.\n"
            "Use retrieval when the user asks to find, search, retrieve, list, inspect, query, "
            "or analyze data.\n"
            "Use base for ordinary chat, and anything that does not require external knowlege.\n\n"
            f"User message: {user_text}"
        )
        decision = self._generate_messages(
            [{"role": "user", "content": prompt}],
            max_new_tokens=32,
            temperature=0.0,
            force_base=True,
        ).lower()

        routed_adapter = "retrieval" if "retrieval" in decision else "base"
        logger.info(
            "adapter route decision: routed_adapter=%s user_chars=%d raw_decision=%r",
            routed_adapter,
            len(user_text),
            decision[:300],
        )
        return routed_adapter

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        return self._generate_messages(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            force_base=False,
        )

    def _generate_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        top_k: int = 64,
        force_base: bool = False,
    ) -> str:
        with huggingface_span(
            "model.generate",
            {
                "gen_ai.system": "huggingface",
                "gen_ai.request.model": self.base_model_id,
                "gen_ai.request.max_tokens": max_new_tokens,
                "gen_ai.request.temperature": temperature,
                "gen_ai.request.top_p": top_p,
                "smolnalysis.hf.top_k": top_k,
                "smolnalysis.hf.adapter": self.active_adapter or "base",
                "smolnalysis.hf.force_base": force_base,
                "smolnalysis.hf.auto_adapter": self.auto_adapter,
                "smolnalysis.hf.message_count": len(messages),
            },
        ) as span:
            self.model.eval()
            device = get_input_device(self.model)
            logger.info(
                "generation started: message_count=%d active_adapter=%s auto_adapter=%s force_base=%s max_new_tokens=%d temperature=%s top_p=%s top_k=%s",
                len(messages),
                self.active_adapter or "base",
                self.auto_adapter,
                force_base,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            )

            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[-1]
            if span is not None:
                span.set_attribute("gen_ai.usage.input_tokens", prompt_len)

            stop_ids = [self.tokenizer.eos_token_id]
            turn_end_id = self.tokenizer.convert_tokens_to_ids("<turn|>")
            if isinstance(turn_end_id, int) and turn_end_id >= 0:
                stop_ids.append(turn_end_id)

            do_sample = temperature > 0
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "eos_token_id": stop_ids,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
                generation_kwargs["top_k"] = top_k

            with torch.no_grad():
                use_base = force_base or self.active_adapter is None
                if use_base and hasattr(self.model, "disable_adapter"):
                    with self.model.disable_adapter():
                        generated = self.model.generate(**inputs, **generation_kwargs)
                else:
                    generated = self.model.generate(**inputs, **generation_kwargs)

            new_tokens = generated[0, prompt_len:]
            if span is not None:
                output_tokens = len(new_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                span.set_attribute("gen_ai.response.finish_reasons", "stop")
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            response = decoded.replace("<turn|>", "").replace("<eos>", "").strip()
            logger.info(
                "generation finished: active_adapter=%s force_base=%s input_tokens=%d output_tokens=%d response_chars=%d",
                self.active_adapter or "base",
                force_base,
                prompt_len,
                len(new_tokens),
                len(response),
            )
            return response


@lru_cache(maxsize=1)
def get_gemma_model(
    base_model_id: str = BASE_MODEL_ID,
    initial_adapter: str | None = DEFAULT_ADAPTER,
) -> GemmaAdapterModel:
    return GemmaAdapterModel(base_model_id=base_model_id, initial_adapter=initial_adapter)


def _latest_user_message(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    last_user_text = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    return [{"role": "user", "content": last_user_text}] if last_user_text else []


def generate_chat_response(
    messages: list[dict[str, str]],
    *,
    adapter: str | None = DEFAULT_ADAPTER,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    logger.info("chat generation requested: adapter=%s messages=%d", adapter or "base", len(messages))
    runner = get_gemma_model(initial_adapter=adapter)
    last_user_messages = _latest_user_message(messages)
    logger.info(f"last user message: {last_user_messages}")

    if runner.auto_adapter:
        last_user_text = last_user_messages[0]["content"] if last_user_messages else ""
        routed_adapter = runner.route_adapter(last_user_text)
        runner.set_adapter(routed_adapter)
        runner.auto_adapter = True
        logger.info("auto adapter selected: routed_adapter=%s", routed_adapter)
    else:
        logger.info("static adapter selected: active_adapter=%s", runner.active_adapter or "base")

    generation_messages = messages if runner.active_adapter is None else last_user_messages
    logger.info(
        "chat prompt selected: active_adapter=%s prompt_messages=%d original_messages=%d history_included=%s",
        runner.active_adapter or "base",
        len(generation_messages),
        len(messages),
        runner.active_adapter is None,
    )

    return runner.generate(
        generation_messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def print_adapters() -> None:
    print("Registered adapters:")
    print(f"  {'auto':10} {'ready':7} base-routes-to-retrieval-or-base")
    print(f"  {'base':10} {'ready':7} {BASE_MODEL_ID} (no adapter)")
    for adapter in list_adapters():
        status = "ready" if adapter.exists else "missing"
        print(f"  {adapter.name:10} {status:7} {adapter.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--list-adapters", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_adapters:
        print_adapters()
        return

    print_adapters()
    if GemmaAdapterModel._is_auto_adapter(args.adapter):
        adapter_label = "auto"
    elif GemmaAdapterModel._is_base_adapter(args.adapter):
        adapter_label = "no adapter"
    else:
        adapter_label = args.adapter
    print(f"\nLoading base model with adapter: {adapter_label}")
    runner = GemmaAdapterModel(
        base_model_id=args.base_model_id,
        initial_adapter=args.adapter,
    )

    print("\nInteractive chat ready.")
    print("Type /exit to quit, /reset to clear history, /adapter NAME to switch adapters.")
    print("Use /adapter auto to let the base model route each message.")
    print("Use /adapter base to chat without an adapter.\n")

    messages: list[dict[str, str]] = []

    while True:
        prompt_name = "auto" if runner.auto_adapter else runner.active_adapter or "base"
        user_text = input(f"{prompt_name}> ").strip()
        if not user_text:
            continue

        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Bye.")
            break

        if user_text.lower() == "/reset":
            messages = []
            print("Conversation reset.\n")
            continue

        if user_text.lower() in {"/adapters", "/adapter"}:
            print_adapters()
            print()
            continue

        if user_text.startswith("/adapter "):
            adapter_name = user_text.split(maxsplit=1)[1].strip()
            try:
                runner.set_adapter(adapter_name)
            except (FileNotFoundError, KeyError) as exc:
                print(f"{exc}\n")
            else:
                messages = []
                active = "auto" if runner.auto_adapter else runner.active_adapter or "base"
                print(f"Active adapter: {active}\n")
            continue

        if runner.auto_adapter:
            adapter_name = runner.route_adapter(user_text)
            try:
                runner.set_adapter(adapter_name)
            except (FileNotFoundError, KeyError) as exc:
                print(f"{exc}\n")
                runner.set_adapter("base")
            else:
                runner.auto_adapter = True
                print(f"[auto -> {adapter_name}]")

        messages.append({"role": "user", "content": user_text})
        assistant_text = runner.generate(messages)
        print(f"Assistant: {assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
