from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .adapter_registry import adapter_source
except ImportError:
    from adapter_registry import adapter_source  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_ID = os.getenv("SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID", os.getenv("MODEL_ID", "openbmb/MiniCPM5-1B"))
ROUTER_LABEL_TO_ADAPTER = {
    "general_agent": None,
    "ckan_retrieval": "ckan_retrieval",
    "openui_translator": "openui_translator",
}
RETRIEVAL_SYSTEM_PROMPT = (
    "You generate the structured tool result for a smolnalysis user question. "
    "Return only valid JSON. Do not include explanations, markdown, or any label prefix."
)
OPENUI_SYSTEM_PROMPT = (
    "You generate OpenUI Lang from a user query and a structured tool result. "
    "Use only the values from the tool result. Do not invent data. "
    "Return only OpenUI Lang assignment statements, without explanations or markdown. "
    "Start with root = Root([...])."
)
GREETING_RESPONSE = "hi, there how can i help you?"


@dataclass(frozen=True)
class AdapterSource:
    name: str
    source: str
    is_path: bool


def _repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _adapter_source_for_role(role: str) -> AdapterSource | None:
    source = adapter_source(role)
    if not source:
        return None
    return AdapterSource(role, source, _looks_like_path(source))


def _looks_like_path(source: str) -> bool:
    return source.startswith(("/", "./", "../", "~")) or Path(source).expanduser().exists()


class SmolnalysisMoE(torch.nn.Module):
    """Small inference wrapper for the fixed smolnalysis adapter workflow."""

    def __init__(
        self,
        model_base_name: str = BASE_MODEL_ID,
        *,
        load_in_4bit: bool = True,
    ) -> None:
        super().__init__()
        self.model_base_name = model_base_name
        self.load_in_4bit = load_in_4bit
        self.loaded_adapters: set[str] = set()
        self.active_adapter: str | None = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_base = self.load_model_base(model_base_name)
        self.vocab_size = len(self.tokenizer)

    def _build_quantization_config(self):
        if not self.load_in_4bit:
            return None

        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def load_model_base(self, model_base_name: str):
        quantization_config = self._build_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_base_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        model.eval()
        return model

    def _normalize_messages(self, inputs: Any) -> list[dict[str, str]]:
        if isinstance(inputs, str):
            return [{"role": "user", "content": inputs}]
        if isinstance(inputs, list):
            return [{"role": str(message["role"]), "content": str(message["content"])} for message in inputs]
        if isinstance(inputs, dict):
            if "messages" in inputs:
                return self._normalize_messages(inputs["messages"])
            if "prompt" in inputs:
                return [{"role": "user", "content": str(inputs["prompt"])}]
            if "content" in inputs:
                return [{"role": str(inputs.get("role", "user")), "content": str(inputs["content"])}]
        raise TypeError("inputs must be tokenized features, a message list, a prompt string, or a dict with messages/prompt")

    @staticmethod
    def _latest_user_message(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                return [{"role": "user", "content": message["content"]}]
        return []

    def _tokenize_messages(self, messages: list[dict[str, str]], *, max_length: int | None = None) -> dict[str, Any]:
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if max_length is not None:
            tokenized["input_ids"] = tokenized["input_ids"][:, -max_length:]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, -max_length:]
        return tokenized

    def _preprocess(self, inputs: Any) -> dict[str, Any]:
        messages = self._normalize_messages(inputs)
        latest_user_messages = self._latest_user_message(messages)
        return {
            "messages": messages,
            "latest_user_messages": latest_user_messages,
        }

    @staticmethod
    def _is_preprocessed(inputs: Any) -> bool:
        return isinstance(inputs, dict) and "input_ids" in inputs

    def _adapter_from_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        label = label.strip()
        try:
            return ROUTER_LABEL_TO_ADAPTER[label]
        except KeyError as exc:
            available = ", ".join(ROUTER_LABEL_TO_ADAPTER)
            raise KeyError(f"Unknown router label '{label}'. Expected one of: {available}") from exc

    def adapter_source_for_role(self, role: str | None) -> AdapterSource | None:
        adapter_name = self._adapter_from_label(role)
        return _adapter_source_for_role(adapter_name) if adapter_name else None

    def set_adapter(self, adapter_name: str | None) -> None:
        adapter_name = self._adapter_from_label(adapter_name)
        if adapter_name is None:
            self.active_adapter = None
            self.model_base.eval()
            return

        adapter = _adapter_source_for_role(adapter_name)
        if adapter is None:
            raise KeyError(f"No MiniCPM adapter source configured for role '{adapter_name}'")
        if adapter.is_path:
            adapter_path = Path(adapter.source).expanduser()
            if not adapter_path.exists():
                raise FileNotFoundError(f"MiniCPM adapter path does not exist for role '{adapter.name}': {adapter_path}")
            adapter_source = str(adapter_path)
        else:
            adapter_source = adapter.source

        from peft import PeftModel

        if not self.loaded_adapters:
            self.model_base = PeftModel.from_pretrained(self.model_base, adapter_source, adapter_name=adapter.name)
            self.loaded_adapters.add(adapter.name)
        elif adapter.name not in self.loaded_adapters:
            self.model_base.load_adapter(adapter_source, adapter_name=adapter.name)
            self.loaded_adapters.add(adapter.name)

        self.model_base.set_adapter(adapter.name)
        self.model_base.eval()
        self.active_adapter = adapter.name

    def _generation_inputs(self, preprocessed: dict[str, Any]) -> dict[str, Any]:
        messages = preprocessed.get("messages") or preprocessed.get("latest_user_messages") or []
        return self._tokenize_messages(messages)

    def forward(
        self,
        inputs: Any,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 64,
        adapter: str | None = None,
    ):
        already_tokenized = isinstance(inputs, dict) and "input_ids" in inputs
        preprocessed = inputs if self._is_preprocessed(inputs) else self._preprocess(inputs)
        selected_adapter = self._adapter_from_label(adapter)
        self.set_adapter(selected_adapter)

        model_inputs = preprocessed if already_tokenized else self._generation_inputs(preprocessed)
        device = next(self.model_base.parameters()).device
        model_inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in model_inputs.items()}
        input_tokens = int(model_inputs["input_ids"].shape[-1])
        generation_kwargs: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_k"] = top_k

        with torch.inference_mode():
            use_base = self.active_adapter is None and hasattr(self.model_base, "disable_adapter")
            if use_base:
                with self.model_base.disable_adapter():
                    outputs = self.model_base.generate(**generation_kwargs)
            else:
                outputs = self.model_base.generate(**generation_kwargs)
        return outputs[:, input_tokens:]

    def generate_text(self, inputs: Any, **generation_kwargs: Any) -> str:
        output_ids = self.forward(inputs, **generation_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    @staticmethod
    def _is_greeting(user_message: str) -> bool:
        return user_message.strip().casefold() == "hi"

    def generate_chat(
        self,
        inputs: Any,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> dict[str, Any]:
        messages = self._normalize_messages(inputs)
        latest_user = self._latest_user_message(messages)
        user_message = latest_user[0]["content"] if latest_user else ""

        if self._is_greeting(user_message):
            return {
                "content": GREETING_RESPONSE,
                "tool_result": "",
                "stages": [{"adapter": None, "input": "hardcoded_greeting"}],
            }

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        retrieval_messages = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_result = self.generate_text(retrieval_messages, adapter="ckan_retrieval", **generation_kwargs)
        openui_messages = [
            {"role": "system", "content": OPENUI_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_message}\n\nTool result:\n{tool_result}"},
        ]
        openui_lang = self.generate_text(openui_messages, adapter="openui_translator", **generation_kwargs)
        return {
            "content": openui_lang,
            "tool_result": tool_result,
            "stages": [
                {"adapter": "ckan_retrieval", "input": "user_message"},
                {"adapter": "openui_translator", "input": "user_message_and_tool_result"},
            ],
        }
