from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTER_SOURCE = REPO_ROOT / "train" / "router" / "router_mlp.py"
DEFAULT_ROUTER_OUTPUT_DIR = Path(
    os.getenv("SMOLNALYSIS_ROUTER_OUTPUT_DIR", str(REPO_ROOT / "train" / "router" / "outputs" / "router-mlp"))
)
BASE_MODEL_ID = os.getenv("SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID", os.getenv("MODEL_ID", "openbmb/MiniCPM5-1B"))
DEFAULT_OPENUI_TRANSLATOR_ADAPTER_PATH = Path(
    os.getenv(
        "SMOLNALYSIS_DEFAULT_OPENUI_TRANSLATOR_ADAPTER_PATH",
        str(REPO_ROOT / "train" / "openui_lang" / "outputs" / "openui-sft-stats-components-lora"),
    )
)
BASE_ADAPTER_NAMES = {"", "base", "none", "no_adapter", "no-adapter", "general", "general_agent"}
ROLE_ENV_KEYS = {
    "general_agent": "GENERAL_AGENT",
    "ckan_retrieval": "CKAN_RETRIEVAL",
    "data_analysis": "DATA_ANALYSIS",
    "openui_translator": "OPENUI_TRANSLATOR",
}
DEFAULT_LABEL_TO_ADAPTER = {
    "general_agent": None,
    "base": None,
    "none": None,
    "ckan": "ckan_retrieval",
    "retrieval": "ckan_retrieval",
    "ckan_retrieval": "ckan_retrieval",
    "openui": "openui_translator",
    "openui_translator": "openui_translator",
    "analysis": "data_analysis",
    "data_analysis": "data_analysis",
}


@dataclass(frozen=True)
class AdapterSource:
    name: str
    source: str
    is_path: bool


@dataclass(frozen=True)
class RouterDecision:
    role: str
    adapter: str | None
    confidence: float
    logits: list[float]


def _clean_env_value(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    lines = []
    for line in str(raw).splitlines():
        value = line.strip().strip('"').strip("'")
        if value and not value.startswith("#"):
            lines.append(value)
    return lines[-1] if lines else default


def _role_env(role: str, suffix: str) -> str:
    return f"SMOLNALYSIS_MINICPM_{ROLE_ENV_KEYS[role]}_{suffix}"


def _repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _adapter_source_for_role(role: str) -> AdapterSource | None:
    if role not in ROLE_ENV_KEYS:
        return None

    default_path = str(DEFAULT_OPENUI_TRANSLATOR_ADAPTER_PATH) if role == "openui_translator" else ""
    adapter_path = _clean_env_value(_role_env(role, "ADAPTER_PATH"), _clean_env_value(_role_env(role, "LORA_PATH"), default_path))
    adapter_repo_id = _clean_env_value(
        _role_env(role, "ADAPTER_REPO_ID"),
        _clean_env_value(_role_env(role, "LORA_REPO_ID"), ""),
    )
    if adapter_path:
        return AdapterSource(role, adapter_path, True)
    if adapter_repo_id:
        return AdapterSource(role, adapter_repo_id, False)
    return None


class SmolnalysisMoE(torch.nn.Module):
    """Small inference wrapper for router-selected LoRA adapters."""

    def __init__(
        self,
        model_base_name: str = BASE_MODEL_ID,
        *,
        task_router: torch.nn.Module | None = None,
        router_labels: list[str] | None = None,
        label_to_adapter: dict[str, str | None] | None = None,
        load_in_4bit: bool = True,
        load_task_router: bool = True,
        router_output_dir: str | Path = DEFAULT_ROUTER_OUTPUT_DIR,
        router_max_length: int = 512,
    ) -> None:
        super().__init__()
        self.model_base_name = model_base_name
        self.load_in_4bit = load_in_4bit
        self.router_output_dir = _repo_path(router_output_dir)
        self.router_max_length = router_max_length
        self.task_router = task_router
        self.router_labels = router_labels
        self.label_to_adapter = {**DEFAULT_LABEL_TO_ADAPTER, **(label_to_adapter or {})}
        self.loaded_adapters: set[str] = set()
        self.active_adapter: str | None = None
        self.last_router_decision: RouterDecision | None = None

        if self.task_router is None and load_task_router:
            self.task_router, loaded_labels = self.load_task_router(self.router_output_dir)
            if self.router_labels is None:
                self.router_labels = loaded_labels
        if self.router_labels is None:
            self.router_labels = ["general_agent", "ckan_retrieval", "openui_translator"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_base = self.load_model_base(model_base_name)
        self.vocab_size = len(self.tokenizer)

    @staticmethod
    def load_task_router(output_dir: str | Path) -> tuple[torch.nn.Module, list[str]]:
        output_dir = _repo_path(output_dir)
        if not (output_dir / "router_mlp.pt").exists() or not (output_dir / "config.json").exists():
            raise FileNotFoundError(f"Router artifacts are missing in {output_dir}")
        spec = importlib.util.spec_from_file_location("smolnalysis_router_mlp", ROUTER_SOURCE)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load router module from {ROUTER_SOURCE}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        router, config = module.load_router_mlp(output_dir)
        return router, list(config.labels)

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
        router_messages = latest_user_messages or messages
        return {
            "messages": messages,
            "latest_user_messages": latest_user_messages,
            "router_inputs": self._tokenize_messages(router_messages, max_length=self.router_max_length),
        }

    @staticmethod
    def _is_preprocessed(inputs: Any) -> bool:
        return isinstance(inputs, dict) and ("input_ids" in inputs or "router_inputs" in inputs)

    def _adapter_from_label(self, label: str | None) -> str | None:
        adapter_name = self.label_to_adapter.get((label or "").strip().casefold(), label)
        if adapter_name is None:
            return None
        adapter_name = adapter_name.strip().casefold()
        return None if adapter_name in BASE_ADAPTER_NAMES else adapter_name

    def adapter_source_for_role(self, role: str | None) -> AdapterSource | None:
        adapter_name = self._adapter_from_label(role)
        return _adapter_source_for_role(adapter_name) if adapter_name else None

    def route(self, inputs: Any) -> tuple[dict[str, Any], RouterDecision | None]:
        preprocessed = inputs if self._is_preprocessed(inputs) else self._preprocess(inputs)
        self.route_adapter(preprocessed)
        return preprocessed, self.last_router_decision

    def route_adapter(self, preprocessed: dict[str, Any]) -> str | None:
        if self.task_router is None:
            self.last_router_decision = None
            return None

        router_features = preprocessed.get("router_inputs", preprocessed)
        try:
            router_device = next(self.task_router.parameters()).device
        except StopIteration:
            router_device = torch.device("cpu")
        router_inputs = {
            key: value.to(router_device) if torch.is_tensor(value) else value
            for key, value in router_features.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            output = self.task_router(**router_inputs)
        logits = output["logits"] if isinstance(output, dict) else output.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        label_index = int(logits.argmax(dim=-1).item())
        label = self.router_labels[label_index]
        adapter = self._adapter_from_label(label)
        self.last_router_decision = RouterDecision(
            role=label,
            adapter=adapter,
            confidence=float(probabilities[label_index].item()),
            logits=[float(value) for value in logits[0].detach().cpu().tolist()],
        )
        return adapter

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
        if self.active_adapter is None:
            messages = preprocessed.get("messages") or preprocessed.get("latest_user_messages") or []
        else:
            messages = preprocessed.get("latest_user_messages") or preprocessed.get("messages") or []
        return self._tokenize_messages(messages)

    def forward(
        self,
        inputs: Any,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 64,
        adapter: str | None = "auto",
    ):
        already_tokenized = isinstance(inputs, dict) and "input_ids" in inputs
        preprocessed = inputs if self._is_preprocessed(inputs) else self._preprocess(inputs)
        requested_adapter = (adapter or "auto").strip().casefold() if isinstance(adapter, str) else adapter
        selected_adapter = self._adapter_from_label(adapter)
        if requested_adapter in {None, "auto", "router"}:
            selected_adapter = self.route_adapter(preprocessed)
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
