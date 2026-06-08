from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_LLM_TIMEOUT_SECONDS = 8.0


class LlmSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    base_url: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_BASE_URL")
    api_key: SecretStr | None = Field(default=None, validation_alias="SMOLNALYSIS_LLM_API_KEY")
    timeout_seconds: float = Field(default=DEFAULT_LLM_TIMEOUT_SECONDS, validation_alias="SMOLNALYSIS_LLM_TIMEOUT_SECONDS")

    general_agent_model: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_GENERAL_AGENT_MODEL")
    ckan_tool_model: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_CKAN_TOOL_MODEL")
    data_analysis_model: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_DATA_ANALYSIS_MODEL")
    openui_translator_model: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_OPENUI_TRANSLATOR_MODEL")

    general_agent_base_url: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_GENERAL_AGENT_BASE_URL")
    ckan_tool_base_url: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_CKAN_TOOL_BASE_URL")
    data_analysis_base_url: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_DATA_ANALYSIS_BASE_URL")
    openui_translator_base_url: str = Field(default="", validation_alias="SMOLNALYSIS_LLM_OPENUI_TRANSLATOR_BASE_URL")

    general_agent_api_key: SecretStr | None = Field(default=None, validation_alias="SMOLNALYSIS_LLM_GENERAL_AGENT_API_KEY")
    ckan_tool_api_key: SecretStr | None = Field(default=None, validation_alias="SMOLNALYSIS_LLM_CKAN_TOOL_API_KEY")
    data_analysis_api_key: SecretStr | None = Field(default=None, validation_alias="SMOLNALYSIS_LLM_DATA_ANALYSIS_API_KEY")
    openui_translator_api_key: SecretStr | None = Field(default=None, validation_alias="SMOLNALYSIS_LLM_OPENUI_TRANSLATOR_API_KEY")

    @field_validator("timeout_seconds")
    @classmethod
    def _minimum_timeout(cls, value: float) -> float:
        return max(1.0, value)


@dataclass(frozen=True)
class LlmRole:
    key: str
    label: str
    description: str
    model_attr: str
    base_url_attr: str
    api_key_attr: str


@dataclass
class LlmRoleStatus:
    key: str
    label: str
    description: str
    configured: bool
    base_url: str
    base_url_display: str
    model: str
    validation_status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LLM_ROLES = [
    LlmRole(
        "general_agent",
        "General agentic",
        "Plans the overall CKAN, analysis, and OpenUI workflow.",
        "general_agent_model",
        "general_agent_base_url",
        "general_agent_api_key",
    ),
    LlmRole(
        "ckan_tool",
        "CKAN tool calling",
        "Reasons over CKAN search and resource-discovery tool calls.",
        "ckan_tool_model",
        "ckan_tool_base_url",
        "ckan_tool_api_key",
    ),
    LlmRole(
        "data_analysis",
        "Data analysis",
        "Analyzes loaded dataset/resource data.",
        "data_analysis_model",
        "data_analysis_base_url",
        "data_analysis_api_key",
    ),
    LlmRole(
        "openui_translator",
        "OpenUI translator",
        "Converts analysis results into OpenUI-Lang.",
        "openui_translator_model",
        "openui_translator_base_url",
        "openui_translator_api_key",
    ),
]


def llm_status() -> dict[str, Any]:
    settings = load_llm_settings()
    return {
        "roles": [role_status(role, settings).to_dict() for role in LLM_ROLES],
        "timeout_seconds": settings.timeout_seconds,
    }


def validate_llms() -> dict[str, Any]:
    settings = load_llm_settings()
    return {
        "roles": [validate_role(role, settings).to_dict() for role in LLM_ROLES],
        "timeout_seconds": settings.timeout_seconds,
    }


def load_llm_settings() -> LlmSettings:
    return LlmSettings()


def role_status(role: LlmRole, settings: LlmSettings | None = None) -> LlmRoleStatus:
    settings = settings or load_llm_settings()
    base_url = _role_base_url(role, settings)
    model = _role_model(role, settings)
    api_key = _role_api_key(role, settings)
    configured = bool(base_url and model and api_key)
    missing = []
    if not base_url:
        missing.append("base URL")
    if not model:
        missing.append("model")
    if not api_key:
        missing.append("API key")

    return LlmRoleStatus(
        key=role.key,
        label=role.label,
        description=role.description,
        configured=configured,
        base_url=base_url,
        base_url_display=_display_base_url(base_url),
        model=model,
        validation_status="not_checked" if configured else "missing",
        message="Configured. Validation has not run." if configured else f"Missing {', '.join(missing)}.",
    )


def validate_role(role: LlmRole, settings: LlmSettings | None = None) -> LlmRoleStatus:
    settings = settings or load_llm_settings()
    status = role_status(role, settings)
    if not status.configured:
        return status

    url = _models_url(status.base_url)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {_role_api_key(role, settings)}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 405, 501}:
            status.validation_status = "unvalidated"
            status.message = "Configured, but this provider does not expose /v1/models."
        else:
            status.validation_status = "error"
            status.message = f"Validation failed with HTTP {exc.code}."
        return status
    except (TimeoutError, urllib.error.URLError, OSError):
        status.validation_status = "error"
        status.message = "Could not reach the OpenAI-compatible provider."
        return status

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        status.validation_status = "error"
        status.message = "Provider returned invalid JSON from /v1/models."
        return status

    model_ids = _model_ids(payload)
    if model_ids is None:
        status.validation_status = "unvalidated"
        status.message = "Configured, but /v1/models returned an unexpected shape."
    elif status.model in model_ids:
        status.validation_status = "valid"
        status.message = "Configured and model was found in /v1/models."
    else:
        status.validation_status = "unvalidated"
        status.message = "Configured, but the model was not listed by /v1/models."
    return status


def _role_base_url(role: LlmRole, settings: LlmSettings) -> str:
    return _clean_base_url(getattr(settings, role.base_url_attr) or settings.base_url)


def _role_api_key(role: LlmRole, settings: LlmSettings) -> str:
    secret = getattr(settings, role.api_key_attr) or settings.api_key
    return secret.get_secret_value().strip() if secret else ""


def _role_model(role: LlmRole, settings: LlmSettings) -> str:
    return str(getattr(settings, role.model_attr)).strip()


def _clean_base_url(raw_url: str) -> str:
    value = raw_url.strip().rstrip("/")
    if not value:
        return ""
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme:
        parsed = urllib.parse.urlsplit(f"https://{value}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _display_base_url(base_url: str) -> str:
    if not base_url:
        return ""
    parsed = urllib.parse.urlsplit(base_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/models"


def _model_ids(payload: Any) -> set[str] | None:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        ids = {item.get("id") for item in payload["data"] if isinstance(item, dict) and isinstance(item.get("id"), str)}
        return ids
    if isinstance(payload, list):
        ids = {item.get("id") for item in payload if isinstance(item, dict) and isinstance(item.get("id"), str)}
        return ids
    return None
