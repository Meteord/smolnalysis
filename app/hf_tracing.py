from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator


LOGGER = logging.getLogger(__name__)
TRUTHY_VALUES = {"1", "true", "yes", "on"}


class NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


def _is_enabled(value: str | None) -> bool:
    return (value or "").strip().casefold() in TRUTHY_VALUES


@lru_cache(maxsize=1)
def huggingface_tracing_enabled() -> bool:
    return _is_enabled(os.getenv("SMOLNALYSIS_HF_TRACING_ENABLED"))


@lru_cache(maxsize=1)
def _get_tracer() -> Any:
    try:
        from opentelemetry import trace
    except ImportError:
        LOGGER.warning("Hugging Face tracing is enabled, but opentelemetry-api is not installed.")
        return None

    _configure_tracer_provider(trace)
    return trace.get_tracer("smolnalysis.huggingface")


@lru_cache(maxsize=1)
def _configure_tracer_provider(trace: Any) -> None:
    endpoint = os.getenv("SMOLNALYSIS_HF_TRACING_OTLP_ENDPOINT", "").strip()
    console_enabled = _is_enabled(os.getenv("SMOLNALYSIS_HF_TRACING_CONSOLE"))
    if not endpoint and not console_enabled:
        return

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        LOGGER.warning("Hugging Face tracing exporter requested, but opentelemetry-sdk is not installed.")
        return

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": os.getenv("SMOLNALYSIS_HF_TRACING_SERVICE_NAME", "smolnalysis"),
                "service.namespace": "smolnalysis",
            }
        )
    )
    if console_enabled:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            LOGGER.warning("OTLP tracing endpoint configured, but opentelemetry-exporter-otlp is not installed.")
        else:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    try:
        trace.set_tracer_provider(provider)
    except Exception as exc:
        LOGGER.warning("Could not configure Hugging Face tracer provider: %s", exc)


@contextmanager
def huggingface_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    if not huggingface_tracing_enabled():
        yield NoOpSpan()
        return

    tracer = _get_tracer()
    if tracer is None:
        yield NoOpSpan()
        return

    with tracer.start_as_current_span(f"huggingface.{name}") as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        yield span
