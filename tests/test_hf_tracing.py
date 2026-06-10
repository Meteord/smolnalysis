from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import hf_tracing


class HuggingFaceTracingTests(TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()
        _clear_caches()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)
        _clear_caches()

    def test_tracing_is_disabled_by_default(self) -> None:
        with patch.object(hf_tracing, "_get_tracer") as get_tracer:
            with hf_tracing.huggingface_span("model.generate") as span:
                span.set_attribute("key", "value")

        get_tracer.assert_not_called()

    def test_enabled_span_sets_initial_attributes(self) -> None:
        os.environ["SMOLNALYSIS_HF_TRACING_ENABLED"] = "true"
        tracer = _FakeTracer()

        with patch.object(hf_tracing, "_get_tracer", return_value=tracer):
            with hf_tracing.huggingface_span("model.generate", {"gen_ai.system": "huggingface"}) as span:
                span.set_attribute("gen_ai.usage.output_tokens", 12)

        self.assertEqual(tracer.span_name, "huggingface.model.generate")
        self.assertEqual(tracer.span.attributes["gen_ai.system"], "huggingface")
        self.assertEqual(tracer.span.attributes["gen_ai.usage.output_tokens"], 12)


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self) -> None:
        self.span = _FakeSpan()
        self.span_name = ""

    @contextmanager
    def start_as_current_span(self, name: str):
        self.span_name = name
        yield self.span


def _clear_caches() -> None:
    hf_tracing.huggingface_tracing_enabled.cache_clear()
    hf_tracing._get_tracer.cache_clear()
    hf_tracing._configure_tracer_provider.cache_clear()


if __name__ == "__main__":
    main()
