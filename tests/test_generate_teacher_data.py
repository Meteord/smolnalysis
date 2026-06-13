from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from generate_teacher_data import (
    TeacherConfig,
    build_teacher_messages,
    build_training_example_from_teacher,
    call_teacher,
    call_teacher_with_retries,
    generate_examples,
    read_existing_examples,
    teacher_config_from_env,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class GenerateTeacherDataTests(TestCase):
    def test_teacher_config_reads_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "SMOLNALYSIS_TEACHER_BASE_URL=http://teacher.local/v1",
                        "SMOLNALYSIS_TEACHER_API_KEY=test-key",
                        "SMOLNALYSIS_TEACHER_MODEL=test-model",
                        "SMOLNALYSIS_TEACHER_TIMEOUT_SECONDS=12",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = teacher_config_from_env(0.2, str(env_file))

        self.assertEqual(config.base_url, "http://teacher.local/v1")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.timeout_seconds, 12)
        self.assertEqual(config.temperature, 0.2)

    def test_build_teacher_messages_include_scenario(self) -> None:
        messages = build_teacher_messages(
            {
                "request": "Find schools.",
                "endpoint": "https://opendata.muenchen.de/",
                "observed_packages": ["schools"],
                "observed_resources": [],
                "has_enough_evidence": False,
            }
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("strict JSON", messages[0]["content"])
        self.assertIn("tag_search", messages[0]["content"])
        self.assertIn("organization_list", messages[0]["content"])
        self.assertIn("Find schools.", messages[1]["content"])
        self.assertIn("schools", messages[1]["content"])
        self.assertIn("Observed tags", messages[1]["content"])
        self.assertIn("Tool errors", messages[1]["content"])

    def test_call_teacher_posts_openai_compatible_request(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "thought": "Need search.",
                                        "action": "package_search",
                                        "args": {"query": "schools", "rows": 5, "start": 0},
                                        "confidence": 0.7,
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        content = call_teacher(
            TeacherConfig("http://teacher.local/v1", "key", "model", 9, 0.3),
            [{"role": "user", "content": "hello"}],
            urlopen=fake_urlopen,
        )

        self.assertEqual(captured["url"], "http://teacher.local/v1/chat/completions")
        self.assertEqual(captured["timeout"], 9)
        self.assertEqual(captured["body"]["model"], "model")
        self.assertEqual(json.loads(content)["action"], "package_search")

    def test_generate_examples_builds_valid_training_examples(self) -> None:
        scenario = {
            "id": "schools",
            "request": "Find schools.",
            "observed_packages": [],
            "observed_resources": [],
            "has_enough_evidence": False,
        }

        def fake_urlopen(_request, timeout):
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "thought": "Need an initial search.",
                                        "action": "package_search",
                                        "args": {"query": "schools", "rows": 5, "start": 0},
                                        "confidence": 0.7,
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        examples = generate_examples([scenario], TeacherConfig("http://teacher.local/v1", "key", "model"), urlopen=fake_urlopen, show_progress=True)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["metadata"]["scenario_id"], "schools")
        self.assertEqual(json.loads(examples[0]["messages"][-1]["content"])["action"], "package_search")

    def test_call_teacher_retries_after_runtime_error(self) -> None:
        calls = {"count": 0}

        def fake_urlopen(_request, timeout):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("slow")
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "thought": "Need search.",
                                        "action": "package_search",
                                        "args": {"query": "schools", "rows": 5, "start": 0},
                                        "confidence": 0.7,
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        content = call_teacher_with_retries(
            TeacherConfig("http://teacher.local/v1", "key", "model"),
            [{"role": "user", "content": "hello"}],
            retries=1,
            retry_delay_seconds=0,
            urlopen=fake_urlopen,
        )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(json.loads(content)["action"], "package_search")

    def test_generate_examples_skips_existing_on_resume(self) -> None:
        existing = build_training_example_from_teacher(
            {"id": "done", "request": "Find schools."},
            json.dumps(
                {
                    "thought": "Need search.",
                    "action": "package_search",
                    "args": {"query": "schools", "rows": 5, "start": 0},
                    "confidence": 0.7,
                }
            ),
        )
        calls = {"count": 0}

        def fake_urlopen(_request, timeout):
            calls["count"] += 1
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "thought": "Need another search.",
                                        "action": "package_search",
                                        "args": {"query": "parks", "rows": 5, "start": 0},
                                        "confidence": 0.7,
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        examples = generate_examples(
            [{"id": "done", "request": "Find schools."}, {"id": "new", "request": "Find parks."}],
            TeacherConfig("http://teacher.local/v1", "key", "model"),
            existing_examples=[existing],
            urlopen=fake_urlopen,
        )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(examples), 2)

    def test_read_existing_examples_ignores_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(read_existing_examples(Path(tmpdir) / "missing.jsonl"), [])

    def test_build_training_example_preserves_context(self) -> None:
        example = build_training_example_from_teacher(
            {
                "id": "pkg",
                "request": "Inspect package.",
                "observed_packages": ["known"],
                "observed_resources": [],
                "observed_tags": ["Schulen"],
                "observed_groups": ["educ"],
                "observed_organizations": ["referat-fuer-bildung-und-sport"],
                "has_enough_evidence": False,
            },
            json.dumps(
                {
                    "thought": "Inspect observed package.",
                    "action": "package_show",
                    "args": {"package_id": "known"},
                    "confidence": 0.8,
                }
            ),
        )

        self.assertEqual(example["metadata"]["ckan_context"]["observed_packages"], ["known"])
        self.assertEqual(example["metadata"]["ckan_context"]["observed_tags"], ["Schulen"])
        self.assertEqual(example["metadata"]["ckan_context"]["observed_groups"], ["educ"])
        self.assertEqual(example["messages"][-1]["role"], "assistant")


if __name__ == "__main__":
    main()
