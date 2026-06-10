import json
from pathlib import Path


def load_tool_examples(path):
    """Return (user_query, assistant_json) examples from llm_user_queries.jsonl."""
    examples = []

    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            query = row.get("query")
            target = row.get("search_target")

            if not query or not target:
                raise ValueError(f"Missing query/search_target in {path}:{line_no}")

            examples.append(
                {
                    "user": query,
                    "assistant": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                }
            )

    return examples
