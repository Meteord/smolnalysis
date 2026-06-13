from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_ADAPTER_DIR = Path("train/ckan/adapters/smolnalysis-ckan-retrieval-minicpm5-lora")
DEFAULT_REPO_ID = "build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora"
ALLOW_PATTERNS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "eval_metrics.json",
    "README.md",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload the CKAN retrieval PEFT adapter to Hugging Face Hub.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--repo-id", default=os.getenv("SMOLNALYSIS_CKAN_ADAPTER_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload smolnalysis CKAN retrieval MiniCPM5 LoRA adapter")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_adapter_dir(adapter_dir: Path) -> None:
    missing = [name for name in ["adapter_config.json", "adapter_model.safetensors"] if not (adapter_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Adapter directory {adapter_dir} is missing required files: {joined}")


def main() -> int:
    args = build_arg_parser().parse_args()
    adapter_dir = args.adapter_dir.resolve()
    validate_adapter_dir(adapter_dir)

    print(f"Adapter: {adapter_dir}")
    print(f"Repo:    {args.repo_id}")
    print("Files:")
    for pattern in ALLOW_PATTERNS:
        path = adapter_dir / pattern
        if path.exists():
            print(f"  - {pattern} ({path.stat().st_size} bytes)")

    if args.dry_run:
        print("Dry run only; no upload performed.")
        return 0

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN with write access before uploading.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=ALLOW_PATTERNS,
        commit_message=args.commit_message,
    )
    print(f"Uploaded adapter to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
