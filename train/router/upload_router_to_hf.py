from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_ROUTER_DIR = Path("train/router/outputs/router-mlp")
DEFAULT_REPO_ID = "build-small-hackathon/smolnalysis-adapter-router"
ALLOW_PATTERNS = [
    "config.json",
    "router_mlp.pt",
    "metrics.json",
    "README.md",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload the smolnalysis router artifacts to Hugging Face Hub.")
    parser.add_argument("--router-dir", "--adapter-dir", dest="router_dir", type=Path, default=DEFAULT_ROUTER_DIR)
    parser.add_argument("--repo-id", default=os.getenv("SMOLNALYSIS_ROUTER_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload smolnalysis router artifacts")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_router_dir(router_dir: Path) -> None:
    missing = [name for name in ["config.json", "router_mlp.pt"] if not (router_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Router directory {router_dir} is missing required files: {joined}")


def main() -> int:
    args = build_arg_parser().parse_args()
    router_dir = args.router_dir.resolve()
    validate_router_dir(router_dir)

    print(f"Router: {router_dir}")
    print(f"Repo:   {args.repo_id}")
    print("Files:")
    for pattern in ALLOW_PATTERNS:
        path = router_dir / pattern
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
        folder_path=str(router_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=ALLOW_PATTERNS,
        commit_message=args.commit_message,
    )
    print(f"Uploaded router to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
