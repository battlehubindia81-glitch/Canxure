#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: str | Path | None = None) -> str:
    completed = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone or pull the GitHub source of truth inside Colab.")
    parser.add_argument("--repo-url", default=os.getenv("REPO_URL", "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"))
    parser.add_argument("--repo-dir", default=os.getenv("REPO_DIR", "/content/precision_oncology_digital_twin"))
    args = parser.parse_args()
    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        run(["git", "clone", args.repo_url, str(repo_dir)])
    else:
        run(["git", "pull"], cwd=repo_dir)
    print("Commit:", run(["git", "rev-parse", "HEAD"], cwd=repo_dir))
    print("Branch:", run(["git", "branch", "--show-current"], cwd=repo_dir))
    print("Status:", run(["git", "status", "--short"], cwd=repo_dir) or "clean")


if __name__ == "__main__":
    main()
