"""Check a source-only release without reading credential files or requiring Git."""

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".html", ".cff"}
SECRET_NAMES = {".env", ".netrc", ".git-credentials", "config.toml", "id_rsa", "id_ed25519"}
PRIVATE_PATH = re.compile(r"/(?:home|data-nfs)/[^\s\"'<>`]+")


def check_tree(root=ROOT):
    errors, files = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            errors.append(f"Symlink not allowed: {relative}")
            continue
        if not path.is_file():
            continue
        if (path.name in SECRET_NAMES or path.suffix in {".pem", ".key"}
                or path.name.startswith(".env.") or ".local." in path.name):
            errors.append(f"Private file (content not read): {relative}")
            continue
        if (path.name not in {"LICENSE", ".gitignore"} and path.suffix not in ALLOWED_SUFFIXES
                or any(p in {"data", "models", "outputs", "logs", ".cache", "__pycache__", ".venv"} for p in relative.parts)
                or path.stat().st_size > 1024 * 1024):
            errors.append(f"Not a source release file: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        if PRIVATE_PATH.search(text):
            errors.append(f"Machine-specific absolute path: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                errors.append(f"Invalid Python: {relative}: {error}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                    continue
                target = target.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    errors.append(f"Broken documentation link: {relative}: {target}")
        files.append(str(relative))
    return errors, files


def check_help():
    entries = [ROOT / name for name in ("train.py", "generate_predictions.py", "run_evaluation.py")]
    entries += sorted((ROOT / "scripts").glob("*.py"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    for entry in entries:
        result = subprocess.run([sys.executable, str(entry), "--help"], cwd=ROOT.parent,
                                env=env, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"{entry.name} --help failed: {result.stderr[-2000:]}")
        print(f"HELP OK: {entry.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-help", action="store_true")
    args = parser.parse_args()
    errors, files = check_tree()
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        raise SystemExit(1)
    if args.check_help:
        check_help()
    print(f"Source release verified: {len(files)} text files; no bundled assets or private machine paths")


if __name__ == "__main__":
    main()
