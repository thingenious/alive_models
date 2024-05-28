"""Cleanup unnecessary files and directories."""

import glob
import os
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DIR_PATTERNS = [
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "build",
]

FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyc~",
    "*.py~",
    "*~",
    ".*~",
    ".coverage*",
]


def main() -> None:
    """Cleanup unnecessary files and directories."""
    cwd = os.getcwd()
    if str(ROOT_DIR) != cwd:
        os.chdir(ROOT_DIR)
    for pattern in DIR_PATTERNS:
        for dirpath in glob.glob(f"./**/{pattern}", recursive=True):
            # if the folder is ./data/.cache, keep it, it has the huggingface cache
            if os.path.basename(dirpath) == ".cache" and os.path.basename(os.path.dirname(dirpath)) == "data":
                print(f"keeping {dirpath}")
                continue
            print(f"removing {dirpath}")
            shutil.rmtree(dirpath)

    for pattern in FILE_PATTERNS:
        for filepath in glob.glob(f"./**/{pattern}", recursive=True):
            print(f"removing {filepath}")
            os.remove(filepath)
    if str(ROOT_DIR) != cwd:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
