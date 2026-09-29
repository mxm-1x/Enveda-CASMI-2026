#!/usr/bin/env python3
"""Stage a minimal, versioned code dataset for offline Kaggle notebooks."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/private/tmp/casmi-source")
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Choose an output directory outside the repository")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Commit the source snapshot before packaging so SOURCE_COMMIT.txt identifies the exact code")
    commit = subprocess.check_output(["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    if output.exists():
        shutil.rmtree(output)
    (output / "src").mkdir(parents=True)
    for source in (ROOT / "src").glob("*.py"):
        shutil.copy2(source, output / "src" / source.name)
    shutil.copy2(ROOT / "requirements-kaggle.txt", output / "requirements-kaggle.txt")
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "CASMI 2026 source snapshot for offline Kaggle notebooks.\n"
        f"Git commit: {commit}\n",
        encoding="utf-8",
    )
    print(f"Staged {len(list((output / 'src').glob('*.py')))} Python files at {output}")
    print(f"Source commit: {commit}")


if __name__ == "__main__":
    main()
