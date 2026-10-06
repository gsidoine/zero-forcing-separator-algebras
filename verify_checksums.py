#!/usr/bin/env python3
"""Cross-platform verification of a GNU-style SHA-256 manifest."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="SHA256SUMS.txt", type=Path)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    checked = 0
    for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            expected, relative = line.split(None, 1)
        except ValueError as exc:
            raise SystemExit(f"Malformed manifest line {line_number}: {line!r}") from exc
        relative = relative.lstrip(" *")
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Missing file: {relative}")
        observed = digest(path)
        if observed != expected:
            raise SystemExit(
                f"Checksum mismatch for {relative}: expected {expected}, observed {observed}"
            )
        checked += 1
        print(f"OK  {relative}")
    print(f"All {checked} listed files passed SHA-256 verification.")


if __name__ == "__main__":
    main()
