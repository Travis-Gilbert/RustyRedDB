#!/usr/bin/env python3
"""Fail when a manifest pulls private Theorem paths or product crates.

Point --manifest at a Cargo.toml. The default is this repository's workspace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FORBIDDEN = (
    "theorem-harness",
    "agpui",
    "theoremweb",
    "commonplace",
    "apps/theorem",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(manifest), "--no-deps"],
            text=True,
        )
    )
    problems = []
    for package in metadata["packages"]:
        manifest_path = Path(package["manifest_path"])
        try:
            manifest_path.relative_to(root)
        except ValueError:
            problems.append(f"package outside root: {manifest_path}")
        for token in FORBIDDEN:
            if token in package["name"]:
                problems.append(f"{package['name']} name contains {token}")
                break
        for dep in package["dependencies"]:
            name = dep.get("name") or ""
            path = dep.get("path") or ""
            for token in FORBIDDEN:
                if token in name or token in path:
                    problems.append(f"{package['name']} -> {name} contains {token}")
            path = dep.get("path")
            if path:
                dep_path = Path(path)
                if not dep_path.is_absolute():
                    dep_path = (manifest_path.parent / dep_path).resolve()
                try:
                    dep_path.relative_to(root)
                except ValueError:
                    problems.append(f"{package['name']} path dep escapes root: {dep_path}")
    if problems:
        print("dependency boundary failed:", file=sys.stderr)
        for item in problems:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(f"dependency boundary ok ({len(metadata['packages'])} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
