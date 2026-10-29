#!/usr/bin/env python3
"""Review guard for declaration inputs and dispatch/build policy, not an ABI proof."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "declaration_inputs": [
        "include/**/*.h",
        "src/base/**/*.h",
        "src/*.h",
        "src/common/**/*.h",
        "docs/compatibility.md",
    ],
    "dispatch_and_generation": [
        "src/operator.h",
        "src/dispatcher.h",
        "scripts/generate_*.py",
        "scripts/ops_config.py",
        "scripts/resolve_linked_ops.py",
        "scripts/torch_ops.yaml",
        "ops.json",
        "scripts/configs/**/*.json",
    ],
    "build_and_dependency_recipes": [
        "CMakeLists.txt",
        "src/**/CMakeLists.txt",
        "src/**/*.cmake",
        "src/**/*.cmake.in",
        "src/**/*.sh",
        "src/linked/**/*.yaml",
        "src/linked/**/*.yml",
        "scripts/*.sh",
        "scripts/dev/**/*.sh",
        "scripts/prepare_*.py",
        "cmake/**/*.cmake",
        "cmake/**/*.cmake.in",
        "pyproject.toml",
        ".github/ci_config.yml",
        ".github/workflows/ci*.yml",
        "images/**/Dockerfile*",
    ],
}


def strict_object(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key `{key}`.")

        result[key] = value

    return result


def valid_revision(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def digest(path):
    # Git checkouts with CRLF and LF must compare identically.
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inventory(root):
    groups = {}
    for name, patterns in GROUPS.items():
        files = sorted(
            {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
        )
        if not files:
            raise ValueError(f"Empty protected group `{name}`.")
        groups[name] = {p.relative_to(root).as_posix(): digest(p) for p in files}
    # These anchors distinguish a real source checkout from an empty/partial root.
    for anchor in (
        "include/infini/ops.h",
        "src/base/add.h",
        "src/operator.h",
        "src/config.h",
        "CMakeLists.txt",
        ".github/ci_config.yml",
    ):
        if not (root / anchor).is_file():
            raise ValueError(f"Missing required source `{anchor}`.")
    return groups


def compare(baseline, actual):
    if (
        not isinstance(baseline, dict)
        or type(baseline.get("format_version")) is not int
        or baseline.get("format_version") != 1
        or baseline.get("scope") != GROUPS
        or not valid_revision(baseline.get("source_revision"))
    ):
        raise ValueError("Unsupported baseline format, revision, or collector scope.")
    expected = baseline.get("groups")
    if not isinstance(expected, dict) or set(expected) != set(actual):
        raise ValueError("Baseline protected groups are missing or unknown.")
    changes = {}
    for group, files in actual.items():
        old = expected[group]
        if not isinstance(old, dict) or not old:
            raise ValueError(f"Invalid baseline group `{group}`.")
        for path, value in old.items():
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or "\\" in path
                or ":" in path
                or any(part in ("", ".", "..") for part in path.split("/"))
                or not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"Invalid path or SHA256 in baseline group `{group}`.")

        delta = {
            "added": sorted(files.keys() - old.keys()),
            "removed": sorted(old.keys() - files.keys()),
            "changed": sorted(
                p for p in files.keys() & old.keys() if files[p] != old[p]
            ),
        }
        if any(delta.values()):
            changes[group] = delta
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--source-revision", help="Required provenance when creating a snapshot"
    )
    args = parser.parse_args()
    try:
        actual = inventory(args.root)
        if args.mode == "capture":
            if not valid_revision(args.source_revision):
                raise ValueError("Capture requires a full lowercase source commit SHA.")
            payload = {
                "format_version": 1,
                "source_revision": args.source_revision,
                "scope": GROUPS,
                "groups": actual,
            }
            # Never silently bless a changed tree by overwriting the accepted baseline.
            with args.baseline.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(payload, output, indent=2, sort_keys=True)
                output.write("\n")
            result = {
                "status": "captured",
                "files": len({p for group in actual.values() for p in group}),
            }
        else:
            changes = compare(
                json.loads(
                    args.baseline.read_text(encoding="utf-8"),
                    object_pairs_hook=strict_object,
                ),
                actual,
            )
            result = {
                "status": "review_required" if changes else "unchanged",
                "changes": changes,
            }
        print(json.dumps(result, indent=2))
        return 1 if result["status"] == "review_required" else 0
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
