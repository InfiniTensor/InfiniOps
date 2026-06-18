#!/usr/bin/env python3
"""Generate the public `operator.h` installed for downstream consumers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_CALL_SIGNATURE_RE = re.compile(
    r"(?P<indent>^[ \t]*)template\s*<typename\.\.\.\s+Args>\s*\n"
    r"(?P=indent)static\s+void\s+Call\s*\(\s*"
    r"const\s+Handle\s*&\s*handle\s*,\s*"
    r"const\s+Config\s*&\s*config\s*,\s*"
    r"const\s+Args\s*&\s*\.\.\.\s*args\s*\)\s*",
    re.MULTILINE,
)


def _find_matching_brace(text: str, open_brace: int) -> int:
    depth = 0
    index = open_brace
    state = "normal"
    raw_string_end = ""

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if state == "line_comment":
            if char == "\n":
                state = "normal"
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
                continue
            index += 1
            continue

        if state == "string":
            if char == "\\":
                index += 2
                continue
            if char == '"':
                state = "normal"
            index += 1
            continue

        if state == "char":
            if char == "\\":
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue

        if state == "raw_string":
            if text.startswith(raw_string_end, index):
                state = "normal"
                index += len(raw_string_end)
                continue
            index += 1
            continue

        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue

        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue

        if char == "R" and next_char == '"':
            delimiter_start = index + 2
            delimiter_end = text.find("(", delimiter_start)

            if delimiter_end != -1:
                delimiter = text[delimiter_start:delimiter_end]
                raw_string_end = f'){delimiter}"'
                state = "raw_string"
                index = delimiter_end + 1
                continue

        if char == '"':
            state = "string"
            index += 1
            continue

        if char == "'":
            state = "char"
            index += 1
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return index

        index += 1

    raise RuntimeError("Could not find the end of `Operator::Call`.")


def _hide_operator_call_definition(text: str) -> str:
    match = _CALL_SIGNATURE_RE.search(text)

    if match is None:
        raise RuntimeError("Could not find the `Operator::Call` signature.")

    body_start = text.find("{", match.end())

    if body_start == -1:
        raise RuntimeError("Could not find the `Operator::Call` body.")

    if text[match.end() : body_start].strip():
        raise RuntimeError("Unexpected tokens before the `Operator::Call` body.")

    body_end = _find_matching_brace(text, body_start)
    signature = text[match.start() : match.end()].rstrip()

    return f"{text[: match.start()]}{signature};{text[body_end + 1 :]}"


def generate_public_operator_header(source: Path, output: Path) -> None:
    public_text = _hide_operator_call_definition(source.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(public_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("src/operator.h"))
    parser.add_argument(
        "--output", type=Path, default=Path("generated/include/operator.h")
    )
    args = parser.parse_args()
    generate_public_operator_header(args.source, args.output)


if __name__ == "__main__":
    main()
