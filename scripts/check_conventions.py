#!/usr/bin/env python3
"""Check InfiniOps project-specific contribution conventions.

This script intentionally complements, rather than replaces, `clang-format`,
`clang-tidy`, and `ruff`.  It focuses on repository rules that generic tools
do not enforce reliably.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import dataclasses
import io
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]

CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".maca",
    ".mlu",
    ".mu",
}

PYTHON_EXTENSIONS = {".py"}

TEXT_EXTENSIONS = {
    *CPP_EXTENSIONS,
    *PYTHON_EXTENSIONS,
    ".cmake",
    ".css",
    ".gitignore",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_FILENAMES = {
    ".clang-format",
    ".clang-tidy",
    ".gitignore",
    "CMakeLists.txt",
    "Dockerfile",
}

SKIPPED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "generated",
    "node_modules",
}

CONVENTIONAL_TYPES = {
    "feat",
    "fix",
    "perf",
    "refactor",
    "test",
    "docs",
    "build",
    "ci",
    "chore",
}

CONVENTIONAL_RE = re.compile(
    rf"^({'|'.join(sorted(CONVENTIONAL_TYPES))})"
    r"(\([a-z0-9][a-z0-9/_-]*\))?"
    r"!?: .+"
)

BRANCH_RE = re.compile(
    rf"^({'|'.join(sorted(CONVENTIONAL_TYPES))})/"
    r"[a-z0-9]+(?:-[a-z0-9]+)*$"
)

CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "with",
    "try",
    "elif",
    "else",
    "except",
    "finally",
    "match",
}

CPP_CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
}

KNOWN_KERNEL_NAMES = {
    "add",
    "add_rms_norm",
    "blas",
    "cast",
    "cat",
    "causal_softmax",
    "cnblas",
    "cublas",
    "cublaslt",
    "flash_attention",
    "flash_attention_v2",
    "gemm",
    "kernel",
    "linear",
    "matmul",
    "mcblas",
    "mul",
    "mublas",
    "reshape_and_cache",
    "rms_norm",
    "rotary_embedding",
    "swiglu",
}

OPERATOR_DIR_NAMES = {
    "add",
    "add_rms_norm",
    "cast",
    "cat",
    "causal_softmax",
    "flash_attention",
    "gemm",
    "linear",
    "matmul",
    "mul",
    "reshape_and_cache",
    "rms_norm",
    "rotary_embedding",
    "swiglu",
}

SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:password|passwd|secret|token)\s*=\s*['\"][^'\"]{8,}"),
)


@dataclasses.dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    line: int | None
    column: int | None
    message: str

    def sort_key(self) -> tuple[str, int, int, str]:
        line = self.line if self.line is not None else 0
        column = self.column if self.column is not None else 0

        return (self.path, line, column, self.code)

    def to_text(self) -> str:
        location = self.path

        if self.line is not None:
            location += f":{self.line}"

            if self.column is not None:
                location += f":{self.column}"

        return f"{location}: {self.severity}: {self.code}: {self.message}"


@dataclasses.dataclass(frozen=True)
class CheckInfo:
    code: str
    severity: str
    source: str
    description: str


CHECKS = (
    CheckInfo(
        "TXT001",
        "error",
        "CONTRIBUTING.md §Code/General; PR checklist line 137",
        "Text files must end with a trailing newline.",
    ),
    CheckInfo(
        "TXT002",
        "error",
        "PR checklist line 137",
        "Text files must end with exactly one trailing newline.",
    ),
    CheckInfo(
        "TXT003",
        "error",
        "PR checklist line 138",
        "No trailing whitespace.",
    ),
    CheckInfo(
        "TXT004",
        "error",
        "PR checklist line 138",
        "No mixed tab/space indentation.",
    ),
    CheckInfo(
        "TXT005",
        "error",
        "PR checklist line 138",
        "No UTF-8 byte-order marks.",
    ),
    CheckInfo(
        "TXT006",
        "warning",
        "PR checklist line 131",
        "Files should not carry an executable bit unless they are executable.",
    ),
    CheckInfo(
        "SEC001",
        "error",
        "PR checklist line 200",
        "No obvious committed secrets.",
    ),
    CheckInfo(
        "TODO001",
        "warning",
        "PR checklist line 130",
        "`TODO` comments should include an owner and issue link/reference.",
    ),
    CheckInfo(
        "COMMENT001",
        "warning",
        "CONTRIBUTING.md §Code/General; PR checklist lines 140-141",
        "Comments should be English-only.",
    ),
    CheckInfo(
        "COMMENT002",
        "warning",
        "CONTRIBUTING.md §Code/General; PR checklist lines 139, 141",
        "Strict comment prose and backtick heuristics.",
    ),
    CheckInfo(
        "CXX001",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 149",
        "No C++ exceptions.",
    ),
    CheckInfo(
        "CXX002",
        "error",
        "PR checklist line 158",
        "No raw `new` or `delete`.",
    ),
    CheckInfo(
        "CXX003",
        "warning",
        "CONTRIBUTING.md §C++; PR checklist line 149",
        "`assert` messages should include `__FILE__`, `__LINE__`, and `__func__`.",
    ),
    CheckInfo(
        "CXX004",
        "warning",
        "PR checklist line 130",
        "No production debug prints such as `printf` or `std::cout`.",
    ),
    CheckInfo(
        "CXX005",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 151",
        "Kernel files must use project-approved names.",
    ),
    CheckInfo(
        "CXX006",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 152",
        "Kernel definitions and launchers must stay separated.",
    ),
    CheckInfo(
        "CXX007",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 155",
        "Exactly one blank line between class members.",
    ),
    CheckInfo(
        "CXX008",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 156",
        "Exactly one blank line after namespace opens and before namespace closes.",
    ),
    CheckInfo(
        "CXX009",
        "warning",
        "CONTRIBUTING.md §C++; PR checklist line 154",
        "Exactly one blank line between namespace-scope classes/functions.",
    ),
    CheckInfo(
        "CXX010",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 153",
        "Constructor initializer order must match member declaration order.",
    ),
    CheckInfo(
        "CXX011",
        "error",
        "CONTRIBUTING.md §C++; PR checklist line 148",
        "Base operator parameters must be inputs, then attributes, then outputs.",
    ),
    CheckInfo(
        "CXX012",
        "error",
        "CONTRIBUTING.md §Adding an Operator; PR checklist line 157",
        "Base operator classes must inherit from `Operator<Op>`.",
    ),
    CheckInfo(
        "PY001",
        "error",
        "CONTRIBUTING.md §Python; PR checklist line 166",
        "No blank line after a function signature unless a docstring/comment follows.",
    ),
    CheckInfo(
        "PY002",
        "error",
        "CONTRIBUTING.md §Python; PR checklist line 168",
        "A blank line should appear before `return` unless it follows control flow.",
    ),
    CheckInfo(
        "PY003",
        "warning",
        "CONTRIBUTING.md §Python; PR checklist line 167",
        "Control-flow statements should be surrounded by blank lines.",
    ),
    CheckInfo(
        "PY004",
        "warning",
        "PR checklist line 130",
        "No production `print(...)` debug output.",
    ),
    CheckInfo(
        "PY005",
        "error",
        "CONTRIBUTING.md §Adding an Operator; PR checklist line 177",
        "`pytest.mark.parametrize` names should follow function-argument order.",
    ),
    CheckInfo(
        "PY006",
        "error",
        "CONTRIBUTING.md §Adding an Operator; PR checklist line 178",
        "`auto_act_and_assert` tests should return `Payload`, and Payload tests should use the marker.",
    ),
    CheckInfo(
        "GIT001",
        "error",
        "CONTRIBUTING.md §Branches; PR checklist line 121",
        "Branch name should follow `<type>/xxx-yyyy-zzzz`.",
    ),
    CheckInfo(
        "GIT002",
        "error",
        "CONTRIBUTING.md §Commits; PR checklist lines 120, 122",
        "PR title and commit messages should follow Conventional Commits.",
    ),
    CheckInfo(
        "GIT003",
        "error",
        "PR checklist lines 124-125",
        "No merge, fixup, squash, or wip commits in the checked range.",
    ),
)


def run_git(args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    return result.stdout


def rel_path(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def should_skip(path: pathlib.Path) -> bool:
    rel = path.resolve().relative_to(ROOT)

    return any(part in SKIPPED_PARTS for part in rel.parts)


def is_text_candidate(path: pathlib.Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True

    return path.suffix in TEXT_EXTENSIONS


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:4096]


def list_tracked_files() -> list[pathlib.Path]:
    files = []

    for name in run_git(["ls-files"]).splitlines():
        path = ROOT / name

        if path.is_file() and not should_skip(path):
            files.append(path)

    return files


def list_diff_files(base: str, head: str) -> list[pathlib.Path]:
    output = run_git(
        ["diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}"]
    )
    files = []

    for name in output.splitlines():
        path = ROOT / name

        if path.is_file() and not should_skip(path):
            files.append(path)

    return files


def list_changed_files() -> list[pathlib.Path]:
    names = set()

    for args in (
        ["diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "HEAD"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        names.update(run_git(args, check=False).splitlines())

    files = []

    for name in sorted(names):
        path = ROOT / name

        if path.is_file() and not should_skip(path):
            files.append(path)

    return files


def line_starts(text: str) -> list[int]:
    starts = [0]

    for match in re.finditer("\n", text):
        starts.append(match.end())

    return starts


def line_col(starts: Sequence[int], offset: int) -> tuple[int, int]:
    line_index = bisect.bisect_right(starts, offset) - 1

    return line_index + 1, offset - starts[line_index] + 1


def add_finding(
    findings: list[Finding],
    severity: str,
    code: str,
    path: pathlib.Path,
    message: str,
    line: int | None = None,
    column: int | None = None,
) -> None:
    findings.append(
        Finding(
            severity=severity,
            code=code,
            path=rel_path(path),
            line=line,
            column=column,
            message=message,
        )
    )


def check_text_file(path: pathlib.Path, data: bytes) -> list[Finding]:
    findings: list[Finding] = []

    if not is_text_candidate(path) or is_binary(data):
        return findings

    if data.startswith(b"\xef\xbb\xbf"):
        add_finding(findings, "error", "TXT005", path, "Remove UTF-8 BOM.", 1, 1)

    if data and not data.endswith(b"\n"):
        add_finding(
            findings,
            "error",
            "TXT001",
            path,
            "File must end with a newline.",
        )

    if data.endswith(b"\n\n"):
        add_finding(
            findings,
            "error",
            "TXT002",
            path,
            "File must end with exactly one trailing newline.",
        )

    for line_no, raw_line in enumerate(data.splitlines(keepends=True), 1):
        line = raw_line.rstrip(b"\r\n")

        if line.endswith((b" ", b"\t")):
            column = len(line.rstrip(b" \t")) + 1
            add_finding(
                findings,
                "error",
                "TXT003",
                path,
                "Remove trailing whitespace.",
                line_no,
                column,
            )

        leading = re.match(rb"^[ \t]+", line)

        if leading and b" " in leading.group(0) and b"\t" in leading.group(0):
            add_finding(
                findings,
                "error",
                "TXT004",
                path,
                "Do not mix tabs and spaces in indentation.",
                line_no,
                1,
            )

    mode = path.stat().st_mode

    if mode & stat.S_IXUSR and not has_shebang(data) and path.suffix != ".sh":
        add_finding(
            findings,
            "warning",
            "TXT006",
            path,
            "Executable bit is set on a file without a shebang.",
        )

    return findings


def has_shebang(data: bytes) -> bool:
    return data.startswith(b"#!")


def check_secrets(path: pathlib.Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    starts = line_starts(text)

    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if is_obvious_fake_secret(match.group(0)):
                continue

            line, column = line_col(starts, match.start())
            add_finding(
                findings,
                "error",
                "SEC001",
                path,
                "Potential secret-like value found.",
                line,
                column,
            )

    return findings


def is_obvious_fake_secret(value: str) -> bool:
    lowered = value.lower()

    return any(
        marker in lowered
        for marker in ("dummy", "example", "fake", "fixture", "my-secret", "test")
    )


def todo_has_owner_and_issue(comment: str) -> bool:
    owner = re.search(r"\bTODO\([^)]+\)", comment)
    issue = re.search(r"(https?://\S+|#[0-9]+|[A-Z][A-Z0-9]+-[0-9]+)", comment)

    return bool(owner and issue)


def check_todo(path: pathlib.Path, text: str) -> list[Finding]:
    findings: list[Finding] = []

    if path.suffix in CPP_EXTENSIONS:
        candidates = extract_cpp_comments(text)
    elif path.suffix in PYTHON_EXTENSIONS:
        candidates = extract_python_comments(text)
    else:
        candidates = list(enumerate(text.splitlines(), 1))

    for line_no, comment in candidates:
        if "TODO" not in comment:
            continue

        if not todo_has_owner_and_issue(comment):
            add_finding(
                findings,
                "warning",
                "TODO001",
                path,
                "`TODO` should include an owner and issue link/reference.",
                line_no,
                comment.find("TODO") + 1,
            )

    return findings


def mask_cpp_comments_and_strings(text: str) -> str:
    chars = list(text)
    i = 0
    state = "normal"
    quote = ""

    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""

        if state == "normal":
            if c == "/" and nxt == "/":
                chars[i] = chars[i + 1] = " "
                i += 2
                state = "line_comment"
                continue

            if c == "/" and nxt == "*":
                chars[i] = chars[i + 1] = " "
                i += 2
                state = "block_comment"
                continue

            if c in {"'", '"'}:
                quote = c
                chars[i] = " "
                i += 1
                state = "string"
                continue

            i += 1
            continue

        if state == "line_comment":
            if c == "\n":
                state = "normal"
            else:
                chars[i] = " "

            i += 1
            continue

        if state == "block_comment":
            if c == "*" and nxt == "/":
                chars[i] = chars[i + 1] = " "
                i += 2
                state = "normal"
                continue

            if c != "\n":
                chars[i] = " "

            i += 1
            continue

        if state == "string":
            if c == "\\":
                chars[i] = " "

                if i + 1 < len(chars):
                    chars[i + 1] = " "

                i += 2
                continue

            if c == quote:
                chars[i] = " "
                i += 1
                state = "normal"
                continue

            if c != "\n":
                chars[i] = " "

            i += 1
            continue

    return "".join(chars)


def extract_cpp_comments(text: str) -> list[tuple[int, str]]:
    comments: list[tuple[int, str]] = []
    lines = text.splitlines()
    in_block = False
    block_start = 0
    block_parts: list[str] = []

    for line_no, line in enumerate(lines, 1):
        i = 0

        while i < len(line):
            if in_block:
                end = line.find("*/", i)

                if end == -1:
                    block_parts.append(line[i:])
                    break

                block_parts.append(line[i:end])
                comments.append((block_start, "\n".join(block_parts)))
                block_parts = []
                in_block = False
                i = end + 2
                continue

            line_comment = line.find("//", i)
            block_comment = line.find("/*", i)

            if line_comment == -1 and block_comment == -1:
                break

            if line_comment != -1 and (
                block_comment == -1 or line_comment < block_comment
            ):
                comments.append((line_no, line[line_comment + 2 :]))
                break

            end = line.find("*/", block_comment + 2)

            if end == -1:
                in_block = True
                block_start = line_no
                block_parts = [line[block_comment + 2 :]]
                break

            comments.append((line_no, line[block_comment + 2 : end]))
            i = end + 2

    return comments


def extract_python_comments(text: str) -> list[tuple[int, str]]:
    comments: list[tuple[int, str]] = []

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)

        for token in tokens:
            if token.type == tokenize.COMMENT:
                comments.append((token.start[0], token.string[1:]))
    except tokenize.TokenError:
        return comments

    return comments


def check_comments(
    path: pathlib.Path,
    text: str,
    *,
    strict_comments: bool,
) -> list[Finding]:
    findings: list[Finding] = []

    if path.suffix in CPP_EXTENSIONS:
        comments = extract_cpp_comments(text)
    elif path.suffix in PYTHON_EXTENSIONS:
        comments = extract_python_comments(text)
    else:
        return findings

    for line_no, comment in comments:
        clean = normalize_comment_text(comment)

        if not clean:
            continue

        if re.search(r"[\u3400-\u9fff]", clean):
            add_finding(
                findings,
                "warning",
                "COMMENT001",
                path,
                "Comment appears to contain non-English CJK text.",
                line_no,
                1,
            )

        if strict_comments:
            check_strict_comment(path, line_no, clean, findings)

    return findings


def normalize_comment_text(comment: str) -> str:
    lines = []

    for line in comment.splitlines():
        stripped = line.strip()

        if stripped.startswith("*"):
            stripped = stripped[1:].strip()

        lines.append(stripped)

    return " ".join(part for part in lines if part).strip()


def check_strict_comment(
    path: pathlib.Path,
    line_no: int,
    clean: str,
    findings: list[Finding],
) -> None:
    if should_skip_comment_style(clean):
        return

    first_alpha = re.search(r"[A-Za-z]", clean)

    if first_alpha and not clean[first_alpha.start()].isupper():
        add_finding(
            findings,
            "warning",
            "COMMENT002",
            path,
            "Comment sentence should start with a capital letter.",
            line_no,
            1,
        )

    if clean[-1] not in ".!?:`)>'\"":
        add_finding(
            findings,
            "warning",
            "COMMENT002",
            path,
            "Comment sentence should end with punctuation.",
            line_no,
            1,
        )

    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::|\(|_)[A-Za-z0-9_:()]*", clean):
        if f"`{token}`" not in clean and not token.startswith(("http", "TODO")):
            add_finding(
                findings,
                "warning",
                "COMMENT002",
                path,
                f"Code-like token `{token}` in comment should be backtick-fenced.",
                line_no,
                1,
            )
            break


def should_skip_comment_style(clean: str) -> bool:
    if len(clean.split()) <= 2:
        return True

    skip_prefixes = (
        "#",
        "-",
        "TODO",
        "NOLINT",
        "clang-format",
        "http://",
        "https://",
        "Copyright",
    )

    if clean.startswith(skip_prefixes):
        return True

    if set(clean) <= {"-", "=", "/", "*", " "}:
        return True

    return False


def find_matching_brace(masked: str, open_index: int) -> int | None:
    depth = 0

    for i in range(open_index, len(masked)):
        if masked[i] == "{":
            depth += 1
        elif masked[i] == "}":
            depth -= 1

            if depth == 0:
                return i

    return None


def next_nonspace(text: str, start: int, stop: int | None = None) -> str:
    limit = len(text) if stop is None else stop
    index = start

    while index < limit:
        if not text[index].isspace():
            return text[index]

        index += 1

    return ""


def find_matching_paren(masked: str, open_index: int) -> int | None:
    depth = 0

    for i in range(open_index, len(masked)):
        if masked[i] == "(":
            depth += 1
        elif masked[i] == ")":
            depth -= 1

            if depth == 0:
                return i

    return None


@dataclasses.dataclass
class ClassSpan:
    name: str
    open_index: int
    close_index: int
    open_line: int
    close_line: int


def find_class_spans(masked: str) -> list[ClassSpan]:
    starts = line_starts(masked)
    spans: list[ClassSpan] = []
    pattern = re.compile(r"\b(?:class|struct)\s+([A-Za-z_]\w*)[^;{]*\{")

    for match in pattern.finditer(masked):
        open_index = masked.find("{", match.start(), match.end())

        if open_index == -1:
            continue

        close_index = find_matching_brace(masked, open_index)

        if close_index is None:
            continue

        open_line, _ = line_col(starts, open_index)
        close_line, _ = line_col(starts, close_index)
        spans.append(
            ClassSpan(
                name=match.group(1),
                open_index=open_index,
                close_index=close_index,
                open_line=open_line,
                close_line=close_line,
            )
        )

    return spans


@dataclasses.dataclass
class Segment:
    kind: str
    start_line: int
    end_line: int
    text: str


def collect_segments_in_body(
    text: str,
    masked: str,
    open_index: int,
    close_index: int,
    *,
    class_body: bool,
) -> list[Segment]:
    starts = line_starts(masked)
    segments: list[Segment] = []
    i = open_index + 1
    current_start: int | None = None
    paren_depth = 0
    bracket_depth = 0
    body_depth = 0

    while i < close_index:
        c = masked[i]

        if current_start is None:
            if c.isspace():
                i += 1
                continue

            line, _ = line_col(starts, i)
            line_text = text.splitlines()[line - 1].strip()

            if class_body and re.fullmatch(r"(public|private|protected):", line_text):
                i = starts[line] if line < len(starts) else close_index
                continue

            current_start = i

        if c == "(":
            paren_depth += 1
        elif c == ")" and paren_depth > 0:
            paren_depth -= 1
        elif c == "[":
            bracket_depth += 1
        elif c == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif c == "{" and paren_depth == 0 and bracket_depth == 0:
            if body_depth > 0:
                body_depth += 1

            elif is_body_open_brace(masked, current_start, i, close_index):
                body_depth = 1
        elif c == "}" and body_depth > 0:
            body_depth -= 1

            if body_depth == 0 and current_start is not None:
                end_index = i
                kind = classify_segment(masked[current_start : end_index + 1])
                start_line, _ = line_col(starts, current_start)
                end_line, _ = line_col(starts, end_index)
                segments.append(
                    Segment(kind, start_line, end_line, text[current_start : end_index + 1])
                )
                current_start = None
        elif c == ";" and body_depth == 0 and paren_depth == 0 and bracket_depth == 0:
            end_index = i
            kind = classify_segment(masked[current_start : end_index + 1])
            start_line, _ = line_col(starts, current_start)
            end_line, _ = line_col(starts, end_index)
            segments.append(
                Segment(kind, start_line, end_line, text[current_start : end_index + 1])
            )
            current_start = None

        i += 1

    return segments


def is_body_open_brace(
    masked: str,
    segment_start: int,
    open_index: int,
    close_limit: int,
) -> bool:
    prefix = masked[segment_start:open_index]

    if not (")" in prefix or prefix.strip().startswith(("class ", "struct "))):
        return False

    close_index = find_matching_brace(masked, open_index)

    if close_index is None or close_index > close_limit:
        return False

    following = next_nonspace(masked, close_index + 1, close_limit)

    return following not in {",", ";", "{"}


def classify_segment(segment_text: str) -> str:
    stripped = segment_text.strip()

    if re.match(r"(template\s*<[^>]+>\s*)?(class|struct)\b", stripped, re.S):
        return "class"

    if "(" in stripped and not stripped.startswith(("using ", "typedef ")):
        first_word = stripped.split(None, 1)[0] if stripped.split() else ""

        if first_word not in CPP_CONTROL_KEYWORDS:
            return "function"

    if stripped.endswith(";") and not any(
        stripped.startswith(prefix)
        for prefix in ("using ", "typedef ", "static_assert", "friend ")
    ):
        return "member"

    return "other"


def blank_lines_between(lines: Sequence[str], end_line: int, start_line: int) -> int:
    if start_line <= end_line:
        return 0

    return sum(1 for line in lines[end_line:start_line - 1] if line.strip() == "")


def check_class_member_spacing(path: pathlib.Path, text: str, masked: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()

    for span in find_class_spans(masked):
        segments = collect_segments_in_body(
            text,
            masked,
            span.open_index,
            span.close_index,
            class_body=True,
        )
        member_segments = [
            segment
            for segment in segments
            if segment.kind in {"class", "function", "member"}
        ]

        for previous, current in zip(member_segments, member_segments[1:]):
            blanks = blank_lines_between(lines, previous.end_line, current.start_line)

            if blanks != 1:
                add_finding(
                    findings,
                    "error",
                    "CXX007",
                    path,
                    "Expected exactly one blank line between class members.",
                    current.start_line,
                    1,
                )

    return findings


def check_namespace_spacing(path: pathlib.Path, text: str, masked: str) -> list[Finding]:
    findings: list[Finding] = []
    starts = line_starts(masked)
    lines = text.splitlines()
    pattern = re.compile(r"\bnamespace(?:\s+[A-Za-z_][A-Za-z0-9_:]*)?\s*\{")

    for match in pattern.finditer(masked):
        open_index = masked.find("{", match.start(), match.end())

        if open_index == -1:
            continue

        close_index = find_matching_brace(masked, open_index)

        if close_index is None:
            continue

        open_line, _ = line_col(starts, open_index)
        close_line, _ = line_col(starts, close_index)

        if close_line <= open_line + 1:
            continue

        if open_line < len(lines) and lines[open_line].strip() != "":
            add_finding(
                findings,
                "error",
                "CXX008",
                path,
                "Expected one blank line after namespace opening brace.",
                open_line + 1,
                1,
            )

        if close_line >= 2 and lines[close_line - 2].strip() != "":
            add_finding(
                findings,
                "error",
                "CXX008",
                path,
                "Expected one blank line before namespace closing brace.",
                close_line,
                1,
            )

    return findings


def check_namespace_declaration_spacing(
    path: pathlib.Path,
    text: str,
    masked: str,
) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    namespace_pattern = re.compile(r"\bnamespace(?:\s+[A-Za-z_][A-Za-z0-9_:]*)?\s*\{")

    for match in namespace_pattern.finditer(masked):
        open_index = masked.find("{", match.start(), match.end())

        if open_index == -1:
            continue

        close_index = find_matching_brace(masked, open_index)

        if close_index is None:
            continue

        segments = collect_segments_in_body(
            text,
            masked,
            open_index,
            close_index,
            class_body=False,
        )
        decls = [segment for segment in segments if segment.kind in {"class", "function"}]

        for previous, current in zip(decls, decls[1:]):
            blanks = blank_lines_between(lines, previous.end_line, current.start_line)

            if blanks != 1:
                add_finding(
                    findings,
                    "warning",
                    "CXX009",
                    path,
                    "Expected exactly one blank line between namespace-scope classes/functions.",
                    current.start_line,
                    1,
                )

    return findings


def check_initializer_order(path: pathlib.Path, text: str, masked: str) -> list[Finding]:
    findings: list[Finding] = []

    for span in find_class_spans(masked):
        segments = collect_segments_in_body(
            text,
            masked,
            span.open_index,
            span.close_index,
            class_body=True,
        )
        member_order = extract_member_order(segments)

        if not member_order:
            continue

        member_index = {name: index for index, name in enumerate(member_order)}

        for segment in segments:
            if segment.kind != "function":
                continue

            if not re.search(rf"\b{re.escape(span.name)}\s*\(", segment.text):
                continue

            initializer_names = extract_initializer_names(span.name, segment.text)
            initializer_names = [
                name for name in initializer_names if name in member_index
            ]
            indexes = [member_index[name] for name in initializer_names]

            if indexes != sorted(indexes):
                add_finding(
                    findings,
                    "error",
                    "CXX010",
                    path,
                    "Constructor initializer list order differs from member declaration order.",
                    segment.start_line,
                    1,
                )

    return findings


def extract_member_order(segments: Sequence[Segment]) -> list[str]:
    names: list[str] = []

    for segment in segments:
        if segment.kind != "member":
            continue

        stripped = segment.text.strip()

        if "(" in stripped:
            continue

        if stripped.startswith(("using ", "typedef ", "static_assert", "friend ")):
            continue

        match = re.search(r"\b([A-Za-z_]\w*)\s*(?:\{|=|;|\[[^\]]*\]\s*;)", stripped)

        if match:
            names.append(match.group(1))

    return names


def extract_initializer_names(class_name: str, segment_text: str) -> list[str]:
    match = re.search(rf"\b{re.escape(class_name)}\s*\([^)]*\)\s*:(.*)\{{", segment_text, re.S)

    if not match:
        return []

    initializer_list = match.group(1)

    return re.findall(r"\b([A-Za-z_]\w*)\s*(?:\{|\()", initializer_list)


def check_cpp_tokens(path: pathlib.Path, text: str, masked: str) -> list[Finding]:
    findings: list[Finding] = []
    starts = line_starts(masked)

    token_checks = (
        ("error", "CXX001", r"\b(?:throw|try|catch)\b", "Do not use C++ exceptions."),
        ("error", "CXX002", r"\bnew\s+", "Use RAII, smart pointers, or existing allocators instead of raw `new`."),
        ("error", "CXX002", r"\bdelete\b", "Use RAII, smart pointers, or existing allocators instead of raw `delete`."),
    )

    for severity, code, pattern, message in token_checks:
        for match in re.finditer(pattern, masked):
            prefix = masked[max(0, match.start() - 16) : match.start()]

            if code == "CXX002" and "operator" in prefix:
                continue

            if code == "CXX002" and pattern == r"\bdelete\b":
                previous = previous_nonspace(masked, match.start())

                if previous == "=":
                    continue

            line, column = line_col(starts, match.start())
            add_finding(findings, severity, code, path, message, line, column)

    if "examples" not in path.parts:
        for match in re.finditer(r"\bprintf\s*\(|\bstd::cout\b", masked):
            line, column = line_col(starts, match.start())
            add_finding(
                findings,
                "warning",
                "CXX004",
                path,
                "Avoid production debug output.",
                line,
                column,
            )

    for match in re.finditer(r"(?<!static_)\bassert\s*\(", masked):
        open_index = masked.find("(", match.start())
        close_index = find_matching_paren(masked, open_index)

        if close_index is None:
            continue

        snippet = text[match.start() : close_index + 1]
        required = {"__FILE__", "__LINE__", "__func__"}

        if not required.issubset(set(re.findall(r"__[A-Z_a-z]+__", snippet))):
            line, column = line_col(starts, match.start())
            add_finding(
                findings,
                "warning",
                "CXX003",
                path,
                "`assert` should include `__FILE__`, `__LINE__`, and `__func__` in its message.",
                line,
                column,
            )

    return findings


def previous_nonspace(text: str, start: int) -> str:
    index = start - 1

    while index >= 0:
        if not text[index].isspace():
            return text[index]

        index -= 1

    return ""


def check_kernel_files(path: pathlib.Path, text: str, masked: str) -> list[Finding]:
    findings: list[Finding] = []

    if "src" not in path.parts:
        return findings

    if not is_operator_kernel_path(path):
        return findings

    has_kernel_marker = bool(
        re.search(r"__global__|\b[A-Za-z_]\w*Kernel\b|<<<", masked)
    )
    kernel_like_extension = path.suffix in {".cu", ".cuh", ".maca", ".mlu", ".mu"}

    if (has_kernel_marker or kernel_like_extension) and not is_allowed_kernel_name(path):
        add_finding(
            findings,
            "error",
            "CXX005",
            path,
            "Kernel file name should be `kernel`, `kernel_vN`, a well-known algorithm, or a library name.",
        )

    if path.suffix == ".h" and re.search(r"__global__", masked):
        add_finding(
            findings,
            "error",
            "CXX006",
            path,
            "Kernel definitions should not live in launcher `.h` files.",
        )

    if path.suffix in {".cu", ".cuh"} and "<<<" in masked:
        add_finding(
            findings,
            "error",
            "CXX006",
            path,
            "Kernel launchers should live in `.h` files, not kernel source files.",
        )

    return findings


def is_operator_kernel_path(path: pathlib.Path) -> bool:
    parts = set(path.parts)

    return "op_kernel" in parts or bool(parts & OPERATOR_DIR_NAMES)


def is_allowed_kernel_name(path: pathlib.Path) -> bool:
    name = path.stem

    if re.fullmatch(r"kernel(?:_v[0-9]+)?", name):
        return True

    return name in KNOWN_KERNEL_NAMES


def check_base_operator_structure(path: pathlib.Path, text: str) -> list[Finding]:
    findings: list[Finding] = []

    if path.parent != ROOT / "src" / "base" or path.suffix != ".h":
        return findings

    op_name = snake_to_pascal(path.stem)

    if not re.search(rf"\bclass\s+{re.escape(op_name)}\s*:\s*public\s+Operator\s*<\s*{re.escape(op_name)}\s*>", text):
        add_finding(
            findings,
            "error",
            "CXX012",
            path,
            f"`{op_name}` should inherit from `Operator<{op_name}>`.",
        )

    return findings


def check_base_operator_parameter_order(path: pathlib.Path, text: str) -> list[Finding]:
    findings: list[Finding] = []

    if path.parent != ROOT / "src" / "base" or path.suffix != ".h":
        return findings

    op_name = snake_to_pascal(path.stem)
    signatures = extract_operator_signatures(text, op_name)

    for line_no, name, params in signatures:
        categories = [classify_operator_param(param) for param in params]
        order_values = [category_order(category) for category in categories]

        if order_values != sorted(order_values):
            add_finding(
                findings,
                "error",
                "CXX011",
                path,
                f"`{name}` parameters should be inputs, attributes, then outputs.",
                line_no,
                1,
            )

    return findings


def extract_operator_signatures(text: str, op_name: str) -> list[tuple[int, str, list[str]]]:
    signatures: list[tuple[int, str, list[str]]] = []
    starts = line_starts(text)
    pattern = re.compile(rf"\b({re.escape(op_name)}|operator\s*\(\))\s*\(")

    for match in pattern.finditer(text):
        open_index = text.find("(", match.start(), match.end())
        close_index = find_matching_paren(text, open_index)

        if close_index is None:
            continue

        line, _ = line_col(starts, match.start())
        params = split_params(text[open_index + 1 : close_index])
        signatures.append((line, match.group(1).replace(" ", ""), params))

    return signatures


def split_params(params_text: str) -> list[str]:
    params: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    brace_depth = 0

    for index, char in enumerate(params_text):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth > 0:
            paren_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
        elif char == "," and angle_depth == paren_depth == brace_depth == 0:
            param = params_text[start:index].strip()

            if param:
                params.append(param)

            start = index + 1

    tail = params_text[start:].strip()

    if tail:
        params.append(tail)

    return params


def classify_operator_param(param: str) -> str:
    normalized = " ".join(param.replace("&", " & ").replace("*", " * ").split())
    name_match = re.search(r"([A-Za-z_]\w*)\s*(?:=.*)?$", normalized)
    name = name_match.group(1) if name_match else ""

    if "Tensor" not in normalized:
        return "attribute"

    if "std::vector" in normalized:
        return "input"

    if "std::optional" in normalized and not re.search(r"(^|_)out(?:put)?s?$", name):
        return "input"

    if re.search(r"\bconst\s+Tensor\b", normalized):
        return "input"

    return "output"


def category_order(category: str) -> int:
    return {"input": 0, "attribute": 1, "output": 2}[category]


def snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def check_cpp_file(path: pathlib.Path, text: str) -> list[Finding]:
    masked = mask_cpp_comments_and_strings(text)
    findings: list[Finding] = []
    findings.extend(check_cpp_tokens(path, text, masked))
    findings.extend(check_kernel_files(path, text, masked))
    findings.extend(check_class_member_spacing(path, text, masked))
    findings.extend(check_namespace_spacing(path, text, masked))
    findings.extend(check_namespace_declaration_spacing(path, text, masked))
    findings.extend(check_initializer_order(path, text, masked))
    findings.extend(check_base_operator_structure(path, text))
    findings.extend(check_base_operator_parameter_order(path, text))

    return findings


def parse_python(path: pathlib.Path, text: str) -> ast.Module | None:
    try:
        return ast.parse(text, filename=rel_path(path))
    except SyntaxError:
        return None


def check_python_file(path: pathlib.Path, text: str) -> list[Finding]:
    tree = parse_python(path, text)

    if tree is None:
        return []

    findings: list[Finding] = []
    lines = text.splitlines()
    findings.extend(check_python_function_spacing(path, lines, tree))
    findings.extend(check_python_return_spacing(path, lines, tree))
    findings.extend(check_python_control_flow_spacing(path, lines, tree))
    findings.extend(check_python_debug_print(path, tree))
    findings.extend(check_pytest_patterns(path, tree))

    return findings


def check_python_function_spacing(
    path: pathlib.Path,
    lines: Sequence[str],
    tree: ast.AST,
) -> list[Finding]:
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        header_end = find_python_header_end(lines, node)

        if header_end is None or not node.body:
            continue

        first_body_line = node.body[0].lineno
        first_body_is_docstring = (
            isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )

        if first_body_is_docstring:
            continue

        between = lines[header_end:first_body_line - 1]
        first_content = next((line.strip() for line in between if line.strip()), "")

        if first_content.startswith("#"):
            continue

        if any(line.strip() == "" for line in between):
            add_finding(
                findings,
                "error",
                "PY001",
                path,
                "Remove blank line between function signature and body.",
                header_end + 1,
                1,
            )

    return findings


def find_python_header_end(
    lines: Sequence[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int | None:
    body_start = node.body[0].lineno if node.body else node.lineno

    for line_no in range(node.lineno, body_start + 1):
        stripped = lines[line_no - 1].strip()

        if stripped.endswith(":"):
            return line_no

    return None


def check_python_return_spacing(
    path: pathlib.Path,
    lines: Sequence[str],
    tree: ast.AST,
) -> list[Finding]:
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue

        if node.lineno <= 1:
            continue

        previous = lines[node.lineno - 2].strip()

        if previous == "":
            continue

        if previous.endswith(":"):
            continue

        add_finding(
            findings,
            "error",
            "PY002",
            path,
            "Expected a blank line before `return`.",
            node.lineno,
            1,
        )

    return findings


def is_control_header(line: str) -> bool:
    if not line.endswith(":"):
        return False

    first = line.split(None, 1)[0].rstrip(":")

    return first in CONTROL_KEYWORDS


def check_python_control_flow_spacing(
    path: pathlib.Path,
    lines: Sequence[str],
    tree: ast.AST,
) -> list[Finding]:
    findings: list[Finding] = []
    node_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)

    for node in ast.walk(tree):
        if not isinstance(node, node_types):
            continue

        previous_line = lines[node.lineno - 2].strip() if node.lineno > 1 else ""

        if previous_line and not previous_line.startswith(("#", "@")) and not previous_line.endswith(":"):
            add_finding(
                findings,
                "warning",
                "PY003",
                path,
                "Expected a blank line before control-flow statement.",
                node.lineno,
                1,
            )

        end_line = getattr(node, "end_lineno", None)

        if end_line is None or end_line >= len(lines):
            continue

        next_line = lines[end_line].strip()

        if not next_line:
            continue

        if next_line.startswith(("elif ", "else:", "except ", "except:", "finally:")):
            continue

        add_finding(
            findings,
            "warning",
            "PY003",
            path,
            "Expected a blank line after control-flow block.",
            end_line + 1,
            1,
        )

    return findings


def check_python_debug_print(path: pathlib.Path, tree: ast.AST) -> list[Finding]:
    findings: list[Finding] = []

    rel_parts = path.resolve().relative_to(ROOT).parts

    if not rel_parts or rel_parts[0] not in {"src", "tests"}:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name) and node.func.id == "print":
            add_finding(
                findings,
                "warning",
                "PY004",
                path,
                "Avoid production `print(...)` debug output.",
                node.lineno,
                node.col_offset + 1,
            )

    return findings


def check_pytest_patterns(path: pathlib.Path, tree: ast.AST) -> list[Finding]:
    findings: list[Finding] = []

    if not path.name.startswith("test_"):
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue

        findings.extend(check_parametrize_order(path, node))
        findings.extend(check_auto_payload_marker(path, node))

    return findings


def check_parametrize_order(path: pathlib.Path, node: ast.FunctionDef) -> list[Finding]:
    findings: list[Finding] = []
    arg_names = [arg.arg for arg in node.args.args]
    group_starts: list[int] = []

    for decorator in node.decorator_list:
        names = parametrize_names(decorator)

        if not names:
            continue

        indexes = []

        for name in names:
            if name not in arg_names:
                add_finding(
                    findings,
                    "error",
                    "PY005",
                    path,
                    f"`pytest.mark.parametrize` references unknown argument `{name}`.",
                    decorator.lineno,
                    1,
                )
                continue

            indexes.append(arg_names.index(name))

        if indexes and indexes != sorted(indexes):
            add_finding(
                findings,
                "error",
                "PY005",
                path,
                "Parametrized argument names should follow function-argument order.",
                decorator.lineno,
                1,
            )

        if indexes:
            group_starts.append(indexes[0])

    if group_starts != sorted(group_starts):
        add_finding(
            findings,
            "error",
            "PY005",
            path,
            "Independent `pytest.mark.parametrize` decorators should follow function-argument order.",
            node.lineno,
            1,
        )

    return findings


def parametrize_names(decorator: ast.expr) -> list[str]:
    if not isinstance(decorator, ast.Call):
        return []

    if dotted_name(decorator.func) != "pytest.mark.parametrize":
        return []

    if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
        return []

    value = decorator.args[0].value

    if not isinstance(value, str):
        return []

    return [part.strip() for part in value.replace("(", "").replace(")", "").split(",") if part.strip()]


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)

        if parent:
            return f"{parent}.{node.attr}"

    return ""


def check_auto_payload_marker(path: pathlib.Path, node: ast.FunctionDef) -> list[Finding]:
    findings: list[Finding] = []
    has_marker = any(
        dotted_name(decorator) == "pytest.mark.auto_act_and_assert"
        or (
            isinstance(decorator, ast.Call)
            and dotted_name(decorator.func) == "pytest.mark.auto_act_and_assert"
        )
        for decorator in node.decorator_list
    )
    returns_payload = any(
        isinstance(child, ast.Return) and is_payload_call(child.value)
        for child in ast.walk(node)
    )

    if has_marker and not returns_payload:
        add_finding(
            findings,
            "error",
            "PY006",
            path,
            "`auto_act_and_assert` test should return `Payload`.",
            node.lineno,
            1,
        )

    if returns_payload and not has_marker:
        add_finding(
            findings,
            "error",
            "PY006",
            path,
            "`Payload`-returning test should use `pytest.mark.auto_act_and_assert`.",
            node.lineno,
            1,
        )

    return findings


def is_payload_call(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Call) and dotted_name(node.func).endswith("Payload")


def check_git_conventions(
    *,
    base: str | None,
    head: str,
    pr_title: str | None,
) -> list[Finding]:
    findings: list[Finding] = []
    synthetic_path = ROOT / ".git"

    branch = os.environ.get("GITHUB_HEAD_REF")

    if not branch:
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False).strip()

    if branch and branch != "HEAD" and not BRANCH_RE.fullmatch(branch):
        add_finding(
            findings,
            "error",
            "GIT001",
            synthetic_path,
            f"Branch `{branch}` does not match `<type>/xxx-yyyy-zzzz`.",
        )

    title = pr_title or os.environ.get("PR_TITLE")

    if title and not CONVENTIONAL_RE.fullmatch(title):
        add_finding(
            findings,
            "error",
            "GIT002",
            synthetic_path,
            "PR title does not follow Conventional Commits.",
        )

    if not base:
        return findings

    rev_range = f"{base}..{head}"
    subjects = run_git(["log", "--format=%H%x00%s", rev_range], check=False)

    for row in subjects.splitlines():
        if "\0" not in row:
            continue

        commit, subject = row.split("\0", 1)

        if re.match(r"(?i)^(fixup!|squash!|wip\b)", subject):
            add_finding(
                findings,
                "error",
                "GIT003",
                synthetic_path,
                f"Commit `{commit[:12]}` has temporary subject `{subject}`.",
            )
        elif not CONVENTIONAL_RE.fullmatch(subject):
            add_finding(
                findings,
                "error",
                "GIT002",
                synthetic_path,
                f"Commit `{commit[:12]}` does not follow Conventional Commits.",
            )

    merges = run_git(["rev-list", "--merges", rev_range], check=False)

    for commit in merges.splitlines():
        add_finding(
            findings,
            "error",
            "GIT003",
            synthetic_path,
            f"Merge commit `{commit[:12]}` is present in the checked range.",
        )

    return findings


def check_file(
    path: pathlib.Path,
    *,
    strict_comments: bool,
) -> list[Finding]:
    data = path.read_bytes()
    findings = check_text_file(path, data)

    if is_binary(data):
        return findings

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return findings

    findings.extend(check_secrets(path, text))
    findings.extend(check_todo(path, text))
    findings.extend(check_comments(path, text, strict_comments=strict_comments))

    if path.suffix in CPP_EXTENSIONS:
        findings.extend(check_cpp_file(path, text))
    elif path.suffix in PYTHON_EXTENSIONS:
        findings.extend(check_python_file(path, text))

    return findings


def print_check_catalog() -> None:
    for check in CHECKS:
        print(f"{check.code} [{check.severity}] {check.description}")
        print(f"  Source: {check.source}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check InfiniOps project-specific conventions."
    )
    parser.add_argument("paths", nargs="*", help="Specific files to check.")
    parser.add_argument("--base", help="Base ref for diff-only checks.")
    parser.add_argument("--head", default="HEAD", help="Head ref for diff-only checks.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all tracked files even when `--base` is provided.",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="Check staged and unstaged working-tree changes against `HEAD`.",
    )
    parser.add_argument(
        "--check-git",
        action="store_true",
        help="Check branch, PR title, and commit-message conventions.",
    )
    parser.add_argument("--pr-title", help="PR title to validate.")
    parser.add_argument(
        "--strict-comments",
        action="store_true",
        help="Enable noisy comment sentence/backtick heuristics.",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Exit non-zero when warnings are found.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=300,
        help="Maximum findings to print before truncating text output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--summary-by-code",
        action="store_true",
        help="Print finding counts grouped by check code.",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List implemented checks and exit.",
    )

    return parser.parse_args(argv)


def resolve_files(args: argparse.Namespace) -> list[pathlib.Path]:
    if args.paths:
        files = []

        for raw_path in args.paths:
            path = (ROOT / raw_path).resolve()

            if path.is_file() and not should_skip(path):
                files.append(path)

        return files

    if args.changed:
        return list_changed_files()

    if args.base and not args.all:
        return list_diff_files(args.base, args.head)

    return list_tracked_files()


def summarize(findings: Sequence[Finding]) -> dict[str, int]:
    summary: dict[str, int] = {}

    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1

    return summary


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if args.list_checks:
        print_check_catalog()

        return 0

    findings: list[Finding] = []

    for path in resolve_files(args):
        findings.extend(check_file(path, strict_comments=args.strict_comments))

    if args.check_git:
        findings.extend(
            check_git_conventions(
                base=args.base,
                head=args.head,
                pr_title=args.pr_title,
            )
        )

    findings.sort(key=Finding.sort_key)

    if args.json:
        print(json.dumps([dataclasses.asdict(f) for f in findings], indent=2))
    else:
        for finding in findings[: args.max_findings]:
            print(finding.to_text())

        if len(findings) > args.max_findings:
            remaining = len(findings) - args.max_findings
            print(f"... truncated {remaining} additional finding(s).")

        summary = summarize(findings)
        print(
            "Summary: "
            f"{summary.get('error', 0)} error(s), "
            f"{summary.get('warning', 0)} warning(s)."
        )

        if args.summary_by_code:
            counts: dict[tuple[str, str], int] = {}

            for finding in findings:
                key = (finding.severity, finding.code)
                counts[key] = counts.get(key, 0) + 1

            for (severity, code), count in sorted(counts.items()):
                print(f"  {code} [{severity}]: {count}")

    has_errors = any(finding.severity == "error" for finding in findings)
    has_warnings = any(finding.severity == "warning" for finding in findings)

    if has_errors or (args.warnings_as_errors and has_warnings):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
