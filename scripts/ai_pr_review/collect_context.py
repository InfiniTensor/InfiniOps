#!/usr/bin/env python3
"""Collect stable, read-only GitHub PR context for automated PR review.

The collector intentionally uses only GitHub's REST API. It never evaluates or
executes content authored by the PR contributor, and never interpolates GitHub
metadata into a shell command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

MAX_PER_PAGE = 100
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Config:
    owner: str
    repo: str
    pr_number: int
    output_dir: Path
    token: str | None
    api_root: str

    @property
    def repo_slug(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True, slots=True)
class ApiError(RuntimeError):
    status: int | None
    message: str

    def __str__(self) -> str:
        return f"GitHub API error{f' (HTTP {self.status})' if self.status else ''}: {self.message}"


def _request(config: Config, path: str, *, params: dict[str, Any] | None = None) -> Any:
    url = f"{config.api_root}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "InfiniOps-AI-PR-Review",
    }
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            payload = json.load(error)
            message = str(payload.get("message") or error.reason)
        except Exception:
            message = str(error.reason)
        raise ApiError(error.code, f"{message} (path={path})") from error
    except URLError as error:
        raise ApiError(None, f"{error.reason} (path={path})") from error
    except json.JSONDecodeError as error:
        raise ApiError(None, f"invalid JSON response: {error}") from error


def _paginated(config: Config, path: str, *, key: str | None = None) -> Iterator[Any]:
    page = 1
    while True:
        payload = _request(
            config, path, params={"per_page": MAX_PER_PAGE, "page": page}
        )
        items = payload.get(key, []) if key else payload
        if not isinstance(items, list):
            raise ApiError(
                None, f"expected a JSON list from {path}, got {type(items).__name__}"
            )
        yield from items
        if len(items) < MAX_PER_PAGE:
            return
        page += 1
        if page > 50:
            raise ApiError(None, f"pagination guard reached while reading {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sanitize_status_status(status: dict[str, Any]) -> dict[str, Any]:
    keep = ("state", "context", "description", "target_url")
    return {key: status.get(key) for key in keep}


def _sanitize_check(check: dict[str, Any]) -> dict[str, Any]:
    keep = ("name", "status", "conclusion", "started_at", "completed_at", "details_url")
    return {key: check.get(key) for key in keep}


def _truncate_patch(patch: str | None) -> str | None:
    if patch is None:
        return None
    # Keep the complete patch for ordinary review-sized files, but bound a
    # malicious or generated file before it can exhaust the model context.
    return (
        patch[:512_000] + "\n... [patch truncated by collector]"
        if len(patch) > 512_000
        else patch
    )


def _collect(config: Config) -> dict[str, Any]:
    pull_path = f"repos/{config.repo_slug}/pulls/{config.pr_number}"
    pull = _request(config, pull_path)
    if not isinstance(pull, dict):
        raise ApiError(
            None,
            f"expected a pull request object for {config.repo_slug}#{config.pr_number}",
        )

    head = pull.get("head") or {}
    base = pull.get("base") or {}
    head_sha = str(head.get("sha") or "")
    if not _SHA_RE.fullmatch(head_sha):
        raise ApiError(None, f"PR head SHA is missing or invalid: {head_sha!r}")

    files = list(_paginated(config, f"{pull_path}/files"))
    commits = list(_paginated(config, f"{pull_path}/commits"))
    issue_comments = list(
        _paginated(
            config, f"repos/{config.repo_slug}/issues/{config.pr_number}/comments"
        )
    )
    reviews = list(_paginated(config, f"{pull_path}/reviews"))
    review_ids = [
        str(review.get("id")) for review in reviews if review.get("id") is not None
    ]
    review_comments: list[dict[str, Any]] = []
    for review_id in review_ids:
        review_comments.extend(
            _paginated(config, f"{pull_path}/reviews/{review_id}/comments")
        )
    review_comments.sort(key=lambda item: str(item.get("created_at") or ""))

    labels = [
        label.get("name")
        for label in pull.get("labels", [])
        if label.get("name") is not None
    ]
    requested_reviewers = [
        user.get("login")
        for user in pull.get("requested_reviewers", [])
        if user.get("login") is not None
    ]
    requested_teams = [
        team.get("slug")
        for team in pull.get("requested_teams", [])
        if team.get("slug") is not None
    ]

    branch_protection_url = (
        f"repos/{config.repo_slug}/branches/"
        f"{quote(str(base.get('ref') or ''), safe='/')}/protection"
    )
    try:
        branch_protection = _request(config, branch_protection_url)
    except ApiError as error:
        # Branch-protection reads require repository administration access.
        # The workflow token only needs pull-request context, so report no
        # required-check names rather than making the whole review unavailable.
        if error.status != 403:
            raise
        branch_protection = {}
    required_checks = branch_protection.get("required_status_checks", {}).get(
        "contexts", []
    )
    combined = _request(config, f"repos/{config.repo_slug}/commits/{head_sha}/status")
    statuses = [
        _sanitize_status_status(status) for status in combined.get("statuses", [])
    ]
    checks = [
        _sanitize_check(check)
        for check in _paginated(
            config,
            f"repos/{config.repo_slug}/commits/{head_sha}/check-runs",
            key="check_runs",
        )
    ]

    summary_files = []
    for file in files:
        summary_files.append(
            {
                "filename": file.get("filename"),
                "status": file.get("status"),
                "additions": file.get("additions"),
                "deletions": file.get("deletions"),
                "changes": file.get("changes"),
            }
        )

    summary_commits = [
        {
            "sha": commit.get("sha"),
            "message": commit.get("commit", {}).get("message"),
            "author": commit.get("commit", {}).get("author", {}).get("name"),
        }
        for commit in commits
    ]

    def _comment(comment: dict[str, Any]) -> dict[str, Any]:
        return {
            "author": (comment.get("user") or {}).get("login"),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "body": comment.get("body"),
            "url": comment.get("html_url"),
        }

    summary_reviews = [
        {
            "author": (review.get("user") or {}).get("login"),
            "state": review.get("state"),
            "submitted_at": review.get("submitted_at"),
            "body": review.get("body"),
            "url": review.get("html_url"),
        }
        for review in reviews
    ]

    summary_review_comments = [
        {
            "author": (item.get("user") or {}).get("login"),
            "path": item.get("path"),
            "line": item.get("line"),
            "original_line": item.get("original_line"),
            "side": item.get("side"),
            "in_reply_to_id": item.get("in_reply_to_id"),
            "created_at": item.get("created_at"),
            "body": item.get("body"),
            "url": item.get("html_url"),
        }
        for item in review_comments
    ]

    return {
        "repository": config.repo_slug,
        "collected_at": datetime.now(UTC).isoformat(),
        "pull_request": {
            "number": pull.get("number"),
            "title": pull.get("title"),
            "body": pull.get("body"),
            "state": pull.get("state"),
            "draft": pull.get("draft"),
            "created_at": pull.get("created_at"),
            "updated_at": pull.get("updated_at"),
            "mergeable_state": pull.get("mergeable_state"),
            "author": (pull.get("user") or {}).get("login"),
            "head": {
                "label": head.get("label"),
                "ref": head.get("ref"),
                "sha": head.get("sha"),
                "repository": (head.get("repo") or {}).get("full_name"),
            },
            "base": {
                "label": base.get("label"),
                "ref": base.get("ref"),
                "sha": base.get("sha"),
                "repository": (base.get("repo") or {}).get("full_name"),
            },
            "labels": labels,
            "requested_reviewers": requested_reviewers,
            "requested_teams": requested_teams,
        },
        "files": files,
        "commits": summary_commits,
        "issue_comments": [_comment(item) for item in issue_comments],
        "reviews": summary_reviews,
        "review_comments": summary_review_comments,
        "checks": {
            "required": required_checks,
            "statuses": statuses,
            "check_runs": checks,
        },
        "summary_files": summary_files,
    }


def _write_summary(data: dict[str, Any], output_dir: Path) -> None:
    pull = data["pull_request"]
    files = data["files"]
    additions = sum(int(file.get("additions") or 0) for file in files)
    deletions = sum(int(file.get("deletions") or 0) for file in files)
    required = data["checks"]["required"]
    check_runs = data["checks"]["check_runs"]
    failed_checks = [
        f"{check.get('name')} ({check.get('conclusion') or check.get('status')})"
        for check in check_runs
        if check.get("conclusion") not in (None, "success", "skipped")
    ]
    lines = [
        "# PR context",
        "",
        f"- Repository: `{data['repository']}`",
        f"- PR: #{pull['number']} — {pull.get('title') or '(untitled)'}",
        f"- Author: `{pull.get('author') or 'unknown'}`",
        f"- State: `{pull.get('state') or 'unknown'}`; draft: `{str(bool(pull.get('draft'))).lower()}`; mergeable_state: `{pull.get('mergeable_state') or 'unknown'}`",
        f"- Head: `{(pull.get('head') or {}).get('sha') or 'unknown'}` (`{(pull.get('head') or {}).get('ref') or 'unknown'}`)",
        f"- Base: `{(pull.get('base') or {}).get('sha') or 'unknown'}` (`{(pull.get('base') or {}).get('ref') or 'unknown'}`)",
        f"- Files changed: {len(files)} (+{additions}/-{deletions})",
        f"- Labels: {', '.join(f'`{label}`' for label in pull.get('labels', [])) or '(none)'}",
        f"- Required checks: {', '.join(f'`{item}`' for item in required) or '(none reported)'}",
        f"- Failed/not-yet-successful checks: {', '.join(failed_checks) if failed_checks else '(none reported)'}",
        f"- Review-level comments: {len(data['review_comments'])} (REST cannot report exact thread resolution)",
        "",
        "## Commits",
        "",
    ]
    for commit in data["commits"]:
        first_line = (
            str(commit.get("message") or "").splitlines()[0]
            if commit.get("message")
            else "(no message)"
        )
        lines.append(f"- `{commit.get('sha')}` {first_line}")

    lines.extend(["", "## Files", ""])
    for file in files:
        lines.append(
            f"- `{file.get('filename')}` — {file.get('status')}, +{file.get('additions', 0)}/-{file.get('deletions', 0)}"
        )

    lines.extend(["", "## PR description", ""])
    lines.append(str(pull.get("body") or "(empty)"))
    lines.extend(["", "## Reviews", ""])
    for review in data["reviews"]:
        if review.get("body"):
            lines.append(
                f"- **{review.get('author')} ({review.get('state')})**: {review['body']}"
            )
    lines.extend(["", "## Issue comments", ""])
    for comment in data["issue_comments"]:
        if comment.get("body"):
            lines.append(f"- **{comment.get('author')}**: {comment['body']}")

    (output_dir / "summary.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def _write_diff(files: list[dict[str, Any]], path: Path) -> None:
    chunks: list[str] = []
    for file in files:
        filename = str(file.get("filename") or "unknown")
        header = f"diff --git a/{filename} b/{filename}\nindex unknown"
        patch = _truncate_patch(file.get("patch"))
        chunks.append(
            f"{header}\n--- a/{filename}\n+++ b/{filename}\n{patch if patch is not None else 'Binary file or patch unavailable'}\n"
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


def collect(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = _collect(config)
    _write_json(config.output_dir / "pr.json", data["pull_request"])
    _write_json(config.output_dir / "files.json", data["files"])
    _write_json(
        config.output_dir / "comments.json",
        {
            "issue_comments": data["issue_comments"],
            "review_comments": data["review_comments"],
        },
    )
    _write_json(config.output_dir / "reviews.json", data["reviews"])
    _write_json(config.output_dir / "checks.json", data["checks"])
    _write_json(config.output_dir / "commits.json", data["commits"])
    _write_diff(data["files"], config.output_dir / "diff.patch")
    _write_summary(data, config.output_dir)


def _env_config() -> Config | None:
    slug = os.environ.get("GITHUB_REPOSITORY")
    pr_ref = os.environ.get("GITHUB_REF", "")
    match = re.fullmatch(r"refs/pull/([0-9]+)/merge", pr_ref)
    if not slug or not match:
        return None
    owner, repo = slug.split("/", 1)
    return Config(
        owner=owner,
        repo=repo,
        pr_number=int(match.group(1)),
        output_dir=Path("context"),
        token=os.environ.get("GITHUB_TOKEN"),
        api_root=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )


def main(argv: list[str] | None = None) -> int:
    env_config = _env_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=env_config.owner if env_config else None)
    parser.add_argument("--repo", default=env_config.repo if env_config else None)
    parser.add_argument(
        "--pr", type=int, default=env_config.pr_number if env_config else None
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=env_config.output_dir if env_config else Path("context"),
    )
    parser.add_argument(
        "--api-root", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not all((args.owner, args.repo, args.pr)):
        parser.error("owner, repo, and pr are required (or run inside GitHub Actions)")
    if token == "":
        token = None
    config = Config(args.owner, args.repo, args.pr, args.output, token, args.api_root)

    try:
        collect(config)
    except ApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
