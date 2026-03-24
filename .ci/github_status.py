#!/usr/bin/env python3
"""GitHub Commit Status API wrapper using urllib (zero external dependencies)."""

import json
import os
import re
import sys
import urllib.error
import urllib.request


def parse_repo_url(url):
    """Extract (owner, repo) from a GitHub URL.

    Handles:
      - https://github.com/Owner/Repo.git
      - git@github.com:Owner/Repo.git
    """
    # HTTPS format
    m = re.match(r"https?://[^/]+/([^/]+)/([^/]+?)(?:\.git)?$", url)

    if m:
        return m.group(1), m.group(2)

    # SSH format
    m = re.match(r"git@[^:]+:([^/]+)/([^/]+?)(?:\.git)?$", url)

    if m:
        return m.group(1), m.group(2)

    return "", ""


def build_status_context(prefix, job_name):
    """Build status context string, e.g. 'ci/infiniops/nvidia_gpu'."""
    return f"{prefix}/{job_name}"


def post_commit_status(
    owner,
    repo,
    sha,
    state,
    context,
    description,
    target_url=None,
    token=None,
):
    """Post a commit status to GitHub.

    Args:
        state: One of 'pending', 'success', 'failure', 'error'.
        Returns True on success, False on failure.
    """
    token = token or os.environ.get("GITHUB_TOKEN", "")

    if not token:
        print("warning: GITHUB_TOKEN not set, skipping status update", file=sys.stderr)
        return False

    if not owner or not repo or not sha:
        print(
            "warning: missing owner/repo/sha, skipping status update", file=sys.stderr
        )
        return False

    url = f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"
    body = {
        "state": state,
        "context": context,
        "description": description[:140],
    }

    if target_url:
        body["target_url"] = target_url

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        print(
            f"warning: GitHub status API returned {e.code}: {e.reason}",
            file=sys.stderr,
        )
        return False
    except urllib.error.URLError as e:
        print(f"warning: GitHub status API error: {e.reason}", file=sys.stderr)
        return False
