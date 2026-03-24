import json
from unittest.mock import MagicMock


import github_status as gh


# ---------------------------------------------------------------------------
# parse_repo_url
# ---------------------------------------------------------------------------


def test_parse_repo_url_https():
    owner, repo = gh.parse_repo_url("https://github.com/InfiniTensor/InfiniOps.git")
    assert owner == "InfiniTensor"
    assert repo == "InfiniOps"


def test_parse_repo_url_https_no_git():
    owner, repo = gh.parse_repo_url("https://github.com/Owner/Repo")
    assert owner == "Owner"
    assert repo == "Repo"


def test_parse_repo_url_ssh():
    owner, repo = gh.parse_repo_url("git@github.com:Owner/Repo.git")
    assert owner == "Owner"
    assert repo == "Repo"


def test_parse_repo_url_invalid():
    owner, repo = gh.parse_repo_url("not-a-url")
    assert owner == ""
    assert repo == ""


# ---------------------------------------------------------------------------
# build_status_context
# ---------------------------------------------------------------------------


def test_build_status_context():
    ctx = gh.build_status_context("ci/infiniops", "nvidia_gpu")
    assert ctx == "ci/infiniops/nvidia_gpu"


# ---------------------------------------------------------------------------
# post_commit_status
# ---------------------------------------------------------------------------


def test_post_status_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = gh.post_commit_status("owner", "repo", "abc123", "success", "ctx", "desc")
    assert result is False


def test_post_status_missing_owner():
    result = gh.post_commit_status(
        "", "repo", "abc123", "success", "ctx", "desc", token="tok"
    )
    assert result is False


def test_post_status_success(monkeypatch):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    captured_req = {}

    def mock_urlopen(req, **kwargs):
        captured_req["url"] = req.full_url
        captured_req["data"] = json.loads(req.data)
        captured_req["headers"] = dict(req.headers)
        return mock_response

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    result = gh.post_commit_status(
        "InfiniTensor",
        "InfiniOps",
        "abc123def",
        "success",
        "ci/infiniops/nvidia_gpu",
        "Tests passed",
        token="ghp_test_token",
    )

    assert result is True
    assert "abc123def" in captured_req["url"]
    assert captured_req["data"]["state"] == "success"
    assert captured_req["data"]["context"] == "ci/infiniops/nvidia_gpu"
    assert "ghp_test_token" in captured_req["headers"]["Authorization"]


def test_post_status_http_error(monkeypatch):
    import urllib.error

    def mock_urlopen(req, **kwargs):
        raise urllib.error.HTTPError(
            url="", code=422, msg="Unprocessable", hdrs=None, fp=None
        )

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    result = gh.post_commit_status(
        "owner", "repo", "sha", "success", "ctx", "desc", token="tok"
    )
    assert result is False


def test_post_status_url_error(monkeypatch):
    import urllib.error

    def mock_urlopen(req, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    result = gh.post_commit_status(
        "owner", "repo", "sha", "success", "ctx", "desc", token="tok"
    )
    assert result is False


def test_post_status_truncates_description(monkeypatch):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    captured = {}

    def mock_urlopen(req, **kwargs):
        captured["data"] = json.loads(req.data)
        return mock_response

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    long_desc = "x" * 200
    gh.post_commit_status("o", "r", "sha", "success", "ctx", long_desc, token="tok")

    assert len(captured["data"]["description"]) == 140
