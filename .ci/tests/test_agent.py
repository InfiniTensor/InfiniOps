import hashlib
import hmac
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent
import ci_resource as res
from utils import normalize_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config():
    raw = {
        "repo": {
            "url": "https://github.com/InfiniTensor/InfiniOps.git",
            "branch": "master",
        },
        "github": {
            "status_context_prefix": "ci/infiniops",
        },
        "agents": {
            "nvidia": {"url": "http://nvidia-host:8080"},
            "iluvatar": {"url": "http://iluvatar-host:8080"},
        },
        "platforms": {
            "nvidia": {
                "image": {
                    "dockerfile": ".ci/images/nvidia/",
                    "build_args": {"BASE_IMAGE": "nvcr.io/nvidia/pytorch:24.10-py3"},
                },
                "setup": "pip install .[dev]",
                "jobs": {
                    "gpu": {
                        "resources": {
                            "gpu_ids": "0",
                            "memory": "32GB",
                            "shm_size": "16g",
                            "timeout": 3600,
                        },
                        "stages": [{"name": "test", "run": "pytest tests/ -v"}],
                    },
                },
            },
            "iluvatar": {
                "image": {
                    "dockerfile": ".ci/images/iluvatar/",
                    "build_args": {"BASE_IMAGE": "corex:qs_pj20250825"},
                },
                "setup": "pip install .[dev]",
                "jobs": {
                    "gpu": {
                        "resources": {
                            "gpu_ids": "0",
                            "gpu_style": "none",
                            "memory": "32GB",
                            "shm_size": "16g",
                            "timeout": 3600,
                        },
                        "stages": [{"name": "test", "run": "pytest tests/ -v"}],
                    },
                },
            },
        },
    }
    return normalize_config(raw)


@pytest.fixture
def mock_resource_pool():
    pool = MagicMock(spec=res.ResourcePool)
    pool.platform = "nvidia"
    pool.allocate.return_value = ([0], True)
    pool.release.return_value = None
    pool.get_status.return_value = {"platform": "nvidia", "gpus": [], "allocated_gpu_ids": [], "system": {}}
    return pool


# ---------------------------------------------------------------------------
# select_jobs
# ---------------------------------------------------------------------------


def test_select_jobs_by_name(agent_config):
    jobs = agent.select_jobs(agent_config, job_name="nvidia_gpu")
    assert jobs == ["nvidia_gpu"]


def test_select_jobs_by_platform(agent_config):
    jobs = agent.select_jobs(agent_config, platform="nvidia")
    assert jobs == ["nvidia_gpu"]


def test_select_jobs_by_platform_iluvatar(agent_config):
    jobs = agent.select_jobs(agent_config, platform="iluvatar")
    assert jobs == ["iluvatar_gpu"]


def test_select_jobs_all(agent_config):
    jobs = agent.select_jobs(agent_config)
    assert set(jobs) == {"nvidia_gpu", "iluvatar_gpu"}


def test_select_jobs_invalid_name(agent_config):
    with pytest.raises(ValueError, match="not_exist"):
        agent.select_jobs(agent_config, job_name="not_exist")



# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


def test_verify_signature_valid():
    secret = "my-secret"
    body = b'{"action": "push"}'
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert agent.verify_signature(secret, body, sig) is True


def test_verify_signature_invalid():
    assert agent.verify_signature("secret", b"body", "sha256=wrong") is False


def test_verify_signature_empty():
    assert agent.verify_signature("secret", b"body", "") is False


# ---------------------------------------------------------------------------
# JobRequest / JobResult
# ---------------------------------------------------------------------------


def test_job_request_fields(agent_config):
    req = agent.JobRequest("nvidia_gpu", "master", "abc123", agent_config)
    assert req.job_name == "nvidia_gpu"
    assert req.platform == "nvidia"
    assert req.commit_sha == "abc123"
    assert len(req.job_id) == 8
    d = req.to_dict()
    assert d["job_name"] == "nvidia_gpu"


def test_job_result_success():
    r = agent.JobResult("id1", "nvidia_gpu", "abc", 0, Path("/tmp/res"), 42.5)
    assert r.state == "success"


def test_job_result_failure():
    r = agent.JobResult("id1", "nvidia_gpu", "abc", 1, Path("/tmp/res"), 10.0)
    assert r.state == "failure"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_submit_and_run(agent_config, mock_resource_pool, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: MagicMock(returncode=0))
    monkeypatch.setattr("agent.gh.post_commit_status", lambda *a, **kw: True)

    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        results_dir=Path("/tmp/test-results"),
        no_status=True, dry_run=True,
    )
    req = agent.JobRequest("nvidia_gpu", "master", "abc123", agent_config,
                           results_dir=Path("/tmp/test-results"))
    jid = scheduler.submit(req)
    results = scheduler.wait_all()
    assert len(results) == 1
    assert results[0].state == "success"


def test_scheduler_queues_when_no_resources(agent_config, monkeypatch):
    pool = MagicMock(spec=res.ResourcePool)
    pool.allocate.return_value = ([], False)
    pool.get_status.return_value = {"platform": "nvidia", "gpus": [], "allocated_gpu_ids": [], "system": {}}

    scheduler = agent.Scheduler(
        agent_config, "nvidia", pool,
        no_status=True, dry_run=False,
    )

    req = agent.JobRequest("nvidia_gpu", "master", "abc123", agent_config)
    scheduler.submit(req)

    info = scheduler.get_job(req.job_id)
    assert info["state"] == "queued"


def test_scheduler_get_status(agent_config, mock_resource_pool):
    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True, dry_run=True,
    )

    status = scheduler.get_status()
    assert "queued" in status
    assert "running" in status
    assert "completed" in status
    assert "resources" in status


# ---------------------------------------------------------------------------
# WebhookHandler — push event parsing
# ---------------------------------------------------------------------------


def test_webhook_parse_push():
    handler = agent.WebhookHandler.__new__(agent.WebhookHandler)
    payload = {"ref": "refs/heads/feat/test", "after": "abc123def456"}
    branch, sha = handler._parse_push(payload)
    assert branch == "feat/test"
    assert sha == "abc123def456"


def test_webhook_parse_pr():
    handler = agent.WebhookHandler.__new__(agent.WebhookHandler)
    payload = {
        "pull_request": {
            "head": {
                "ref": "feat/pr-branch",
                "sha": "def789",
            }
        }
    }
    branch, sha = handler._parse_pull_request(payload)
    assert branch == "feat/pr-branch"
    assert sha == "def789"


# ---------------------------------------------------------------------------
# Integration-style: webhook HTTP test
# ---------------------------------------------------------------------------


def _urlopen_no_proxy(url_or_req, **kwargs):
    """urlopen that bypasses any HTTP_PROXY."""
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url_or_req, **kwargs)


def test_health_endpoint(agent_config, mock_resource_pool):
    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True,
    )
    server = agent.AgentServer(
        "127.0.0.1", 0, agent_config, scheduler, "nvidia",
    )
    port = server.server_address[1]

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    try:
        resp = _urlopen_no_proxy(f"http://127.0.0.1:{port}/health", timeout=5)
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert data["platform"] == "nvidia"
    finally:
        server.server_close()


def test_api_run_endpoint(agent_config, mock_resource_pool, monkeypatch):
    monkeypatch.setattr("agent.gh.post_commit_status", lambda *a, **kw: True)

    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True, dry_run=True,
    )
    server = agent.AgentServer(
        "127.0.0.1", 0, agent_config, scheduler, "nvidia",
        results_dir=Path("/tmp/test-results"),
    )
    port = server.server_address[1]

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import urllib.request

    body = json.dumps({"branch": "master", "commit_sha": "abc123"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/run",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = _urlopen_no_proxy(req, timeout=5)
        data = json.loads(resp.read())
        assert data["accepted"] is True
        assert len(data["job_ids"]) >= 1
    finally:
        server.server_close()


def test_webhook_with_signature(agent_config, mock_resource_pool, monkeypatch):
    monkeypatch.setattr("agent.gh.post_commit_status", lambda *a, **kw: True)

    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True, dry_run=True,
    )
    secret = "test-secret"
    server = agent.AgentServer(
        "127.0.0.1", 0, agent_config, scheduler, "nvidia",
        webhook_secret=secret,
        results_dir=Path("/tmp/test-results"),
    )
    port = server.server_address[1]

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import urllib.request

    payload = json.dumps({
        "ref": "refs/heads/master",
        "after": "abc123def456",
    }).encode()
    sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/webhook",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": sig,
        },
    )

    try:
        resp = _urlopen_no_proxy(req, timeout=5)
        data = json.loads(resp.read())
        assert data["accepted"] is True
    finally:
        server.server_close()


def test_webhook_invalid_signature(agent_config, mock_resource_pool):
    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True,
    )
    server = agent.AgentServer(
        "127.0.0.1", 0, agent_config, scheduler, "nvidia",
        webhook_secret="real-secret",
    )
    port = server.server_address[1]

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import urllib.error
    import urllib.request

    payload = b'{"ref": "refs/heads/master", "after": "abc"}'
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/webhook",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": "sha256=invalid",
        },
    )

    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _urlopen_no_proxy(req, timeout=5)

        assert exc_info.value.code == 401
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# API token authentication
# ---------------------------------------------------------------------------


def test_api_run_requires_token(agent_config, mock_resource_pool, monkeypatch):
    """When api_token is set, /api/run rejects requests without valid token."""
    monkeypatch.setattr("agent.gh.post_commit_status", lambda *a, **kw: True)

    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True, dry_run=True,
    )
    server = agent.AgentServer(
        "127.0.0.1", 0, agent_config, scheduler, "nvidia",
        api_token="my-secret-token",
        results_dir=Path("/tmp/test-results"),
    )
    port = server.server_address[1]

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import urllib.error
    import urllib.request

    body = json.dumps({"branch": "master", "commit_sha": "abc123"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/run",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _urlopen_no_proxy(req, timeout=5)

        assert exc_info.value.code == 401
    finally:
        server.server_close()


def test_api_run_accepts_valid_token(agent_config, mock_resource_pool, monkeypatch):
    """When api_token is set, /api/run accepts requests with correct Bearer token."""
    monkeypatch.setattr("agent.gh.post_commit_status", lambda *a, **kw: True)

    scheduler = agent.Scheduler(
        agent_config, "nvidia", mock_resource_pool,
        no_status=True, dry_run=True,
    )
    server = agent.AgentServer(
        "127.0.0.1", 0, agent_config, scheduler, "nvidia",
        api_token="my-secret-token",
        results_dir=Path("/tmp/test-results"),
    )
    port = server.server_address[1]

    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import urllib.request

    body = json.dumps({"branch": "master", "commit_sha": "abc123"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/run",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer my-secret-token",
        },
    )

    try:
        resp = _urlopen_no_proxy(req, timeout=5)
        data = json.loads(resp.read())
        assert data["accepted"] is True
    finally:
        server.server_close()
