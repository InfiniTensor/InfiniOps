from unittest.mock import MagicMock

import build


# ---------------------------------------------------------------------------
# Tests for `build_image_tag`.
# ---------------------------------------------------------------------------


def test_build_image_tag_with_registry():
    tag = build.build_image_tag("localhost:5000", "infiniops", "nvidia", "latest")
    assert tag == "localhost:5000/infiniops/nvidia:latest"


def test_build_image_tag_without_registry():
    tag = build.build_image_tag("", "infiniops", "nvidia", "abc1234")
    assert tag == "infiniops-ci/nvidia:abc1234"


def test_build_image_tag_commit_hash():
    tag = build.build_image_tag(
        "registry.example.com:5000", "proj", "ascend", "deadbeef"
    )
    assert tag == "registry.example.com:5000/proj/ascend:deadbeef"


# ---------------------------------------------------------------------------
# Tests for `has_dockerfile_changed`.
# ---------------------------------------------------------------------------


def test_has_dockerfile_changed_true_when_stdout_nonempty(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: MagicMock(returncode=0, stdout="Dockerfile\n"),
    )
    assert build.has_dockerfile_changed(".ci/images/nvidia/") is True


def test_has_dockerfile_changed_false_when_stdout_empty(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: MagicMock(returncode=0, stdout=""),
    )
    assert build.has_dockerfile_changed(".ci/images/nvidia/") is False


def test_has_dockerfile_changed_true_on_git_error(monkeypatch):
    # Shallow clone or initial commit: `git diff` returns non-zero.
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: MagicMock(returncode=128, stdout=""),
    )
    assert build.has_dockerfile_changed(".ci/images/nvidia/") is True


# ---------------------------------------------------------------------------
# Tests for `docker_login`.
# ---------------------------------------------------------------------------


def test_docker_login_no_credentials_env(monkeypatch):
    called = []
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(1))
    result = build.docker_login({"url": "localhost:5000"}, dry_run=False)
    assert result is True
    assert not called


def test_docker_login_token_not_set(monkeypatch):
    monkeypatch.delenv("REGISTRY_TOKEN", raising=False)
    called = []
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(1))
    cfg = {"url": "localhost:5000", "credentials_env": "REGISTRY_TOKEN"}
    result = build.docker_login(cfg, dry_run=False)
    assert result is False
    assert not called


def test_docker_login_dry_run_does_not_call_subprocess(monkeypatch):
    monkeypatch.setenv("REGISTRY_TOKEN", "mytoken")
    called = []
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(1))
    cfg = {"url": "localhost:5000", "credentials_env": "REGISTRY_TOKEN"}
    result = build.docker_login(cfg, dry_run=True)
    assert result is True
    assert not called


def test_docker_login_success(monkeypatch):
    monkeypatch.setenv("REGISTRY_TOKEN", "mytoken")
    captured = {}

    def mock_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_run)
    cfg = {"url": "localhost:5000", "credentials_env": "REGISTRY_TOKEN"}
    result = build.docker_login(cfg, dry_run=False)
    assert result is True
    assert "docker" in captured["cmd"]
    assert "login" in captured["cmd"]


# ---------------------------------------------------------------------------
# Tests for `build_image` dry-run mode and proxy forwarding.
# ---------------------------------------------------------------------------


def _platform_cfg():
    return {
        "dockerfile": ".ci/images/nvidia/",
        "build_args": {"BASE_IMAGE": "nvcr.io/nvidia/pytorch:24.10-py3"},
    }


def _registry_cfg():
    return {"url": "localhost:5000", "project": "infiniops"}


def test_build_image_dry_run_no_subprocess(monkeypatch, capsys):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    called = []
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(1))
    build.build_image(
        "nvidia",
        _platform_cfg(),
        _registry_cfg(),
        "abc1234",
        push=False,
        dry_run=True,
        logged_in=True,
    )
    assert not called
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


def test_build_image_dry_run_output_contains_image_tag(monkeypatch, capsys):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))
    build.build_image(
        "nvidia",
        _platform_cfg(),
        _registry_cfg(),
        "abc1234",
        push=False,
        dry_run=True,
        logged_in=True,
    )
    captured = capsys.readouterr()
    assert "abc1234" in captured.out


def test_build_image_proxy_in_build_args(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.test:3128")
    captured = {}

    def mock_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_run)
    build.build_image(
        "nvidia",
        _platform_cfg(),
        _registry_cfg(),
        "abc1234",
        push=False,
        dry_run=False,
        logged_in=True,
    )
    joined = " ".join(captured["cmd"])
    assert "HTTP_PROXY=http://proxy.test:3128" in joined
    assert "http_proxy=http://proxy.test:3128" in joined


def test_build_image_returns_false_on_docker_error(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=1))
    result = build.build_image(
        "nvidia",
        _platform_cfg(),
        _registry_cfg(),
        "abc1234",
        push=False,
        dry_run=False,
        logged_in=True,
    )
    assert result is False
