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


def test_has_dockerfile_changed_true_when_stdout_nonempty(mocker):
    mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(returncode=0, stdout="Dockerfile\n"),
    )
    assert build.has_dockerfile_changed(".ci/images/nvidia/") is True


def test_has_dockerfile_changed_false_when_stdout_empty(mocker):
    mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(returncode=0, stdout=""),
    )
    assert build.has_dockerfile_changed(".ci/images/nvidia/") is False


def test_has_dockerfile_changed_true_on_git_error(mocker):
    # Shallow clone or initial commit: `git diff` returns non-zero.
    mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(returncode=128, stdout=""),
    )
    assert build.has_dockerfile_changed(".ci/images/nvidia/") is True


# ---------------------------------------------------------------------------
# Tests for `docker_login`.
# ---------------------------------------------------------------------------


def test_docker_login_no_credentials_env(mocker):
    run_mock = mocker.patch("subprocess.run")
    result = build.docker_login({"url": "localhost:5000"}, dry_run=False)
    assert result is True
    run_mock.assert_not_called()


def test_docker_login_token_not_set(mocker, monkeypatch, capsys):
    monkeypatch.delenv("REGISTRY_TOKEN", raising=False)
    run_mock = mocker.patch("subprocess.run")
    cfg = {"url": "localhost:5000", "credentials_env": "REGISTRY_TOKEN"}
    result = build.docker_login(cfg, dry_run=False)
    assert result is False
    run_mock.assert_not_called()


def test_docker_login_dry_run_does_not_call_subprocess(mocker, monkeypatch):
    monkeypatch.setenv("REGISTRY_TOKEN", "mytoken")
    run_mock = mocker.patch("subprocess.run")
    cfg = {"url": "localhost:5000", "credentials_env": "REGISTRY_TOKEN"}
    result = build.docker_login(cfg, dry_run=True)
    assert result is True
    run_mock.assert_not_called()


def test_docker_login_success(mocker, monkeypatch):
    monkeypatch.setenv("REGISTRY_TOKEN", "mytoken")
    run_mock = mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(returncode=0),
    )
    cfg = {"url": "localhost:5000", "credentials_env": "REGISTRY_TOKEN"}
    result = build.docker_login(cfg, dry_run=False)
    assert result is True
    run_mock.assert_called_once()
    cmd = run_mock.call_args[0][0]
    assert "docker" in cmd
    assert "login" in cmd


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


def test_build_image_dry_run_no_subprocess(mocker, monkeypatch, capsys):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    run_mock = mocker.patch("subprocess.run")
    build.build_image(
        "nvidia",
        _platform_cfg(),
        _registry_cfg(),
        "abc1234",
        push=False,
        dry_run=True,
        logged_in=True,
    )
    run_mock.assert_not_called()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


def test_build_image_dry_run_output_contains_image_tag(mocker, monkeypatch, capsys):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    mocker.patch("subprocess.run")
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


def test_build_image_proxy_in_build_args(mocker, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.test:3128")
    run_mock = mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(returncode=0),
    )
    build.build_image(
        "nvidia",
        _platform_cfg(),
        _registry_cfg(),
        "abc1234",
        push=False,
        dry_run=False,
        logged_in=True,
    )
    called_cmd = run_mock.call_args[0][0]
    joined = " ".join(called_cmd)
    assert "HTTP_PROXY=http://proxy.test:3128" in joined
    assert "http_proxy=http://proxy.test:3128" in joined


def test_build_image_returns_false_on_docker_error(mocker, monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(returncode=1),
    )
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
