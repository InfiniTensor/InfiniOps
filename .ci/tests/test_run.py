from pathlib import Path

import pytest

import run


# ---------------------------------------------------------------------------
# Tests for `resolve_image`.
# ---------------------------------------------------------------------------


def test_resolve_image_with_registry():
    cfg = {"registry": {"url": "localhost:5000", "project": "infiniops"}}
    img = run.resolve_image(cfg, "nvidia", "latest")
    assert img == "localhost:5000/infiniops/nvidia:latest"


def test_resolve_image_without_registry(minimal_config):
    img = run.resolve_image(minimal_config, "nvidia", "abc1234")
    assert img == "infiniops-ci/nvidia:abc1234"


# ---------------------------------------------------------------------------
# Tests for `build_runner_script`.
# ---------------------------------------------------------------------------


def test_runner_script_contains_git_clone():
    script = run.build_runner_script()
    assert "git clone" in script


def test_runner_script_contains_setup_cmd():
    script = run.build_runner_script()
    assert "SETUP_CMD" in script


def test_runner_script_exits_on_failure():
    script = run.build_runner_script()
    assert "exit $rc" in script


def test_runner_script_creates_results_dir():
    script = run.build_runner_script()
    assert "mkdir -p /workspace/results" in script


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` basic structure.
# ---------------------------------------------------------------------------


def test_docker_args_basic_structure(minimal_config):
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
    )
    assert args[0] == "docker"
    assert "run" in args
    assert "--rm" in args


def test_docker_args_correct_image(minimal_config):
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
    )
    assert "infiniops-ci/nvidia:latest" in args


def test_docker_args_image_tag_override(minimal_config):
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        "abc1234",
    )
    assert "infiniops-ci/nvidia:abc1234" in args


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` proxy passthrough.
# ---------------------------------------------------------------------------


def test_docker_args_proxy_present_when_set(minimal_config, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080")
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
    )
    assert "-e" in args
    assert "HTTP_PROXY=http://proxy.example.com:8080" in args
    assert "http_proxy=http://proxy.example.com:8080" in args


def test_docker_args_proxy_absent_when_not_set(minimal_config, monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
    )

    for arg in args:
        assert not arg.startswith("HTTP_PROXY=")
        assert not arg.startswith("http_proxy=")
        assert not arg.startswith("HTTPS_PROXY=")
        assert not arg.startswith("https_proxy=")
        assert not arg.startswith("NO_PROXY=")
        assert not arg.startswith("no_proxy=")


def test_docker_args_proxy_lowercase_fallback(minimal_config, monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setenv("http_proxy", "http://lowercase.proxy:3128")
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
    )
    assert "HTTP_PROXY=http://lowercase.proxy:3128" in args
    assert "http_proxy=http://lowercase.proxy:3128" in args


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` GPU flags.
# ---------------------------------------------------------------------------


def _make_args(config, gpu_id_override=None):
    return run.build_docker_args(
        config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
        gpu_id_override=gpu_id_override,
    )


def test_docker_args_gpu_auto_no_override(minimal_config):
    """gpu_ids=auto (default) without override produces no --gpus flag."""
    args = _make_args(minimal_config)
    assert "--gpus" not in args


def test_docker_args_gpu_auto_with_override(minimal_config):
    """gpu_ids=auto with allocator override sets --gpus device=..."""
    args = _make_args(minimal_config, gpu_id_override="2")
    idx = args.index("--gpus")
    assert args[idx + 1] == "device=2"


def test_docker_args_gpu_static(minimal_config):
    """Static gpu_ids pins to specific devices."""
    minimal_config["jobs"]["nvidia_gpu"]["resources"]["gpu_ids"] = "0"
    args = _make_args(minimal_config)
    idx = args.index("--gpus")
    assert args[idx + 1] == "device=0"


def test_docker_args_gpu_all(minimal_config):
    minimal_config["jobs"]["nvidia_gpu"]["resources"]["gpu_ids"] = "all"
    args = _make_args(minimal_config)
    idx = args.index("--gpus")
    assert args[idx + 1] == "all"


def test_docker_args_gpu_override_trumps_static(minimal_config):
    """CLI gpu_id_override takes precedence over static gpu_ids."""
    minimal_config["jobs"]["nvidia_gpu"]["resources"]["gpu_ids"] = "0"
    args = _make_args(minimal_config, gpu_id_override="2,3")
    idx = args.index("--gpus")
    assert args[idx + 1] == "device=2,3"


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` platform-specific device env vars.
# ---------------------------------------------------------------------------


def _make_platform_config(platform, job_suffix="gpu"):
    """Build a minimal normalized config for a given platform."""
    from utils import normalize_config

    raw = {
        "platforms": {
            platform: {
                "image": {"dockerfile": f".ci/images/{platform}/"},
                "setup": "pip install .[dev]",
                "jobs": {
                    job_suffix: {
                        "resources": {"ngpus": 1, "memory": "32GB"},
                        "stages": [{"name": "test", "run": "pytest tests/ -v"}],
                    }
                },
            }
        }
    }

    return normalize_config(raw)


def _make_platform_args(platform, job_suffix="gpu", gpu_id_override=None):
    config = _make_platform_config(platform, job_suffix)
    job_name = f"{platform}_{job_suffix}"

    return run.build_docker_args(
        config,
        job_name,
        "https://github.com/example/repo.git",
        "master",
        config["jobs"][job_name]["stages"],
        "/workspace",
        None,
        gpu_id_override=gpu_id_override,
    )


def test_docker_args_moore_mthreads_visible_devices():
    """Moore uses MTHREADS_VISIBLE_DEVICES, not CUDA_VISIBLE_DEVICES."""
    args = _make_platform_args("moore", gpu_id_override="0")
    assert "MTHREADS_VISIBLE_DEVICES=0" in args
    assert all("CUDA_VISIBLE_DEVICES" not in a for a in args)


def test_docker_args_iluvatar_cuda_visible_devices():
    args = _make_platform_args("iluvatar", gpu_id_override="1,2")
    assert "CUDA_VISIBLE_DEVICES=1,2" in args


def test_docker_args_cambricon_mlu_visible_devices():
    args = _make_platform_args("cambricon", gpu_id_override="0")
    assert "MLU_VISIBLE_DEVICES=0" in args


def test_docker_args_ascend_visible_devices():
    args = _make_platform_args("ascend", job_suffix="npu", gpu_id_override="0")
    assert "ASCEND_VISIBLE_DEVICES=0" in args


def test_docker_args_metax_cuda_visible_devices():
    args = _make_platform_args("metax", gpu_id_override="0,1")
    assert "CUDA_VISIBLE_DEVICES=0,1" in args


def test_docker_args_non_nvidia_no_gpus_flag():
    """Non-NVIDIA platforms should never use --gpus Docker flag."""
    for platform in ("iluvatar", "metax", "moore", "cambricon"):
        args = _make_platform_args(platform, gpu_id_override="0")
        assert "--gpus" not in args


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` memory format.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("32GB", "32g"),
        ("512MB", "512m"),
        ("8", "8g"),
        ("16gb", "16g"),
        ("256mb", "256m"),
    ],
)
def test_docker_args_memory_format(minimal_config, raw, expected):
    minimal_config["jobs"]["nvidia_gpu"]["resources"]["memory"] = raw
    args = _make_args(minimal_config)
    idx = args.index("--memory")
    assert args[idx + 1] == expected


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` stages encoding.
# ---------------------------------------------------------------------------


def test_docker_args_num_stages(minimal_config):
    args = _make_args(minimal_config)
    assert "NUM_STAGES=1" in args


def test_docker_args_stage_name_cmd(minimal_config):
    args = _make_args(minimal_config)
    assert "STAGE_1_NAME=test" in args
    assert any(a.startswith("STAGE_1_CMD=") for a in args)


def test_docker_args_multiple_stages(minimal_config):
    minimal_config["jobs"]["nvidia_gpu"]["stages"] = [
        {"name": "lint", "run": "ruff check ."},
        {"name": "test", "run": "pytest tests/"},
    ]
    args = _make_args(minimal_config)
    assert "NUM_STAGES=2" in args
    assert "STAGE_1_NAME=lint" in args
    assert "STAGE_2_NAME=test" in args


# ---------------------------------------------------------------------------
# Tests for `build_docker_args` `results_dir` mount.
# ---------------------------------------------------------------------------


def test_docker_args_results_dir(minimal_config, tmp_path):
    args = run.build_docker_args(
        minimal_config,
        "nvidia_gpu",
        "https://github.com/example/repo.git",
        "master",
        minimal_config["jobs"]["nvidia_gpu"]["stages"],
        "/workspace",
        None,
        results_dir=tmp_path,
    )
    joined = " ".join(str(a) for a in args)
    assert "-v" in args
    assert "/workspace/results" in joined


# ---------------------------------------------------------------------------
# Tests for `build_results_dir`.
# ---------------------------------------------------------------------------


def test_build_results_dir_contains_platform():
    stages = [{"name": "test", "run": "pytest"}]
    d = run.build_results_dir("ci-results", "nvidia", stages, "abc1234")
    assert "nvidia" in d.name


def test_build_results_dir_contains_commit():
    stages = [{"name": "test", "run": "pytest"}]
    d = run.build_results_dir("ci-results", "nvidia", stages, "abc1234")
    assert "abc1234" in d.name


def test_build_results_dir_contains_stage_names():
    stages = [{"name": "lint", "run": "ruff"}, {"name": "test", "run": "pytest"}]
    d = run.build_results_dir("ci-results", "nvidia", stages, "abc1234")
    assert "lint+test" in d.name


def test_build_results_dir_under_base():
    stages = [{"name": "test", "run": "pytest"}]
    d = run.build_results_dir("/tmp/my-results", "ascend", stages, "def5678")
    assert d.parent == Path("/tmp/my-results")



# ---------------------------------------------------------------------------
# Tests for `apply_test_override`.
# ---------------------------------------------------------------------------


def test_apply_test_override_replaces_test_path():
    result = run.apply_test_override("pytest tests/ -v", "tests/test_add.py")
    assert result == "pytest tests/test_add.py -v"


def test_apply_test_override_preserves_flags():
    result = run.apply_test_override(
        "pytest tests/ -n 4 -v --tb=short", "tests/test_gemm.py"
    )
    assert "tests/test_gemm.py" in result
    assert "-n 4" in result
    assert "-v" in result
    assert "--tb=short" in result
    assert "tests/" not in result.split("tests/test_gemm.py")[0]


def test_apply_test_override_non_pytest_passthrough():
    """Non-pytest commands are returned unchanged."""
    assert run.apply_test_override("ruff check .", "tests/foo.py") == "ruff check ."


def test_apply_test_override_empty_passthrough():
    assert run.apply_test_override("", "tests/foo.py") == ""


# ---------------------------------------------------------------------------
# Tests for runner script fail-fast behavior.
# ---------------------------------------------------------------------------


def test_runner_script_breaks_on_failure():
    script = run.build_runner_script()
    assert "break" in script


def test_runner_script_preserves_exit_code():
    script = run.build_runner_script()
    assert "rc=$?" in script


# ---------------------------------------------------------------------------
# Tests for `build_results_dir` uniqueness and sanitization.
# ---------------------------------------------------------------------------


def test_build_results_dir_unique():
    stages = [{"name": "test", "run": "pytest"}]
    d1 = run.build_results_dir("ci-results", "nvidia", stages, "abc1234")
    d2 = run.build_results_dir("ci-results", "nvidia", stages, "abc1234")
    assert d1 != d2


def test_build_results_dir_sanitizes_commit():
    stages = [{"name": "test", "run": "pytest"}]
    d = run.build_results_dir("ci-results", "nvidia", stages, "../../etc/passwd")
    # Path separators are stripped; the result stays under the base directory.
    assert "/" not in d.name
    assert d.parent == Path("ci-results")
