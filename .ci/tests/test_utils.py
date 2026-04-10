from utils import get_git_commit, normalize_config


def test_normalize_creates_flat_jobs():
    raw = {
        "repo": {"url": "https://github.com/org/repo.git"},
        "platforms": {
            "nvidia": {
                "image": {"dockerfile": ".ci/images/nvidia/"},
                "setup": "pip install .",
                "docker_args": ["--gpus", "all"],
                "jobs": {
                    "gpu": {
                        "resources": {"gpu_ids": "0"},
                        "stages": [{"name": "test", "run": "pytest"}],
                    },
                    "multi_gpu": {
                        "resources": {"gpu_ids": "0,1"},
                        "stages": [{"name": "test", "run": "pytest"}],
                    },
                },
            },
        },
    }
    config = normalize_config(raw)

    assert "nvidia_gpu" in config["jobs"]
    assert "nvidia_multi_gpu" in config["jobs"]
    assert config["jobs"]["nvidia_gpu"]["platform"] == "nvidia"
    assert config["jobs"]["nvidia_gpu"]["setup"] == "pip install ."
    assert config["jobs"]["nvidia_gpu"]["docker_args"] == ["--gpus", "all"]
    assert config["jobs"]["nvidia_gpu"]["resources"]["gpu_ids"] == "0"
    assert config["jobs"]["nvidia_multi_gpu"]["resources"]["gpu_ids"] == "0,1"


def test_normalize_extracts_images():
    raw = {
        "platforms": {
            "nvidia": {
                "image": {
                    "dockerfile": ".ci/images/nvidia/",
                    "build_args": {"BASE_IMAGE": "pytorch:latest"},
                },
                "jobs": {},
            },
        },
    }
    config = normalize_config(raw)
    assert config["images"]["nvidia"]["dockerfile"] == ".ci/images/nvidia/"
    assert config["images"]["nvidia"]["build_args"]["BASE_IMAGE"] == "pytorch:latest"


def test_normalize_job_overrides_platform_defaults():
    raw = {
        "platforms": {
            "nvidia": {
                "setup": "default setup",
                "jobs": {
                    "special": {
                        "setup": "custom setup",
                        "stages": [],
                    },
                },
            },
        },
    }
    config = normalize_config(raw)
    assert config["jobs"]["nvidia_special"]["setup"] == "custom setup"


def test_normalize_preserves_top_level_keys():
    raw = {
        "repo": {"url": "https://github.com/org/repo.git"},
        "github": {"status_context_prefix": "ci/test"},
        "agents": {"nvidia": {"url": "http://host:8080"}},
        "platforms": {},
    }
    config = normalize_config(raw)
    assert config["repo"]["url"] == "https://github.com/org/repo.git"
    assert config["github"]["status_context_prefix"] == "ci/test"
    assert config["agents"]["nvidia"]["url"] == "http://host:8080"


def test_normalize_passthrough_flat_config():
    """Old flat format without `platforms` key is returned as-is."""
    flat = {
        "images": {"nvidia": {}},
        "jobs": {"nvidia_gpu": {"platform": "nvidia"}},
    }
    assert normalize_config(flat) is flat


# ---------------------------------------------------------------------------
# Tests for `get_git_commit`.
# ---------------------------------------------------------------------------


def test_get_git_commit_warns_on_failure(monkeypatch, capsys):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "subprocess.run", lambda *a, **kw: MagicMock(returncode=128, stdout="")
    )
    result = get_git_commit()
    assert result == "unknown"

    captured = capsys.readouterr()
    assert "warning:" in captured.err
