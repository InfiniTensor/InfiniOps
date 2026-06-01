import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_bash(script, *, env=None, check=True):
    base_env = {
        "HOME": os.environ.get("HOME", str(REPO_ROOT)),
        "PATH": os.environ.get("PATH", ""),
    }
    if env:
        base_env.update(env)

    return subprocess.run(
        ["bash", "-lc", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
        env=base_env,
    )


def _isolated_probe_env():
    return {
        "DTK_ROOT": str(REPO_ROOT / ".tmp-no-dtk"),
        "INFINIOPS_NVIDIA_DEVICE_GLOB": str(REPO_ROOT / ".tmp-no-dev" / "nvidia*"),
        "INFINIOPS_ILUVATAR_DEVICE_GLOB": str(REPO_ROOT / ".tmp-no-dev" / "iluvatar*"),
        "INFINIOPS_ASCEND_DEVICE_GLOB": str(REPO_ROOT / ".tmp-no-dev" / "davinci0"),
        "INFINIOPS_METAX_PCI_VENDOR_GLOB": str(REPO_ROOT / ".tmp-no-dev" / "vendor"),
    }


def test_dev_build_accepts_nvidia_platform():
    result = _run_bash("bash scripts/dev_build.sh nvidia --help")

    assert "Usage:" in result.stdout


def test_dev_test_accepts_ascend_platform():
    result = _run_bash("bash scripts/dev_test.sh ascend --help")

    assert "Usage:" in result.stdout


def test_detect_platforms_reports_fake_nvidia_probe(tmp_path):
    fake_dev = tmp_path / "dev"
    fake_dev.mkdir()
    (fake_dev / "nvidia0").touch()

    result = _run_bash(
        "source scripts/dev_platforms.sh; infiniops_detect_platforms",
        env={
            **_isolated_probe_env(),
            "INFINIOPS_NVIDIA_DEVICE_GLOB": str(fake_dev / "nvidia*"),
        },
    )

    assert result.stdout.strip() == "nvidia"


def test_detect_platforms_reports_env_backends_in_cmake_order():
    result = _run_bash(
        "source scripts/dev_platforms.sh; infiniops_detect_platforms",
        env={
            **_isolated_probe_env(),
            "MACA_PATH": "/tmp/maca",
            "NEUWARE_HOME": "/tmp/neuware",
        },
    )

    assert result.stdout.strip().splitlines() == ["metax", "cambricon"]


def test_detect_platforms_emits_nothing_when_no_backend_matches():
    result = _run_bash(
        "source scripts/dev_platforms.sh; infiniops_detect_platforms | sed -n l",
        env=_isolated_probe_env(),
    )

    assert result.stdout == ""
