import os
import stat
import subprocess
from pathlib import Path

import pytest


DETECT_SOC = (
    Path(__file__).parents[1]
    / "src"
    / "native"
    / "ascend"
    / "custom"
    / "cmake"
    / "detect_soc.cmake"
)


def _write_executable(path, contents):
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _detect_soc(tmp_path, python_output=None, npu_smi_output=""):
    if os.name == "nt":
        pytest.skip("the detector invokes POSIX Ascend command-line tools")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    npu_smi = bin_dir / "npu-smi"

    if python_output is None:
        _write_executable(python, "#!/bin/sh\nexit 1\n")
    else:
        _write_executable(python, f"#!/bin/sh\nprintf '%s\\n' '{python_output}'\n")

    _write_executable(
        npu_smi,
        "#!/bin/sh\ncat <<'EOF'\n" + npu_smi_output + "\nEOF\n",
    )

    driver = tmp_path / "detect.cmake"
    driver.write_text(
        f'set(Python_EXECUTABLE "{python.as_posix()}")\n'
        f'include("{DETECT_SOC.as_posix()}")\n'
        "infiniops_detect_soc(DETECTED_SOC)\n"
        'message(STATUS "DETECTED_SOC=${DETECTED_SOC}")\n'
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        ["cmake", "-P", str(driver)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    return result.stdout


def test_detect_soc_prefers_exact_torch_npu_name(tmp_path):
    output = _detect_soc(
        tmp_path,
        python_output="Ascend910_9362",
        npu_smi_output="| 7 Ascend910 | 0 0 / 0 3109 / 65536 |",
    )

    assert "DETECTED_SOC=Ascend910_9362" in output


def test_detect_soc_ignores_hbm_numbers_in_npu_smi(tmp_path):
    output = _detect_soc(
        tmp_path,
        npu_smi_output="| 7 Ascend910 | 0 0 / 0 3109 / 65536 |",
    )

    assert "DETECTED_SOC=Ascend910B4" in output
    assert "Ascend3109" not in output


def test_detect_soc_accepts_model_qualified_npu_smi_name(tmp_path):
    output = _detect_soc(
        tmp_path,
        npu_smi_output="| 0 910B4 | 0 0 / 0 2789 / 32768 |",
    )

    assert "DETECTED_SOC=Ascend910B4" in output
