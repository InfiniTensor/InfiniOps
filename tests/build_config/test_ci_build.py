"""Hardware-independent regressions for CI build setup and SOC discovery."""

import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOC_DIR = ROOT / "src/native/ascend/custom/cmake"
CMAKE = shutil.which("cmake")


@pytest.mark.parametrize(
    "platform", ("nvidia", "iluvatar", "metax", "moore", "cambricon", "hygon", "ascend")
)
def test_ci_installs_declared_build_requirements(platform):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    config = yaml.safe_load((ROOT / ".github/ci_config.yml").read_text())
    commands = config["platforms"][platform]["setup"].split("&&")
    install = shlex.split(commands[0])
    assert install[:4] == ["python", "-m", "pip", "install"]
    requirements = {Requirement(arg).name: Requirement(arg) for arg in install[4:]}

    for declared in project["build-system"]["requires"]:
        required = Requirement(declared)
        assert required.name in requirements
        assert requirements[required.name].specifier == required.specifier

    for command in commands:
        tokens = shlex.split(command)

        if "install" in tokens and "pip" in tokens:
            assert tokens[:4] == ["python", "-m", "pip", "install"]


@pytest.mark.skipif(os.name == "nt", reason="CI stages use a POSIX shell")
@pytest.mark.parametrize("allocated_devices", ("0", "2,5", "", None))
def test_moore_smoke_uses_allocated_devices(tmp_path, monkeypatch, allocated_devices):
    config = yaml.safe_load((ROOT / ".github/ci_config.yml").read_text())
    stage = config["platforms"]["moore"]["jobs"]["gpu"]["stages"][0]["run"]
    probe = tmp_path / "pytest"
    probe.write_text('#!/bin/sh\nprintf "%s" "$MUSA_VISIBLE_DEVICES"\n')
    probe.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "7")

    if allocated_devices is None:
        monkeypatch.delenv("MTHREADS_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("MTHREADS_VISIBLE_DEVICES", allocated_devices)

    result = subprocess.run(
        ["/bin/sh", "-c", stage], capture_output=True, text=True, timeout=10
    )

    if allocated_devices:
        assert result.returncode == 0, result.stderr
        assert result.stdout == allocated_devices
    else:
        assert result.returncode != 0
        assert not result.stdout
        assert "CI must select a Moore GPU" in result.stderr


def test_wheel_metadata_accepts_sdist_configuration(tmp_path, monkeypatch):
    from scikit_build_core.build import prepare_metadata_for_build_wheel

    monkeypatch.chdir(ROOT)
    dist_info = prepare_metadata_for_build_wheel(str(tmp_path))
    metadata = (tmp_path / dist_info / "METADATA").read_text()
    assert "Name: InfiniOps" in metadata
    assert "Version: 0.1.0" in metadata


@pytest.fixture
def soc_probe(tmp_path, monkeypatch, fake_smi):
    if not CMAKE:
        pytest.skip("cmake is unavailable")

    fake_modules = tmp_path / "python"
    fake_modules.mkdir()
    (fake_modules / "torch_npu.py").write_text("")
    (fake_modules / "torch.py").write_text(
        "import os\n"
        "class npu:\n"
        "    @staticmethod\n"
        "    def get_device_name(index):\n"
        "        name = os.environ.get('TEST_RUNTIME_SOC')\n"
        "        if not name:\n"
        "            raise RuntimeError('No NPU runtime is available.')\n"
        "        return name\n"
    )
    smi_output = tmp_path / "smi.txt"
    monkeypatch.setenv("TEST_SMI_OUTPUT", str(smi_output))
    monkeypatch.setenv("PATH", str(fake_smi) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", str(fake_modules))
    output = tmp_path / "soc.txt"
    script = tmp_path / "probe.cmake"
    script.write_text(
        f'include("{SOC_DIR.as_posix()}/detect_soc.cmake")\n'
        "infiniops_detect_soc(DETECTED_SOC)\n"
        f'file(WRITE "{output.as_posix()}" "${{DETECTED_SOC}}")\n'
    )

    def run(runtime_soc, smi_text, *, torch_python=True):
        monkeypatch.setenv("TEST_RUNTIME_SOC", runtime_soc)
        smi_output.write_text(smi_text)
        interpreter = "_TORCH_PYTHON" if torch_python else "Python_EXECUTABLE"
        command = [CMAKE, f"-D{interpreter}={sys.executable}"]

        if torch_python:
            command.append("-DPython_EXECUTABLE=/missing/python")

        result = subprocess.run(
            [*command, "-P", str(script)], capture_output=True, text=True, timeout=45
        )
        name = output.read_text() if output.exists() else None

        return result, name

    return run


@pytest.fixture(scope="session")
def fake_smi(tmp_path_factory):
    directory = tmp_path_factory.mktemp("npu-smi")
    smi = directory / ("npu-smi.exe" if os.name == "nt" else "npu-smi")

    if os.name == "nt":
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            pytest.skip("A C++ compiler is required for the Windows npu-smi fixture")
        source = directory / "smi.cc"
        source.write_text(
            "#include <cstdlib>\n#include <fstream>\n#include <iostream>\n"
            'int main() { std::ifstream input(std::getenv("TEST_SMI_OUTPUT")); '
            "std::cout << input.rdbuf(); return 0; }\n"
        )
        subprocess.run([compiler, str(source), "-o", str(smi)], check=True, timeout=30)
    else:
        smi.write_text('#!/bin/sh\ncat "$TEST_SMI_OUTPUT"\n')
        smi.chmod(0o755)

    return directory


@pytest.mark.parametrize("torch_python", (True, False))
def test_runtime_soc_takes_priority_over_smi(soc_probe, torch_python):
    result, name = soc_probe(
        "Ascend910_9362", "| 0 | 910B4 | 3103 |", torch_python=torch_python
    )
    assert result.returncode == 0, result.stderr
    assert name == "Ascend910_9362"


@pytest.mark.parametrize(
    "model", ("910B4", "910B4-1", "Ascend910_9382", "310P3", "310P3vir02")
)
def test_smi_accepts_qualified_models(soc_probe, model):
    result, name = soc_probe("", f"| 0 | {model} | OK |\n| 3103 | 3109 |")
    assert result.returncode == 0, result.stderr
    assert name == (model if model.startswith("Ascend") else f"Ascend{model}")


@pytest.mark.parametrize(
    "smi_text", ("", "| Ascend910 | 3103 | 3109 |", "| 3103 |", "| 910B40garbage |")
)
def test_ambiguous_soc_requires_explicit_configuration(soc_probe, smi_text):
    result, name = soc_probe("Ascend910", smi_text)
    assert result.returncode != 0
    assert name is None
    assert "Set SOC_VERSION explicitly" in result.stderr


def test_explicit_soc_bypasses_detection(tmp_path):
    if not CMAKE:
        pytest.skip("cmake is unavailable")

    toolkit = tmp_path / "toolkit"
    ascendc = toolkit / "tools/tikcpp/ascendc_kernel_cmake"
    ascendc.mkdir(parents=True)
    (ascendc / "ascendc.cmake").write_text("")
    script = tmp_path / "explicit.cmake"
    script.write_text(
        'set(SOC_VERSION "Ascend910_9382")\n'
        f'set(ASCEND_HOME_PATH "{toolkit.as_posix()}")\n'
        f'include("{SOC_DIR.as_posix()}/config_ascend.cmake")\n'
        'if(NOT SOC_VERSION STREQUAL "Ascend910_9382")\n'
        '    message(FATAL_ERROR "Explicit SOC version was changed.")\n'
        "endif()\n"
    )
    result = subprocess.run(
        [CMAKE, "-P", str(script)], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_iluvatar_dispatch_can_include_public_and_generated_headers(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")

    if not CMAKE or not compiler:
        pytest.skip("cmake and a C++ compiler are required")

    source = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
    start = source.index("        set(_iluvatar_dispatch_include_flags")
    end = source.index("        foreach(", start)
    include_flags = source[start:end]

    for directory in ("include", "generated/include"):
        headers = tmp_path / directory / "infini"
        headers.mkdir(parents=True)
        header = (
            "public.h" if directory == "include" else "operator_call_instantiations.h"
        )
        (headers / header).write_text("#pragma once\n")

    dispatch = tmp_path / "dispatch.cc"
    dispatch.write_text(
        '#include "infini/public.h"\n#include "infini/operator_call_instantiations.h"\n'
    )
    script = tmp_path / "compile.cmake"
    script.write_text(
        f'set(PROJECT_SOURCE_DIR "{tmp_path.as_posix()}")\n'
        f'set(CMAKE_CURRENT_SOURCE_DIR "{tmp_path.as_posix()}/src")\n'
        f"{include_flags}\n"
        f'execute_process(COMMAND "{pathlib.Path(compiler).as_posix()}" '
        f'${{_iluvatar_dispatch_include_flags}} -fsyntax-only "{dispatch.as_posix()}" '
        "RESULT_VARIABLE result)\n"
        'if(NOT result STREQUAL "0")\n'
        '    message(FATAL_ERROR "Dispatch header lookup failed.")\n'
        "endif()\n"
    )
    result = subprocess.run(
        [CMAKE, "-P", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
