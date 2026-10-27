"""Compile regressions for backend compiler selection and NVIDIA dispatch."""

import os
import pathlib
import shlex
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name == "nt", reason="MetaX uses a POSIX compiler wrapper")
@pytest.mark.parametrize("auto_detect", (False, True))
def test_metax_compiler_preserves_infinirt_configuration(
    tmp_path, monkeypatch, auto_detect
):
    cmake = shutil.which("cmake")
    compiler = shutil.which("g++") or shutil.which("clang++")

    if not cmake or not compiler:
        pytest.skip("cmake and a C++ compiler are required")

    toolkit = tmp_path / "maca"
    mxcc = toolkit / "mxgpu_llvm/bin/mxcc"
    mxcc.parent.mkdir(parents=True)
    mxcc.write_text(f'#!/bin/sh\nexec {shlex.quote(compiler)} "$@"\n')
    mxcc.chmod(0o755)
    libraries = toolkit / "lib"
    libraries.mkdir()

    for name in ("mcruntime", "mcdnn", "mcblas"):
        (libraries / f"lib{name}.so").touch()

    monkeypatch.setenv("MACA_PATH", str(toolkit))

    for name in (
        "INFINI_RT_ROOT",
        "HPCC_PATH",
        "NEUWARE_HOME",
        "MUSA_ROOT",
        "MUSA_HOME",
        "MUSA_PATH",
        "ASCEND_HOME_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    wrapper = tmp_path / "scripts/mxcc_wrapper.sh"
    wrapper.parent.mkdir()
    shutil.copyfile(ROOT / "scripts/mxcc_wrapper.sh", wrapper)
    wrapper.chmod(0o755)
    prefix = tmp_path / "infinirt"
    (prefix / "include/infini").mkdir(parents=True)
    (prefix / "include/infini/rt.h").touch()
    (prefix / "lib").mkdir()
    (prefix / "lib/libinfinirt.so").touch()

    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    initialization = source.split("if(AUTO_DETECT_BACKENDS)", 1)[0]
    metax_start = source.index("if(WITH_METAX)\n    add_compile_definitions")
    metax = source[metax_start : source.index("if(WITH_MARS)", metax_start)]
    runtime_start = source.index('set(INFINI_RT_ROOT "$ENV{INFINI_RT_ROOT}"')
    runtime = source[
        runtime_start : source.index("add_subdirectory(src)", runtime_start)
    ]
    (tmp_path / "CMakeLists.txt").write_text(
        initialization + metax + runtime + "\nadd_subdirectory(dependency)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/selected-compiler.txt" '
        '"${CMAKE_CXX_COMPILER}")\nadd_library(probe OBJECT probe.cc)\n'
    )
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(Dependency LANGUAGES CXX)\n"
    )
    (tmp_path / "probe.cc").write_text("int probe() { return 0; }\n")
    build = tmp_path / "build"
    configure = [
        cmake,
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        f"-DINFINI_RT_ROOT={prefix}",
        f"-DAUTO_DETECT_DEVICES={'ON' if auto_detect else 'OFF'}",
    ]

    if not auto_detect:
        configure.append("-DWITH_METAX=ON")

    for _ in range(2):
        result = subprocess.run(configure, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "require your cache to be deleted" not in result.stdout + result.stderr
        assert "Using installed InfiniRT:" in result.stdout
        cache = (build / "CMakeCache.txt").read_text()
        assert (build / "selected-compiler.txt").read_text() == str(wrapper)
        assert f"INFINI_RT_ROOT:PATH={prefix}" in cache

    subprocess.run([cmake, "--build", str(build)], check=True, timeout=30)


@pytest.mark.parametrize("architecture", ("sm_52", "sm_80"))
def test_nvidia_dispatch_and_quick_gelu_compile(tmp_path, architecture):
    nvcc = shutil.which("nvcc")
    runtime = os.environ.get("INFINI_RT_ROOT")

    if not nvcc or not runtime:
        pytest.skip("nvcc and an installed INFINI_RT_ROOT are required")

    supported = subprocess.run(
        [nvcc, "--list-gpu-code"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.split()

    if architecture not in supported:
        pytest.skip(f"nvcc does not support {architecture}")

    for source in (
        ROOT / "tests/build_config/nvidia_dispatch.cu",
        ROOT / "src/native/cuda/nvidia/ops/quick_gelu/kernel.cu",
    ):
        result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={architecture}",
                "-DWITH_CPU=1",
                "-DWITH_NVIDIA=1",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
                f"-I{pathlib.Path(runtime) / 'include'}",
                "-c",
                str(source),
                "-o",
                str(tmp_path / (source.stem + ".o")),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
