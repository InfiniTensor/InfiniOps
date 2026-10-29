"""Compile regressions for backend compiler selection and NVIDIA dispatch."""

import os
import pathlib
import runpy
import shlex
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name == "nt", reason="the GPU compiler wrapper uses POSIX")
def test_workspace_probe_uses_host_compiler(tmp_path, monkeypatch):
    compiler = shutil.which("g++") or shutil.which("clang++")

    if not compiler:
        pytest.skip("a host C++ compiler is required")

    gpu_compiler = tmp_path / "gpu-cxx"
    gpu_compiler.write_text(
        '#!/bin/sh\necho "GPU compiler used for host probe" >&2\nexit 1\n'
    )
    gpu_compiler.chmod(0o755)
    monkeypatch.setenv("CXX", str(gpu_compiler))
    monkeypatch.setenv("HOST_CXX", compiler)
    module = runpy.run_path(str(ROOT / "tests/test_cpp_api.py"))
    module["test_ascend_workspace_pool_retains_allocations_until_destruction"](tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="GPU compiler wrappers use POSIX")
@pytest.mark.parametrize("backend", ("metax", "moore"))
@pytest.mark.parametrize("auto_detect", (False, True))
def test_gpu_compiler_preserves_infinirt_configuration(
    tmp_path, monkeypatch, auto_detect, backend
):
    cmake = shutil.which("cmake")
    compiler = shutil.which("g++") or shutil.which("clang++")

    if not cmake or not compiler:
        pytest.skip("cmake and a C++ compiler are required")

    toolkit = tmp_path / "toolkit"
    compiler_path = "mxgpu_llvm/bin/mxcc" if backend == "metax" else "bin/mcc"
    gpu_compiler = toolkit / compiler_path
    gpu_compiler.parent.mkdir(parents=True)
    gpu_compiler.write_text(f'#!/bin/sh\nexec {shlex.quote(compiler)} "$@"\n')
    gpu_compiler.chmod(0o755)
    libraries = toolkit / "lib"
    libraries.mkdir()

    library_names = (
        ("mcruntime", "mcdnn", "mcblas")
        if backend == "metax"
        else ("musa", "musart", "mublas", "iomp5")
    )
    for name in library_names:
        (libraries / f"lib{name}.so").touch()

    for name in (
        "INFINI_RT_ROOT",
        "MACA_PATH",
        "HPCC_PATH",
        "NEUWARE_HOME",
        "MUSA_ROOT",
        "MUSA_HOME",
        "MUSA_PATH",
        "ASCEND_HOME_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("MACA_PATH" if backend == "metax" else "MUSA_HOME", str(toolkit))
    wrapper_name = "mxcc_wrapper.sh" if backend == "metax" else "mcc_wrapper.sh"
    wrapper = tmp_path / "scripts" / wrapper_name
    wrapper.parent.mkdir()
    shutil.copyfile(ROOT / "scripts" / wrapper_name, wrapper)
    wrapper.chmod(0o755)
    prefix = tmp_path / "infinirt"
    (prefix / "include/infini").mkdir(parents=True)
    (prefix / "include/infini/rt.h").touch()
    (prefix / "lib").mkdir()
    (prefix / "lib/libinfinirt.so").touch()

    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    initialization = source.split("if(AUTO_DETECT_BACKENDS)", 1)[0]
    backend_option = f"WITH_{backend.upper()}"
    backend_start = source.index(f"if({backend_option})\n    add_compile_definitions")
    next_backend = "WITH_MARS" if backend == "metax" else "WITH_CAMBRICON"
    backend_config = source[
        backend_start : source.index(f"if({next_backend})", backend_start)
    ]
    runtime_start = source.index('set(INFINI_RT_ROOT "$ENV{INFINI_RT_ROOT}"')
    runtime = source[
        runtime_start : source.index("add_subdirectory(src)", runtime_start)
    ]
    (tmp_path / "CMakeLists.txt").write_text(
        initialization + backend_config + runtime + "\nadd_subdirectory(dependency)\n"
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
        configure.append(f"-D{backend_option}=ON")

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

    public_header = runpy.run_path(
        str(ROOT / "scripts/generate_public_operator_header.py")
    )
    public_header["generate_public_operator_header"](
        ROOT / "src/operator.h", tmp_path / "operator.h"
    )
    (tmp_path / "generated_dispatch.h").write_text('#include "operator.h"\n')
    (tmp_path / "infini").mkdir()
    (tmp_path / "infini/operator_call_instantiations.h").write_text("\n")
    generator = runpy.run_path(str(ROOT / "scripts/generate_wrappers.py"))
    dispatch = tmp_path / "generated_dispatch_regression.cu"
    dispatch.write_text(
        generator["_generate_generated_dispatch_source"](
            ["native/cuda/nvidia/ops/cat/kernel.h"], []
        )
    )

    for source in (
        ROOT / "tests/build_config/nvidia_dispatch.cu",
        ROOT / "src/native/cuda/nvidia/ops/quick_gelu/kernel.cu",
        dispatch,
    ):
        result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={architecture}",
                "-DWITH_CPU=1",
                "-DWITH_NVIDIA=1",
                *([f"-I{tmp_path}"] if source == dispatch else []),
                f"-I{ROOT}",
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
