"""Build the native workspace test adapter against an installed InfiniOps SDK."""

import os
import pathlib
import subprocess
import sysconfig

import pybind11

root = pathlib.Path(__file__).resolve().parents[2]
prefix = pathlib.Path(os.environ["INFINI_ROOT"])
cuda = pathlib.Path(os.environ.get("CUDA_HOME", "/opt/dtk/cuda/cuda"))
output = root / "build" / "clover-tests"
output.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        str(pathlib.Path(__file__).with_name("clover_workspace_smoke.cpp")),
        "-o",
        str(output / ("workspace_ops" + sysconfig.get_config_var("EXT_SUFFIX"))),
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_path("include"),
        "-I" + str(prefix / "include"),
        "-I" + str(root / "src"),
        "-I" + str(cuda / "include"),
        "-L" + str(prefix / "lib"),
        "-linfiniops",
        "-linfinirt",
        "-Wl,-rpath," + str(prefix / "lib"),
    ],
    check=True,
)
print(output)
