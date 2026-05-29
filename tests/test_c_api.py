import os
import subprocess
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_c_api_header_compiles_with_c(tmp_path):
    source = tmp_path / "header_smoke.c"
    source.write_text(
        "#include <infini/ops.h>\n"
        "int main(void) { return INFINI_OPS_STATUS_SUCCESS; }\n"
    )
    output = tmp_path / "header_smoke.o"

    _run(
        [
            _compiler("CC", "cc"),
            "-std=c11",
            "-Werror",
            *_include_flags(),
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_c_api_header_compiles_with_cpp(tmp_path):
    source = tmp_path / "header_smoke.cc"
    source.write_text(
        "#include <infini/ops.h>\nint main() { return INFINI_OPS_STATUS_SUCCESS; }\n"
    )
    output = tmp_path / "header_smoke.o"

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            *_include_flags(require_cpp_api=True),
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_c_api_add_smoke(tmp_path):
    library_dir = _installed_library_dir()
    source = tmp_path / "add_smoke.c"
    binary = tmp_path / "add_smoke"
    source.write_text(_ADD_SMOKE_SOURCE)

    _run(
        [
            _compiler("CC", "cc"),
            "-std=c11",
            "-Werror",
            *_include_flags(),
            str(source),
            *_library_link_flags(library_dir),
            "-o",
            str(binary),
        ]
    )
    _run([str(binary)])


def _compiler(env_name, default):
    compiler = os.environ.get(env_name, default)

    if not compiler:
        pytest.skip(f"`{env_name}` is not configured.")

    if env_name == "CXX" and "cu-bridge" in compiler:
        compiler = default

    return compiler


def _include_flags(require_cpp_api=False):
    install_prefix = os.environ.get("INFINIOPS_INSTALL_PREFIX")

    if install_prefix:
        return [f"-I{Path(install_prefix) / 'include'}"]

    flags = [f"-I{PROJECT_ROOT / 'include'}", f"-I{PROJECT_ROOT / 'src'}"]
    generated_include_dir = PROJECT_ROOT / "generated" / "include"

    if generated_include_dir.exists():
        flags.append(f"-I{generated_include_dir}")
    elif require_cpp_api:
        pytest.skip("generated C++ API headers are not available.")

    return flags


def _installed_library_dir():
    install_prefix = os.environ.get("INFINIOPS_INSTALL_PREFIX")

    if install_prefix:
        for name in ("lib", "lib64"):
            library_dir = Path(install_prefix) / name
            if (library_dir / "libinfiniops.so").exists():
                return library_dir
        pytest.skip(f"`libinfiniops.so` was not found under `{install_prefix}`.")

    library_dir = os.environ.get("INFINIOPS_LIBRARY_DIR")

    if library_dir:
        return Path(library_dir)

    try:
        import infini.ops
    except ImportError as error:
        pytest.skip(
            "`infini.ops` is not installed and neither "
            "`INFINIOPS_INSTALL_PREFIX` nor `INFINIOPS_LIBRARY_DIR` is set: "
            f"{error}"
        )

    return Path(infini.ops.__file__).resolve().parent


def _library_link_flags(library_dir):
    flags = [f"-L{library_dir}", "-linfiniops", f"-Wl,-rpath,{library_dir}"]

    for runtime_dir in _python_runtime_library_dirs():
        flags.extend(
            [
                f"-L{runtime_dir}",
                f"-Wl,-rpath,{runtime_dir}",
                f"-Wl,-rpath-link,{runtime_dir}",
            ]
        )

        if (runtime_dir / "libiomp5.so").exists():
            flags.append("-liomp5")
        elif (runtime_dir / "libomp.so").exists():
            flags.append("-lomp")

    return flags


def _python_runtime_library_dirs():
    runtime_dirs = []

    try:
        import torch
    except ImportError:
        return runtime_dirs

    site_packages = Path(torch.__file__).resolve().parents[1]
    for name in ("torch/lib", "torch.libs"):
        runtime_dir = site_packages / name
        if runtime_dir.exists():
            runtime_dirs.append(runtime_dir)

    maca_path = os.environ.get("MACA_PATH")
    if maca_path:
        runtime_dir = Path(maca_path) / "mxgpu_llvm" / "lib"
        if runtime_dir.exists():
            runtime_dirs.append(runtime_dir)

    return runtime_dirs


def _run(command):
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as error:
        pytest.skip(f"`{command[0]}` is not available: {error}")
    except subprocess.CalledProcessError as error:
        output = "\n".join((error.stdout, error.stderr)).strip()
        raise AssertionError(output) from error


_ADD_SMOKE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <stddef.h>
    #include <stdint.h>

    static int create_tensor(float* data, size_t byte_size, const int64_t* shape,
                             InfiniOpsTensor* tensor) {
      if (infiniOpsCreateTensor(tensor) != INFINI_OPS_STATUS_SUCCESS) {
        return 1;
      }
      if (infiniOpsSetTensorData(*tensor, data) != INFINI_OPS_STATUS_SUCCESS) {
        return 2;
      }
      if (infiniOpsSetTensorByteSize(*tensor, byte_size) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 3;
      }
      if (infiniOpsSetTensorDataType(*tensor, INFINI_OPS_DATA_TYPE_FLOAT32) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 4;
      }
      if (infiniOpsSetTensorDeviceType(*tensor, INFINI_OPS_DEVICE_TYPE_CPU) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 5;
      }
      if (infiniOpsSetTensorShape(*tensor, 1, shape) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 6;
      }
      return 0;
    }

    int main(void) {
      int64_t shape[1] = {3};

      float input_data[3] = {1.0f, 2.0f, 3.0f};
      float other_data[3] = {4.0f, 5.0f, 6.0f};
      float output_data[3] = {0.0f, 0.0f, 0.0f};

      InfiniOpsTensor input = NULL;
      InfiniOpsTensor other = NULL;
      InfiniOpsTensor output = NULL;
      if (create_tensor(input_data, sizeof(input_data), shape, &input) != 0) {
        return 1;
      }
      if (create_tensor(other_data, sizeof(other_data), shape, &other) != 0) {
        return 2;
      }
      if (create_tensor(output_data, sizeof(output_data), shape, &output) != 0) {
        return 3;
      }

      int32_t rank = 0;
      const int64_t* stored_shape = NULL;
      if (infiniOpsGetTensorShape(input, &rank, &stored_shape) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 4;
      }
      if (rank != 1 || stored_shape == NULL || stored_shape[0] != 3) {
        return 5;
      }

      InfiniOpsHandle handle = NULL;
      if (infiniOpsCreateHandle(&handle) != INFINI_OPS_STATUS_SUCCESS) {
        return 6;
      }

      InfiniOpsConfig config = NULL;
      if (infiniOpsCreateConfig(&config) != INFINI_OPS_STATUS_SUCCESS) {
        return 7;
      }
      if (infiniOpsSetConfigImplementationIndex(config, 0) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 8;
      }

      if (infiniOpsAdd(handle, config, input, other, output) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 9;
      }

      if (output_data[0] != 5.0f || output_data[1] != 7.0f ||
          output_data[2] != 9.0f) {
        return 10;
      }

      if (infiniOpsDestroyConfig(config) != INFINI_OPS_STATUS_SUCCESS) {
        return 11;
      }
      if (infiniOpsDestroyHandle(handle) != INFINI_OPS_STATUS_SUCCESS) {
        return 12;
      }
      if (infiniOpsDestroyTensor(output) != INFINI_OPS_STATUS_SUCCESS) {
        return 13;
      }
      if (infiniOpsDestroyTensor(other) != INFINI_OPS_STATUS_SUCCESS) {
        return 14;
      }
      if (infiniOpsDestroyTensor(input) != INFINI_OPS_STATUS_SUCCESS) {
        return 15;
      }
      return 0;
    }
    """
).lstrip()
