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
            f"-L{library_dir}",
            "-linfiniops",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )
    _run([str(binary)])


def _compiler(env_name, default):
    compiler = os.environ.get(env_name, default)

    if not compiler:
        pytest.skip(f"`{env_name}` is not configured.")

    return compiler


def _include_flags(require_cpp_api=False):
    install_prefix = os.environ.get("INFINIOPS_INSTALL_PREFIX")

    if install_prefix:
        return [f"-I{Path(install_prefix) / 'include'}"]

    flags = [f"-I{PROJECT_ROOT / 'include'}"]
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

    int main(void) {
      int64_t shape[1] = {3};

      float input_data[3] = {1.0f, 2.0f, 3.0f};
      float other_data[3] = {4.0f, 5.0f, 6.0f};
      float output_data[3] = {0.0f, 0.0f, 0.0f};

      InfiniOpsTensor input = {0};
      input.structure_size = sizeof(input);
      input.data = input_data;
      input.byte_size = sizeof(input_data);
      input.data_type = INFINI_OPS_DATA_TYPE_FLOAT32;
      input.device_type = INFINI_OPS_DEVICE_TYPE_CPU;
      input.rank = 1;
      input.shape = shape;

      InfiniOpsTensor other = input;
      other.data = other_data;
      other.byte_size = sizeof(other_data);

      InfiniOpsTensor output = input;
      output.data = output_data;
      output.byte_size = sizeof(output_data);

      InfiniOpsHandleAttributes handle_attributes = {0};
      handle_attributes.structure_size = sizeof(handle_attributes);
      InfiniOpsHandle handle = NULL;
      if (infiniOpsCreateHandle(&handle_attributes, &handle) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 1;
      }

      InfiniOpsConfigAttributes config_attributes = {0};
      config_attributes.structure_size = sizeof(config_attributes);
      InfiniOpsConfig config = NULL;
      if (infiniOpsCreateConfig(&config_attributes, &config) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 2;
      }

      if (infiniOpsAdd(handle, config, &input, &other, &output) !=
          INFINI_OPS_STATUS_SUCCESS) {
        return 3;
      }

      if (output_data[0] != 5.0f || output_data[1] != 7.0f ||
          output_data[2] != 9.0f) {
        return 4;
      }

      if (infiniOpsDestroyConfig(config) != INFINI_OPS_STATUS_SUCCESS) {
        return 5;
      }
      if (infiniOpsDestroyHandle(handle) != INFINI_OPS_STATUS_SUCCESS) {
        return 6;
      }
      return 0;
    }
    """
).lstrip()
