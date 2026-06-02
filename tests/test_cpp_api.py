import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_cpp_operator_call_instantiation_smoke(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source = tmp_path / "add_smoke.cc"
    binary = tmp_path / "add_smoke"
    source.write_text(_ADD_SMOKE_SOURCE)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            "-pthread",
            f"-I{include_dir}",
            str(source),
            f"-L{library_dir}",
            "-linfiniops",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )
    _run([str(binary)])


def _install_prefix():
    prefix = os.environ.get("INFINIOPS_INSTALL_PREFIX")

    if prefix:
        return Path(prefix)

    pytest.skip("`INFINIOPS_INSTALL_PREFIX` is not set.")


def _library_dir(prefix):
    for name in ("lib", "lib64"):
        library_dir = prefix / name
        if (library_dir / "libinfiniops.so").exists():
            return library_dir

    pytest.skip(f"`libinfiniops.so` was not found under `{prefix}`.")


def _compiler(env_name, default):
    compiler = os.environ.get(env_name, default)

    if not compiler:
        pytest.skip(f"`{env_name}` is not configured.")

    return compiler


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

    #include <atomic>
    #include <cmath>
    #include <thread>

    int main() {
      float input_data[3] = {1.0f, 2.0f, 3.0f};
      float other_data[3] = {4.0f, 5.0f, 6.0f};
      float output_data[3] = {0.0f, 0.0f, 0.0f};

      const infini::ops::Tensor::Shape shape{3};
      const infini::ops::Device device{infini::ops::Device::Type::kCpu};
      const infini::ops::DataType data_type{infini::ops::DataType::kFloat32};

      infini::ops::Tensor input(input_data, shape, data_type, device);
      infini::ops::Tensor other(other_data, shape, data_type, device);
      infini::ops::Tensor output(output_data, shape, data_type, device);
      infini::ops::Handle handle;
      infini::ops::Config config;

      infini::ops::Add::Call(handle, config, input, other, output);

      if (std::fabs(output_data[0] - 5.0f) > 1e-6f ||
          std::fabs(output_data[1] - 7.0f) > 1e-6f ||
          std::fabs(output_data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      output_data[0] = 0.0f;
      output_data[1] = 0.0f;
      output_data[2] = 0.0f;

      infini::ops::Add::Call(input, other, output);

      if (std::fabs(output_data[0] - 5.0f) > 1e-6f ||
          std::fabs(output_data[1] - 7.0f) > 1e-6f ||
          std::fabs(output_data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      std::atomic<int> failures{0};
      auto threaded_call = [&]() {
        float threaded_output[3] = {0.0f, 0.0f, 0.0f};
        infini::ops::Tensor threaded_out(threaded_output, shape, data_type,
                                         device);
        for (int i = 0; i < 100; ++i) {
          infini::ops::Add::Call(handle, config, input, other, threaded_out);
          if (std::fabs(threaded_output[0] - 5.0f) > 1e-6f ||
              std::fabs(threaded_output[1] - 7.0f) > 1e-6f ||
              std::fabs(threaded_output[2] - 9.0f) > 1e-6f) {
            failures.fetch_add(1, std::memory_order_relaxed);
          }
          threaded_output[0] = 0.0f;
          threaded_output[1] = 0.0f;
          threaded_output[2] = 0.0f;
        }
      };

      std::thread t0(threaded_call);
      std::thread t1(threaded_call);
      std::thread t2(threaded_call);
      std::thread t3(threaded_call);
      t0.join();
      t1.join();
      t2.join();
      t3.join();

      return failures.load(std::memory_order_relaxed) == 0 ? 0 : 1;
    }
    """
).lstrip()
