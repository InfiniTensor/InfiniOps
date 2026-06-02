import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_cpp_operator_call_instantiation_smoke(tmp_path):
    _compile_and_run(tmp_path, "add_smoke", _ADD_SMOKE_SOURCE)


def test_cpp_operator_call_thread_local_cache_regression(tmp_path):
    _compile_and_run(tmp_path, "thread_local_cache_probe", _THREAD_LOCAL_CACHE_SOURCE)


def _compile_and_run(tmp_path, stem, source_text):
    install_prefix = _install_prefix()
    include_dir = _include_dir(install_prefix)
    library_dir = _library_dir(install_prefix)
    source = tmp_path / f"{stem}.cc"
    binary = tmp_path / stem
    source.write_text(source_text)

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

    try:
        import infini
    except ImportError:
        pytest.skip("INFINIOPS_INSTALL_PREFIX is not set and infini is not installed.")

    return Path(infini.__file__).resolve().parent


def _candidate_prefixes(prefix):
    yield prefix

    try:
        import infini
    except ImportError:
        return

    package_prefix = Path(infini.__file__).resolve().parent
    if package_prefix != prefix:
        yield package_prefix


def _include_dir(prefix):
    for candidate in _candidate_prefixes(prefix):
        include_dir = candidate / "include"
        if (include_dir / "infini" / "ops.h").exists():
            return include_dir

    pytest.skip(f"infini/ops.h was not found under {prefix}.")


def _library_dir(prefix):
    for candidate in _candidate_prefixes(prefix):
        for name in ("lib", "lib64", "."):
            library_dir = candidate / name
            if (library_dir / "libinfiniops.so").exists():
                return library_dir

    pytest.skip(f"libinfiniops.so was not found under {prefix}.")


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


_THREAD_LOCAL_CACHE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <atomic>
    #include <thread>

    namespace infini::ops {

    class ThreadLocalCacheProbe : public Operator<ThreadLocalCacheProbe> {
     public:
      ThreadLocalCacheProbe(const Tensor input, Tensor out) {}

      virtual void operator()(const Tensor input, Tensor out) const = 0;
    };

    template <>
    struct ActiveDevicesImpl<ThreadLocalCacheProbe> {
      using type = List<Device::Type::kCpu>;
    };

    template <>
    class Operator<ThreadLocalCacheProbe, Device::Type::kCpu>
        : public ThreadLocalCacheProbe {
     public:
      Operator(const Tensor input, Tensor out)
          : ThreadLocalCacheProbe{input, out},
            owner_thread_id_{std::this_thread::get_id()} {
        constructions.fetch_add(1, std::memory_order_relaxed);
      }

      void operator()(const Tensor input, Tensor out) const override {
        if (owner_thread_id_ != std::this_thread::get_id()) {
          cross_thread_calls.fetch_add(1, std::memory_order_relaxed);
        }
      }

      static std::atomic<int> constructions;

      static std::atomic<int> cross_thread_calls;

     private:
      std::thread::id owner_thread_id_;
    };

    std::atomic<int> Operator<ThreadLocalCacheProbe,
                              Device::Type::kCpu>::constructions{0};

    std::atomic<int> Operator<ThreadLocalCacheProbe,
                              Device::Type::kCpu>::cross_thread_calls{0};

    }  // namespace infini::ops

    int main() {
      float input_data[1] = {1.0f};
      float output_data[1] = {0.0f};

      const infini::ops::Tensor::Shape shape{1};
      const infini::ops::Device device{infini::ops::Device::Type::kCpu};
      const infini::ops::DataType data_type{infini::ops::DataType::kFloat32};
      infini::ops::Tensor input(input_data, shape, data_type, device);
      infini::ops::Tensor output(output_data, shape, data_type, device);
      infini::ops::Handle handle;
      infini::ops::Config config;

      using Probe = infini::ops::ThreadLocalCacheProbe;
      using ProbeImpl = infini::ops::Operator<
          Probe, infini::ops::Device::Type::kCpu>;
      Probe::clear_cache();
      ProbeImpl::constructions.store(0, std::memory_order_relaxed);
      ProbeImpl::cross_thread_calls.store(0, std::memory_order_relaxed);

      Probe::Call(handle, config, input, output);

      std::thread cache_probe_thread([&]() {
        Probe::Call(handle, config, input, output);
      });
      cache_probe_thread.join();

      if (ProbeImpl::cross_thread_calls.load(std::memory_order_relaxed) != 0) {
        return 1;
      }

      return ProbeImpl::constructions.load(std::memory_order_relaxed) == 2 ? 0
                                                                          : 1;
    }
    """
).lstrip()
