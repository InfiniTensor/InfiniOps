#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dispatcher.h"
#include "torch/tensor_.h"

namespace py = pybind11;
namespace rt = infini::rt;

namespace infini::ops {

constexpr auto kFakeDevice = static_cast<Device::Type>(100);
inline int fake_context_depth = 0;

template <>
struct C10<kFakeDevice> {
  struct StreamGuard {
    explicit StreamGuard(std::nullptr_t) { ++fake_context_depth; }
    ~StreamGuard() { --fake_context_depth; }
  };

  static std::nullptr_t GetStreamFromExternal(void*, int) { return nullptr; }
};

}  // namespace infini::ops

namespace {

struct LegacyContext {};

struct ModernContext {
  int calls = 0;

  void lazyInitDevice(c10::DeviceType type) {
    TORCH_CHECK(type == c10::DeviceType::CPU);
    ++calls;
  }
};

rt::Device::Type ParseDevice(const std::string& name) {
  return rt::Device::TypeFromString(name);
}

void Select(rt::Device::Type type, int index) {
  rt::set_runtime_device_type(type);
  TORCH_CHECK(rt::runtime::SetDevice(index) == rt::runtime::kSuccess,
              "InfiniRT SetDevice failed.");
}

void Check(rt::runtime::Error status) {
  TORCH_CHECK(status == rt::runtime::kSuccess,
              "InfiniRT operation failed: ", static_cast<int>(status));
}

template <typename Callable>
decltype(auto) Dispatch(rt::Device::Type type, Callable&& callable) {
  return infini::ops::DispatchFunc<infini::ops::ActiveDevices<void>>(
      type, std::forward<Callable>(callable), "ATen interop contract probe");
}

// Test-owned cleanup must not leave native state different from torch's TLS.
// The runtime calls themselves deliberately do not update framework caches.
template <typename Callable>
void CleanupOnDevice(rt::Device::Type type, int index, Callable&& cleanup) {
  rt::set_runtime_device_type(type);
  int entry = 0;
  if (rt::runtime::GetDevice(&entry) == rt::runtime::kSuccess &&
      rt::runtime::SetDevice(index) == rt::runtime::kSuccess) {
    cleanup();
    rt::runtime::SetDevice(entry);
  }
}

class NativeStream {
 public:
  NativeStream(const std::string& backend, int index)
      : type_{ParseDevice(backend)}, index_{index} {
    Select(type_, index_);
    Check(rt::runtime::StreamCreate(&stream_));
  }

  ~NativeStream() {
    CleanupOnDevice(type_, index_, [&] {
      rt::runtime::StreamSynchronize(stream_);
      rt::runtime::StreamDestroy(stream_);
    });
  }

  std::uintptr_t pointer() const {
    return reinterpret_cast<std::uintptr_t>(stream_);
  }

  void Synchronize() const {
    Select(type_, index_);
    Check(rt::runtime::StreamSynchronize(stream_));
  }

 private:
  rt::Device::Type type_;
  int index_;
  rt::runtime::Stream stream_{};
};

class NativeBuffer {
 public:
  NativeBuffer(const std::string& backend, int index, std::size_t count)
      : type_{ParseDevice(backend)}, index_{index}, count_{count} {
    TORCH_CHECK(count > 0, "The probe allocation must be nonempty.");
    Select(type_, index_);
    Check(rt::runtime::Malloc(&data_, count_ * sizeof(float)));
  }

  ~NativeBuffer() {
    CleanupOnDevice(type_, index_, [&] {
      rt::runtime::DeviceSynchronize();
      rt::runtime::Free(data_);
    });
  }

  std::uintptr_t pointer() const {
    return reinterpret_cast<std::uintptr_t>(data_);
  }

  at::Tensor View(const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& strides) {
    TORCH_CHECK(shape.size() == strides.size(), "Shape/stride ranks differ.");
    std::size_t span = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
      TORCH_CHECK(shape[i] >= 0 && strides[i] >= 0,
                  "The probe requires nonnegative sizes and strides.");
      if (shape[i] == 0) {
        span = 0;
        break;
      }
      const auto size = static_cast<std::size_t>(shape[i] - 1);
      const auto stride = static_cast<std::size_t>(strides[i]);
      TORCH_CHECK(size == 0 || stride <= count_ / size,
                  "The probe view exceeds its allocation.");
      const auto extra = size * stride;
      TORCH_CHECK(span <= count_ && extra <= count_ - span,
                  "The probe view exceeds its allocation.");
      span += extra;
    }
    TORCH_CHECK(span <= count_, "The probe view exceeds its allocation.");
    Select(type_, index_);
    return Dispatch(type_, [&](auto tag) {
      constexpr auto kDev = decltype(tag)::value;
      return infini::ops::WithTorchExecutionContext<kDev>(nullptr, index_, [&] {
        return infini::ops::ToAtenTensor<kDev>(data_, shape, strides,
                                               rt::DataType::kFloat32, index_);
      });
    });
  }

  void Upload(const at::Tensor& host, std::uintptr_t stream) {
    TORCH_CHECK(
        host.device().is_cpu() && host.scalar_type() == at::kFloat &&
            host.is_contiguous() && host.numel() == count_,
        "Expected a contiguous FP32 host tensor of the allocation size.");
    Select(type_, index_);
    if (type_ == rt::Device::Type::kCpu) {
      Check(rt::runtime::Memcpy(data_, host.data_ptr(), count_ * sizeof(float),
                                rt::runtime::kMemcpyHostToDevice));
      return;
    }
    Check(rt::runtime::MemcpyAsync(
        data_, host.data_ptr(), count_ * sizeof(float),
        rt::runtime::kMemcpyHostToDevice,
        reinterpret_cast<rt::runtime::Stream>(stream)));
  }

  void Download(at::Tensor host, std::uintptr_t stream) {
    TORCH_CHECK(
        host.device().is_cpu() && host.scalar_type() == at::kFloat &&
            host.is_contiguous() && host.numel() == count_,
        "Expected a contiguous FP32 host tensor of the allocation size.");
    Select(type_, index_);
    if (type_ == rt::Device::Type::kCpu) {
      Check(rt::runtime::Memcpy(host.data_ptr(), data_, count_ * sizeof(float),
                                rt::runtime::kMemcpyDeviceToHost));
      return;
    }
    Check(rt::runtime::MemcpyAsync(
        host.data_ptr(), data_, count_ * sizeof(float),
        rt::runtime::kMemcpyDeviceToHost,
        reinterpret_cast<rt::runtime::Stream>(stream)));
  }

 private:
  rt::Device::Type type_;
  int index_;
  std::size_t count_;
  void* data_{};
};

py::object WithContext(const std::string& backend, int index,
                       std::uintptr_t stream, const py::function& callback) {
  auto call = [&] { return callback(); };
  if (backend == "fake") {
    return infini::ops::WithTorchExecutionContext<infini::ops::kFakeDevice>(
        nullptr, index, call);
  }
  return Dispatch(ParseDevice(backend), [&](auto tag) {
    return infini::ops::WithTorchExecutionContext<decltype(tag)::value>(
        reinterpret_cast<void*>(stream), index, call);
  });
}

}  // namespace

PYBIND11_MODULE(_infini_ops_interop_probe, module) {
  module.def("check_context_api_compatibility", [] {
    const auto options = at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    ModernContext modern;
    infini::ops::detail::InitializeTorchDevice(modern, options, 0);
    TORCH_CHECK(modern.calls == 1);
    LegacyContext legacy;
    infini::ops::detail::InitializeTorchDevice(legacy, options, 0);
    infini::ops::detail::InitializeTorchDevice(legacy, options, 0);
  });
  py::class_<NativeStream, std::shared_ptr<NativeStream>>(module,
                                                          "NativeStream")
      .def(py::init<const std::string&, int>())
      .def_property_readonly("pointer", &NativeStream::pointer)
      .def("synchronize", &NativeStream::Synchronize);
  py::class_<NativeBuffer, std::shared_ptr<NativeBuffer>>(module,
                                                          "NativeBuffer")
      .def(py::init<const std::string&, int, std::size_t>())
      .def_property_readonly("pointer", &NativeBuffer::pointer)
      .def("view", &NativeBuffer::View, py::keep_alive<0, 1>())
      .def("upload", &NativeBuffer::Upload)
      .def("download", &NativeBuffer::Download);
  module.def("with_context", &WithContext);
  module.def("fake_depth", [] { return infini::ops::fake_context_depth; });
  module.def("set_native_device", [](const std::string& backend, int index) {
    Select(ParseDevice(backend), index);
  });
  module.def("native_device", [](const std::string& backend) {
    rt::set_runtime_device_type(ParseDevice(backend));
    int index = -1;
    Check(rt::runtime::GetDevice(&index));
    return index;
  });
}
