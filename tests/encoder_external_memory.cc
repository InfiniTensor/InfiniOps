#include <infini/ops.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <vector>

namespace {
namespace rt = infini::rt;
namespace ops = infini::ops;

struct Storage {
  rt::runtime::Stream stream{};
  std::vector<void*> allocations;

  Storage() {
    assert(rt::runtime::StreamCreate(&stream) == rt::runtime::kSuccess);
  }
  ~Storage() {
    (void)rt::runtime::StreamSynchronize(stream);
    for (auto ptr : allocations) (void)rt::runtime::Free(ptr);
    (void)rt::runtime::StreamDestroy(stream);
  }
  void* Allocate(size_t bytes) {
    void* ptr = nullptr;
    assert(rt::runtime::Malloc(&ptr, bytes) == rt::runtime::kSuccess);
    allocations.push_back(ptr);
    return ptr;
  }
};

std::vector<float> Run(const std::string& device_name, int device_index) {
  const rt::Device device{rt::Device::TypeFromString(device_name),
                          device_index};
  rt::set_runtime_device_type(device.type());
  assert(rt::runtime::SetDevice(device_index) == rt::runtime::kSuccess);
  Storage storage;
  const auto tensor = [&](ops::Tensor::Shape shape, rt::DataType dtype) {
    size_t count = 1;
    for (auto dim : shape) count *= dim;
    return ops::Tensor{storage.Allocate(count * rt::kDataTypeToSize.at(dtype)),
                       shape, dtype, device};
  };
  auto table = tensor({8, 4}, rt::DataType::kFloat32);
  auto ids = tensor({2, 3}, rt::DataType::kInt64);
  auto gathered = tensor({2, 3, 4}, rt::DataType::kFloat32);
  auto normalized = tensor({2, 3, 4}, rt::DataType::kFloat32);
  auto mean = tensor({2, 3, 1}, rt::DataType::kFloat32);
  auto rstd = tensor({2, 3, 1}, rt::DataType::kFloat32);
  auto converted = tensor({2, 3}, rt::DataType::kFloat32);
  std::vector<float> values(32);
  for (size_t i = 0; i < values.size(); ++i) values[i] = float(i % 7) - 3;
  const int64_t indices[6] = {7, 0, 3, 1, 6, 2};
  assert(rt::runtime::Memcpy(table.data(), values.data(), values.size() * 4,
                             rt::runtime::kMemcpyHostToDevice) ==
         rt::runtime::kSuccess);
  assert(rt::runtime::Memcpy(ids.data(), indices, sizeof(indices),
                             rt::runtime::kMemcpyHostToDevice) ==
         rt::runtime::kSuccess);
  ops::Handle handle;
  handle.set_stream(storage.stream);
  ops::Config config;
  config.set_implementation_index(8);
  ops::OnnxGather::Call(handle, config, table, ids, int64_t{0}, gathered);
  ops::Cast::Call(handle, config, ids, converted);
  ops::NativeLayerNorm::Call(handle, config, gathered, std::vector<int64_t>{4},
                             std::optional<ops::Tensor>{},
                             std::optional<ops::Tensor>{}, 1e-5, normalized,
                             mean, rstd);
  std::vector<float> result(30);
  if (device.type() == rt::Device::Type::kCpu) {
    assert(rt::runtime::Memcpy(result.data(), normalized.data(), 24 * 4,
                               rt::runtime::kMemcpyDeviceToHost) ==
           rt::runtime::kSuccess);
    assert(rt::runtime::Memcpy(result.data() + 24, converted.data(), 6 * 4,
                               rt::runtime::kMemcpyDeviceToHost) ==
           rt::runtime::kSuccess);
  } else {
    assert(rt::runtime::MemcpyAsync(result.data(), normalized.data(), 24 * 4,
                                    rt::runtime::kMemcpyDeviceToHost,
                                    storage.stream) == rt::runtime::kSuccess);
    assert(rt::runtime::MemcpyAsync(result.data() + 24, converted.data(), 6 * 4,
                                    rt::runtime::kMemcpyDeviceToHost,
                                    storage.stream) == rt::runtime::kSuccess);
  }
  assert(rt::runtime::StreamSynchronize(storage.stream) ==
         rt::runtime::kSuccess);
  return result;
}
}  // namespace

PYBIND11_MODULE(_encoder_external_memory, module) { module.def("run", &Run); }
