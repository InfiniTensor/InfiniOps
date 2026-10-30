// Link with --wrap=cnrtMalloc,--wrap=cnrtFree to count allocations made by
// the real provider header, while running its real kernel on the device.
#include <cassert>
#include <iostream>
#include <vector>

#include "native/cambricon/ops/relu/kernel.h"

static size_t allocations = 0, frees = 0, allocated_bytes = 0;
extern "C" cnrtRet_t __real_cnrtMalloc(void**, size_t);
extern "C" cnrtRet_t __real_cnrtFree(void*);
extern "C" cnrtRet_t __wrap_cnrtMalloc(void** ptr, size_t bytes) {
  const auto result = __real_cnrtMalloc(ptr, bytes);
  if (result == static_cast<cnrtRet_t>(0) && bytes) {
    ++allocations;
    allocated_bytes += bytes;
  }
  return result;
}
extern "C" cnrtRet_t __wrap_cnrtFree(void* ptr) {
  if (ptr) ++frees;
  return __real_cnrtFree(ptr);
}

int main() {
  namespace rt = infini::rt;
  namespace ops = infini::ops;
  rt::set_runtime_device_type(rt::Device::Type::kCambricon);
  CNRT_CHECK(cnrtSetDevice(0));
  cnrtQueue_t queue{};
  CNRT_CHECK(cnrtQueueCreate(&queue));
  const rt::Device device{rt::Device::Type::kCambricon, 0};
  constexpr size_t count = 4096u * 4096u, bytes = count * sizeof(uint16_t);
  void *x{}, *y{}, *workspace{};
  CNRT_CHECK(cnrtMalloc(&x, bytes));
  CNRT_CHECK(cnrtMalloc(&y, bytes));
  const ops::Tensor input{x, {4096, 4096}, rt::DataType::kFloat16, device};
  const ops::Tensor output{y, {4096, 4096}, rt::DataType::kFloat16, device};
  using Relu = ops::Operator<ops::Relu, rt::Device::Type::kCambricon>;
  const size_t scratch = Relu(input, output).workspace_size_in_bytes();
  CNRT_CHECK(cnrtMalloc(&workspace, scratch));
  std::vector<uint16_t> values(count, 0xbc00);  // -1 in FP16
  assert(rt::runtime::Memcpy(x, values.data(), bytes,
                             rt::runtime::kMemcpyHostToDevice) ==
         rt::runtime::kSuccess);
  for (int nodes : {1, 8, 64}) {
    allocations = frees = allocated_bytes = 0;
    {
      std::vector<std::unique_ptr<Relu>> providers;
      for (int i = 0; i < nodes; ++i) {
        providers.push_back(std::make_unique<Relu>(input, output));
        providers.back()->set_stream(queue);
        providers.back()->set_workspace(workspace);
        providers.back()->set_workspace_size_in_bytes(scratch);
      }
      assert(allocations == 0);  // construction must not allocate a fallback
      for (auto& op : providers) (*op)(input, output);
      CNRT_CHECK(cnrtQueueSync(queue));
      assert(allocations == 0 && allocated_bytes == 0);
    }
    assert(frees == 0);  // caller-owned workspace is never freed by providers
    std::cout << "nodes=" << nodes << " provider_mallocs=" << allocations
              << " provider_bytes=" << allocated_bytes
              << " shared_workspace_bytes=" << scratch << '\n';
  }
  assert(rt::runtime::Memcpy(values.data(), y, bytes,
                             rt::runtime::kMemcpyDeviceToHost) ==
         rt::runtime::kSuccess);
  for (auto v : values) assert(v == 0);
  allocations = frees = allocated_bytes = 0;
  {
    Relu op(input, output);
    op.set_stream(queue);
    assert(allocations == 0);
    op(input, output);
    CNRT_CHECK(cnrtQueueSync(queue));
    assert(allocations == 1 && allocated_bytes == scratch);
    op(input, output);
    CNRT_CHECK(cnrtQueueSync(queue));
    assert(allocations == 1);
    op.set_workspace(workspace);
    op.set_workspace_size_in_bytes(scratch);
    op(input, output);
    CNRT_CHECK(cnrtQueueSync(queue));
    assert(allocations == 1);
  }
  assert(frees == 1);
  std::cout << "fallback_mallocs=1 fallback_frees=1\n";
  CNRT_CHECK(cnrtFree(workspace));
  CNRT_CHECK(cnrtFree(y));
  CNRT_CHECK(cnrtFree(x));
  CNRT_CHECK(cnrtQueueDestroy(queue));
}
