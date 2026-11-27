#ifndef INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_H_

#include <cstdint>

#include "base/clover_merge_moe.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_merge_moe/kernel.cuh"

namespace infini::ops {

namespace clover_merge_moe_detail {

constexpr unsigned int kThreads = 128;
constexpr int kVectorBytes = 16;

// Whether a thread can take 16 bytes of a row: the element is 16 or 32
// bits wide, the row's width is a whole number of vectors, and every tensor
// starts on a 16-byte boundary.
inline bool VectorRows(const Tensor& fresh, const Tensor& cache,
                       const Tensor& out) {
  const auto element = fresh.element_size();
  if (element != 2 && element != 4) return false;
  if (fresh.size(1) % (kVectorBytes / element) != 0) return false;
  for (const Tensor* tensor : {&fresh, &cache, &out}) {
    if (reinterpret_cast<std::uintptr_t>(tensor->data()) % kVectorBytes != 0)
      return false;
  }
  return true;
}

}  // namespace clover_merge_moe_detail

template <typename Backend>
class CudaCloverMergeMoe : public CloverMergeMoe {
 public:
  using CloverMergeMoe::CloverMergeMoe;

  void operator()(const Tensor fresh, const Tensor live, Tensor cache,
                  Tensor out) const override {
    const auto rows = fresh.size(0);
    const auto width = fresh.size(1);
    if (rows == 0 || width == 0) return;
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    namespace detail = clover_merge_moe_detail;
    const int per_thread =
        detail::VectorRows(fresh, cache, out)
            ? static_cast<int>(detail::kVectorBytes / fresh.element_size())
            : 1;
    const auto columns = width / per_thread;
    // Rows on the x axis: it takes 2^31 blocks where y would stop at 65535.
    const dim3 grid(
        static_cast<unsigned>(rows),
        static_cast<unsigned>(utils::CeilDiv(columns, detail::kThreads)));

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(fresh.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          const Data* fresh_data = reinterpret_cast<const Data*>(fresh.data());
          const int8_t* live_data =
              reinterpret_cast<const int8_t*>(live.data());
          Data* cache_data = reinterpret_cast<Data*>(cache.data());
          Data* out_data = reinterpret_cast<Data*>(out.data());
          if (per_thread == 1) {
            CloverMergeMoeKernel<Backend::kDeviceType, Data>
                <<<grid, detail::kThreads, 0, cuda_stream>>>(
                    fresh_data, live_data, cache_data, out_data, width);
          } else {
            CloverMergeMoeVectorKernel<Backend::kDeviceType, Data>
                <<<grid, detail::kThreads, 0, cuda_stream>>>(
                    fresh_data, live_data, cache_data, out_data, columns);
          }
        },
        "CudaCloverMergeMoe::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_H_
