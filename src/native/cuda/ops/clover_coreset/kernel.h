#ifndef INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_H_

#include <algorithm>
#include <cstdint>

#include "base/clover_coreset.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_coreset/kernel.cuh"

namespace infini::ops {

namespace coreset_launch_detail {

// A lane's row of eight elements goes through 16-byte accesses when the
// element is 16 or 32 bits wide and both tensors start on a 16-byte boundary.
inline bool VectorRows(const Tensor& logits, const Tensor& out) {
  const auto element = logits.element_size();
  return (element == 2 || element == 4) &&
         reinterpret_cast<std::uintptr_t>(logits.data()) % 16 == 0 &&
         reinterpret_cast<std::uintptr_t>(out.data()) % 16 == 0;
}

}  // namespace coreset_launch_detail

template <typename Backend>
class CudaCloverCoreset : public CloverCoreset {
 public:
  using CloverCoreset::CloverCoreset;

  void operator()(const Tensor logits, size_t block_size, double top_p,
                  Tensor out) const override {
    const auto blocks = logits.size(0) / block_size;
    if (blocks == 0) return;
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    // One thread block per routing block (no workspace: the masses and the
    // allow list stay in its shared memory): a warp per row, at least the
    // eight warps of the sort. The shared memory size stays below the 48 KB
    // every device gives a block without opting in.
    const int rows = static_cast<int>(block_size);
    const int threads = (std::max)(rows, clover_coreset_detail::kSortWarps) *
                        clover_coreset_detail::kWarp;
    const int vector = coreset_launch_detail::VectorRows(logits, out) ? 1 : 0;

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(logits.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          CloverCoresetKernel<Backend::kDeviceType, Data>
              <<<static_cast<unsigned>(blocks), threads,
                 clover_coreset_detail::kSmemBytes, cuda_stream>>>(
                  reinterpret_cast<const Data*>(logits.data()),
                  reinterpret_cast<Data*>(out.data()), rows,
                  static_cast<float>(top_p), vector);
        },
        "CudaCloverCoreset::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_H_
