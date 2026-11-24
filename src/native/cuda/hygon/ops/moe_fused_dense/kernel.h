#ifndef INFINI_OPS_HYGON_MOE_FUSED_DENSE_KERNEL_H_
#define INFINI_OPS_HYGON_MOE_FUSED_DENSE_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "base/moe_fused_dense.h"
#include "native/cuda/device_buffer.h"

#if defined(WITH_TRITON)
#include "triton/ops/moe_fused_dense/jit.h"
#endif

namespace infini::ops {

namespace moe_fused_dense_detail {

// Whether the library carries the native expert kernels, i.e. was built for a Hygon
// architecture with the matrix instructions they use, and they are not switched off.
// Defined in kernel.cu, so that every translation unit sees the same answer.
bool HasNativeKernels();

}  // namespace moe_fused_dense_detail

// Fused MoE expert computation on Hygon. The shapes the native gfx936 kernels take (see
// NativeSupports) run without Triton. With Triton in the build, everything else is left to
// the Triton JIT kernels, whose slot this provider shares.
template <>
class Operator<MoeFusedDense, Device::Type::kHygon, 0> : public MoeFusedDense {
 public:
  Operator(Tensor output, const Tensor hidden, const Tensor w13,
           const Tensor w2, const Tensor weights, const Tensor ids,
           const Tensor sorted, const Tensor experts, const Tensor total,
           size_t block_size, const std::optional<Tensor> bias1 = std::nullopt,
           const std::optional<Tensor> bias2 = std::nullopt)
      : MoeFusedDense{output,     hidden, w13,          w2,    weights,
                      ids,        sorted,  experts,     total, block_size,
                      bias1,      bias2},
        device_index_{output.device().index()} {}

  // Block size 16 without expert biases, like the Triton kernels.
  static bool supports(Tensor output, const Tensor hidden, const Tensor w13,
                       const Tensor w2, const Tensor weights,
                       const Tensor ids, const Tensor sorted,
                       const Tensor experts, const Tensor total,
                       size_t block_size,
                       const std::optional<Tensor> bias1 = std::nullopt,
                       const std::optional<Tensor> bias2 = std::nullopt) {
    if (block_size != 16 || bias1 || bias2 ||
        !::infini::ops::MoeFusedDense::supports(
            output, hidden, w13, w2, weights, ids, sorted, experts, total,
            block_size, bias1, bias2)) {
      return false;
    }
    if (NativeSupports(hidden, w2, block_size, bias1, bias2)) return true;
#if defined(WITH_TRITON)
    return true;  // the Triton JIT kernels take every shape of the contract above
#else
    return false;
#endif
  }

  // The native kernels need the 16-byte loads of their 128-wide k steps to stay inside a
  // row: h and i multiples of 8, at least 32.
  static bool NativeSupports(const Tensor& hidden, const Tensor& w2,
                             size_t block_size,
                             const std::optional<Tensor>& bias1,
                             const std::optional<Tensor>& bias2) {
    const auto h = hidden.size(1), i = w2.size(2);
    return moe_fused_dense_detail::HasNativeKernels() && block_size == 16 &&
           !bias1 && !bias2 && h >= 32 && h % 8 == 0 && i >= 32 && i % 8 == 0;
  }

  // Native: act[pairs, i] in the element type, then the fp32 expert outputs [pairs, h].
  // Triton: [pairs, 2 * i + h] in the element type.
  std::size_t workspace_size_in_bytes() const override {
    const auto pairs = metadata_[1].size(0) * metadata_[5].size(1);
    const auto i = metadata_[3].size(2), h = metadata_[1].size(1);
    std::size_t bytes = pairs * (i * metadata_[1].element_size() +
                                 h * sizeof(float));
#if defined(WITH_TRITON)
    bytes = std::max(bytes, pairs * (2 * i + h) * metadata_[1].element_size());
#endif
    return bytes;
  }

  void operator()(Tensor output, const Tensor hidden, const Tensor w13,
                  const Tensor w2, const Tensor weights, const Tensor ids,
                  const Tensor sorted, const Tensor experts,
                  const Tensor total, size_t block_size,
                  const std::optional<Tensor> bias1,
                  const std::optional<Tensor> bias2) const override {
    Invocation(output, hidden, w13, w2, weights, ids, sorted, experts, total,
               block_size, bias1, bias2);
    // The Python bindings call without a handle: use the caller's workspace
    // when it is large enough, else a buffer owned by this operator.
    void* scratch =
        workspace_ && workspace_size_in_bytes_ >= workspace_size_in_bytes()
            ? workspace_
            : fallback_.Allocate(workspace_size_in_bytes());
    if (NativeSupports(hidden, w2, block_size, bias1, bias2)) {
      Run(output, hidden, w13, w2, weights, ids, sorted, experts, total,
          block_size, scratch);
      return;
    }
#if defined(WITH_TRITON)
    triton::jit::MoeFusedDense<Device::Type::kHygon> fallback{
        output,      hidden,  w13,    w2,          weights, ids,
        sorted,      experts, total,  block_size,  bias1,    bias2};
    fallback.set_stream(stream_);
    fallback.set_workspace(scratch);
    fallback.set_workspace_size_in_bytes(workspace_size_in_bytes());
    fallback(output, hidden, w13, w2, weights, ids, sorted, experts, total,
             block_size, bias1, bias2);
#else
    throw std::invalid_argument(
        "`MoeFusedDense` on Hygon needs h and i to be multiples of 8 of at least 32");
#endif
  }

 protected:
  void Run(Tensor output, const Tensor hidden, const Tensor w13,
           const Tensor w2, const Tensor weights, const Tensor ids,
           const Tensor sorted, const Tensor experts, const Tensor total,
           size_t block_size, void* scratch) const;

 private:
  int device_index_;

  mutable CudaDeviceBuffer<Runtime<Device::Type::kHygon>> fallback_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_MOE_FUSED_DENSE_KERNEL_H_
