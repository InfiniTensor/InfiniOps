#ifndef INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_
#define INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_

// Registration entry of the native MetaX DeepseekMoe provider. The
// shared declarations (constants, workspace layout/lease and the
// device-side template kernels) live in kernel.cuh; kernel.maca is the
// compilation unit that defines the host scheduling, the non-template
// routing kernels, and the RunChunks instantiations, so kernel launches
// and definitions share one TU per the repository convention.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "base/deepseek_moe.h"
#include "native/cuda/metax/blas.h"
#include "native/cuda/metax/ops/deepseek_moe/kernel.cuh"
#include "native/cuda/metax/runtime_.h"

namespace infini::ops {

// Native MetaX DeepseekMoe: expert-grouped GEMM path. The routing
// (count/scan/pack), activation and reduction kernels run on device and
// the per-expert projections go through mcBLAS, so no per-expert operator
// submits cross the dispatcher.
template <>
class Operator<DeepseekMoe, Device::Type::kMetax, 0> : public DeepseekMoe {
 public:
  using DeepseekMoe::DeepseekMoe;

  bool capture_safe() const override { return false; }

  size_t workspace_size_in_bytes() const override {
    const auto& hidden = metadata_[1];
    const size_t esz = hidden.element_size();
    const size_t ntokens = std::min<size_t>(
        hidden.size(0), metax_deepseek_moe_detail::kChunkTokens);
    const size_t routes = ntokens * metadata_[2].size(1);
    return metax_deepseek_moe_detail::Layout(num_experts_, routes,
                                             hidden.size(1), intermediate_, esz)
        .total;
  }

  void operator()(Tensor output, const Tensor hidden, const Tensor ids,
                  const Tensor weights, const std::vector<Tensor> gate,
                  const std::vector<Tensor> up, const std::vector<Tensor> down,
                  size_t intermediate, size_t num_experts) const override {
    namespace detail = metax_deepseek_moe_detail;
    using MetaxBlas = Blas<Device::Type::kMetax>;
    using MetaxRuntime = Runtime<Device::Type::kMetax>;

    Invocation(output, hidden, ids, weights, gate, up, down, intermediate,
               num_experts);
    if (hidden.size(0) == 0) {
      return;
    }
    moe_detail::Require(num_experts <= size_t(detail::kMaxExperts));
    moe_detail::Require(hidden.dtype() == DataType::kFloat16 ||
                        hidden.dtype() == DataType::kBFloat16);

    const size_t hidden_size = hidden.size(1);
    const size_t topk = ids.size(1);
    const size_t ntokens = hidden.size(0);
    const size_t esz = hidden.element_size();

    const auto stream =
        static_cast<MetaxRuntime::Stream>(stream_ ? stream_ : nullptr);
    const auto layout = detail::Layout(
        num_experts, std::min(ntokens, detail::kChunkTokens) * topk,
        hidden_size, intermediate, esz);
    // Dispatch callers pass a stable workspace; python bindings get a
    // stream-ordered lease that is freed after the submission completes.
    detail::WorkspaceLease lease(workspace_, workspace_size_in_bytes_,
                                 layout.total, stream);
    auto* base = static_cast<std::uint8_t*>(lease.data());

    auto handle = detail::GetHandle();
    if (MetaxBlas::BlasSetStream(handle, stream) != MCBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("MetaX DeepseekMoe BLAS set stream failed");
    }

    std::vector<const void*> gate_ptrs(num_experts);
    std::vector<const void*> up_ptrs(num_experts);
    std::vector<const void*> down_ptrs(num_experts);
    for (size_t e = 0; e < num_experts; ++e) {
      gate_ptrs[e] = gate[e].data();
      up_ptrs[e] = up[e].data();
      down_ptrs[e] = down[e].data();
    }

    if (hidden.dtype() == DataType::kFloat16) {
      detail::RunChunks<__half>(
          base, layout, ntokens, hidden_size, topk, intermediate, num_experts,
          reinterpret_cast<const __half*>(hidden.data()),
          reinterpret_cast<const int*>(ids.data()),
          reinterpret_cast<const float*>(weights.data()), gate_ptrs, up_ptrs,
          down_ptrs, reinterpret_cast<__half*>(output.data()), stream, handle);
    } else {
      detail::RunChunks<__nv_bfloat16>(
          base, layout, ntokens, hidden_size, topk, intermediate, num_experts,
          reinterpret_cast<const __nv_bfloat16*>(hidden.data()),
          reinterpret_cast<const int*>(ids.data()),
          reinterpret_cast<const float*>(weights.data()), gate_ptrs, up_ptrs,
          down_ptrs, reinterpret_cast<__nv_bfloat16*>(output.data()), stream,
          handle);
    }
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_
