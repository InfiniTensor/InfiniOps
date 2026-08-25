#ifndef INFINI_OPS_ASCEND_FUSED_ADD_RMS_NORM_KERNEL_CUSTOM_H_
#define INFINI_OPS_ASCEND_FUSED_ADD_RMS_NORM_KERNEL_CUSTOM_H_

#ifdef INFINI_HAS_CUSTOM_KERNELS

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cast.h"
#include "base/fused_add_rms_norm.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

extern "C" uint32_t aclrtlaunch_add_rms_norm(
    uint32_t block_dim, aclrtStream stream, void* x1, void* x2, void* weight,
    void* y, void* x_out, int64_t total_rows, int64_t dim_length,
    int64_t dim_length_align, int64_t former_num, int64_t former_length,
    int64_t tail_length, float eps, int64_t dtype_size);

namespace infini::ops {
namespace ascend_fused_add_rms_norm_custom_detail {

inline void* AllocateFloatOnes(Tensor::Size dim) {
  auto bytes = dim * sizeof(float);
  void* data = nullptr;
  auto ret = aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
  assert(ret == ACL_SUCCESS &&
         "`FusedAddRmsNorm` AscendC unit weight allocation failed");

  std::vector<float> host(dim, 1.0f);
  ret = aclrtMemcpy(data, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  assert(ret == ACL_SUCCESS &&
         "`FusedAddRmsNorm` AscendC unit weight initialization failed");

  return data;
}

}  // namespace ascend_fused_add_rms_norm_custom_detail

// Existing AscendC AddRmsNorm kernel, exposed as implementation index 2 for
// precision and performance comparison with the ACLNN implementations.
template <>
class Operator<FusedAddRmsNorm, Device::Type::kAscend, 2>
    : public FusedAddRmsNorm {
 public:
  Operator(Tensor input, Tensor residual, const std::optional<Tensor> weight,
           float epsilon)
      : FusedAddRmsNorm(input, residual, weight, epsilon),
        dtype_(input.dtype()),
        weight_dtype_(weight.has_value() ? weight->dtype()
                                         : DataType::kFloat32) {
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kFloat32) &&
           "`FusedAddRmsNorm` AscendC kernel supports fp16 and fp32");
    assert(input.IsContiguous() && residual.IsContiguous() &&
           (!weight.has_value() || weight->IsContiguous()) &&
           "`FusedAddRmsNorm` AscendC kernel requires contiguous tensors");

    auto align_elements = 32 / static_cast<int64_t>(kDataTypeToSize.at(dtype_));
    assert(static_cast<int64_t>(dim_) % align_elements == 0 &&
           "`FusedAddRmsNorm` AscendC kernel requires a 32-byte aligned "
           "last dimension");
    assert(dim_ <= 4096 &&
           "`FusedAddRmsNorm` AscendC kernel exceeds the conservative UB "
           "capacity bound; use index 0");

    total_rows_ = static_cast<int64_t>(input.numel() / dim_);
    if (!weight.has_value()) {
      weight_fp32_data_ =
          ascend_fused_add_rms_norm_custom_detail::AllocateFloatOnes(dim_);
    } else if (weight_dtype_ != DataType::kFloat32) {
      auto bytes = static_cast<size_t>(dim_) * sizeof(float);
      auto ret =
          aclrtMalloc(&weight_fp32_data_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      assert(ret == ACL_SUCCESS &&
             "`FusedAddRmsNorm` AscendC weight allocation failed");
      weight_src_cache_ =
          ascend::AclTensorCache({static_cast<int64_t>(dim_)},
                                 ascend::ToAclDtype(weight_dtype_), nullptr);
      weight_dst_cache_ = ascend::AclTensorCache({static_cast<int64_t>(dim_)},
                                                 ACL_FLOAT, weight_fp32_data_);
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    weight_src_cache_.release();
    weight_dst_cache_.release();
    if (weight_fp32_data_) aclrtFree(weight_fp32_data_);
  }

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    void* weight_fp32 = weight_fp32_data_;

    if (weight.has_value() && weight_dtype_ == DataType::kFloat32) {
      weight_fp32 = const_cast<void*>(weight->data());
    } else if (weight.has_value()) {
      auto t_src = weight_src_cache_.get(const_cast<void*>(weight->data()));
      auto t_dst = weight_dst_cache_.get(weight_fp32_data_);
      if (!cast_executor_) {
        aclnnCastGetWorkspaceSize(t_src, ACL_FLOAT, t_dst, &cast_ws_size_,
                                  &cast_executor_);
        aclSetAclOpExecutorRepeatable(cast_executor_);
      } else {
        aclSetInputTensorAddr(cast_executor_, 0, t_src,
                              const_cast<void*>(weight->data()));
        aclSetOutputTensorAddr(cast_executor_, 0, t_dst, weight_fp32_data_);
      }

      auto& arena = ascend::GetWorkspacePool().Ensure(stream, cast_ws_size_);
      aclnnCast(arena.buf, cast_ws_size_, cast_executor_, stream);
      weight_fp32 = weight_fp32_data_;
    }

    static constexpr int64_t kMaxBlockDim = 40;
    auto used_cores = std::min(total_rows_, kMaxBlockDim);
    auto former_length = (total_rows_ + used_cores - 1) / used_cores;
    auto tail_length = former_length - 1;
    auto former_num = total_rows_ - tail_length * used_cores;

    aclrtlaunch_add_rms_norm(
        static_cast<uint32_t>(used_cores), stream, input.data(),
        residual.data(), weight_fp32, input.data(), residual.data(),
        total_rows_, static_cast<int64_t>(dim_), static_cast<int64_t>(dim_),
        former_num, former_length, tail_length, epsilon,
        static_cast<int64_t>(kDataTypeToSize.at(dtype_)));
  }

 private:
  DataType dtype_;

  DataType weight_dtype_;

  int64_t total_rows_{0};

  void* weight_fp32_data_{nullptr};

  mutable ascend::AclTensorCache weight_src_cache_;

  mutable ascend::AclTensorCache weight_dst_cache_;

  mutable aclOpExecutor* cast_executor_{nullptr};

  mutable uint64_t cast_ws_size_{0};
};

}  // namespace infini::ops

#endif  // INFINI_HAS_CUSTOM_KERNELS
#endif  // INFINI_OPS_ASCEND_FUSED_ADD_RMS_NORM_KERNEL_CUSTOM_H_
