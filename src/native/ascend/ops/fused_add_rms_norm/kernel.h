#ifndef INFINI_OPS_ASCEND_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "aclnn_inplace_add_rms_norm.h"
#include "base/fused_add_rms_norm.h"
#include "base/rms_norm.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {
namespace ascend_fused_add_rms_norm_detail {

inline void* AllocateOnes(Tensor::Size dim, DataType dtype) {
  auto element_size = kDataTypeToSize.at(dtype);
  auto bytes = dim * element_size;
  void* data = nullptr;
  auto ret = aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
  ascend::CheckAcl(ret, "FusedAddRmsNorm unit weight allocation");

  if (dtype == DataType::kFloat32) {
    std::vector<float> host(dim, 1.0f);
    ret =
        aclrtMemcpy(data, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  } else {
    assert((dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) &&
           "`FusedAddRmsNorm` Ascend path supports fp16, bf16, or fp32");
    auto one_bits =
        static_cast<uint16_t>(dtype == DataType::kFloat16 ? 0x3c00 : 0x3f80);
    std::vector<uint16_t> host(dim, one_bits);
    ret =
        aclrtMemcpy(data, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  }

  ascend::CheckAcl(ret, "FusedAddRmsNorm unit weight initialization");
  return data;
}

}  // namespace ascend_fused_add_rms_norm_detail

// The single-call ACLNN implementation is the default after passing the full
// precision matrix and outperforming the decomposition on all Llama shapes.
template <>
class Operator<FusedAddRmsNorm, Device::Type::kAscend>
    : public FusedAddRmsNorm {
 public:
  Operator(Tensor input, Tensor residual, const std::optional<Tensor> weight,
           float epsilon)
      : FusedAddRmsNorm(input, residual, weight, epsilon),
        shape_(input.shape()),
        dtype_(input.dtype()),
        element_size_(kDataTypeToSize.at(input.dtype())),
        row_bytes_(dim_ * element_size_),
        tensor_bytes_(num_tokens_ * row_bytes_),
        needs_input_staging_(!input.IsContiguous()),
        needs_residual_staging_(!residual.IsContiguous()) {
    use_small_fallback_ = ascend::IsAscend310P() &&
                          dtype_ == DataType::kFloat16 && row_bytes_ < 32;
    if (!weight.has_value()) {
      unit_weight_data_ =
          ascend_fused_add_rms_norm_detail::AllocateOnes(dim_, input.dtype());
      unit_weight_.emplace(unit_weight_data_, Tensor::Shape{dim_},
                           input.dtype(), input.device());
    }

    const auto& resolved_weight = weight.has_value() ? *weight : *unit_weight_;
    weight_cache_ = ascend::AclTensorCache(resolved_weight);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    if (needs_input_staging_) {
      input_cache_ =
          ascend::AclTensorCache(shape, ascend::ToAclDtype(dtype_), nullptr);
    } else {
      input_cache_ = ascend::AclTensorCache(input);
    }
    if (needs_residual_staging_) {
      residual_cache_ =
          ascend::AclTensorCache(shape, ascend::ToAclDtype(dtype_), nullptr);
    } else {
      residual_cache_ = ascend::AclTensorCache(residual);
    }

    rstd_shape_.assign(input.shape().begin(), input.shape().end());
    rstd_shape_.back() = 1;
    rstd_size_ = num_tokens_ * sizeof(float);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    input_cache_.release();
    residual_cache_.release();
    weight_cache_.release();
    if (unit_weight_data_) aclrtFree(unit_weight_data_);
    // `rstd_tensor_` remains owned by the repeatable ACLNN executor.
  }

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    void* input_data = input.data();
    void* residual_data = residual.data();

    if (needs_input_staging_) {
      auto& staging = ascend::GetWorkspacePool().Ensure(
          stream, tensor_bytes_, "fused_add_rms_norm_input");
      PackRows(staging.buf, input.data(), input_strides_, stream);
      input_data = staging.buf;
    }
    if (needs_residual_staging_) {
      auto& staging = ascend::GetWorkspacePool().Ensure(
          stream, tensor_bytes_, "fused_add_rms_norm_residual");
      PackRows(staging.buf, residual.data(), residual_strides_, stream);
      residual_data = staging.buf;
    }

    const void* weight_data =
        weight.has_value() ? weight->data() : unit_weight_data_;
    auto t_input = input_cache_.get(input_data);
    auto t_residual = residual_cache_.get(residual_data);
    auto t_weight = weight_cache_.get(const_cast<void*>(weight_data));

    if (use_small_fallback_) {
      RunSmallFallback(input, input_data, residual_data, t_weight, epsilon,
                       stream);
      if (needs_input_staging_) {
        UnpackRows(input.data(), input_data, input_strides_, stream);
      }
      if (needs_residual_staging_) {
        UnpackRows(residual.data(), residual_data, residual_strides_, stream);
      }
      return;
    }

    auto& rstd_arena = ascend::GetWorkspacePool().Ensure(
        stream, rstd_size_, "fused_add_rms_norm_rstd");
    if (!rstd_tensor_) {
      rstd_tensor_ = aclCreateTensor(
          rstd_shape_.data(), static_cast<int64_t>(rstd_shape_.size()),
          ACL_FLOAT, /*strides=*/nullptr, 0, ACL_FORMAT_ND, rstd_shape_.data(),
          static_cast<int64_t>(rstd_shape_.size()), rstd_arena.buf);
    } else {
      ascend::CheckAcl(aclSetRawTensorAddr(rstd_tensor_, rstd_arena.buf),
                       "FusedAddRmsNorm rstd address");
    }

    if (!executor_) {
      ascend::CheckAcl(
          aclnnInplaceAddRmsNormGetWorkspaceSize(
              t_input, t_residual, t_weight, static_cast<double>(epsilon),
              rstd_tensor_, &ws_size_, &executor_),
          "FusedAddRmsNorm workspace query");
      ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                       "FusedAddRmsNorm repeatable executor");
    } else {
      ascend::CheckAcl(aclSetInputTensorAddr(executor_, 0, t_input, input_data),
                       "FusedAddRmsNorm input address");
      ascend::CheckAcl(
          aclSetInputTensorAddr(executor_, 1, t_residual, residual_data),
          "FusedAddRmsNorm residual address");
      ascend::CheckAcl(aclSetInputTensorAddr(executor_, 2, t_weight,
                                             const_cast<void*>(weight_data)),
                       "FusedAddRmsNorm weight address");
      // InplaceAddRmsNorm has outputs [x1Ref, rstdOut, x2Ref]. Rebind
      // the in-place outputs too when a cached executor sees new buffers.
      ascend::CheckAcl(
          aclSetOutputTensorAddr(executor_, 0, t_input, input_data),
          "FusedAddRmsNorm input output address");
      ascend::CheckAcl(
          aclSetOutputTensorAddr(executor_, 1, rstd_tensor_, rstd_arena.buf),
          "FusedAddRmsNorm rstd output address");
      ascend::CheckAcl(
          aclSetOutputTensorAddr(executor_, 2, t_residual, residual_data),
          "FusedAddRmsNorm residual output address");
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    ascend::CheckAcl(
        aclnnInplaceAddRmsNorm(arena.buf, ws_size_, executor_, stream),
        "FusedAddRmsNorm execution");

    if (needs_input_staging_) {
      UnpackRows(input.data(), input_data, input_strides_, stream);
    }
    if (needs_residual_staging_) {
      UnpackRows(residual.data(), residual_data, residual_strides_, stream);
    }
  }

 private:
  void RunSmallFallback(const Tensor input, void* input_data,
                        void* residual_data, const aclTensor* t_weight,
                        float epsilon, aclrtStream stream) const {
    // A single short row works on 310P, whereas packed short rows do not.
    // Separate 32-byte arenas avoid adjacent-row reads/writes in that path.
    auto& input_arena = ascend::GetWorkspacePool().Ensure(
        stream, 32, "fused_add_rms_norm_short_input");
    auto& residual_arena = ascend::GetWorkspacePool().Ensure(
        stream, 32, "fused_add_rms_norm_short_residual");
    auto& rstd_arena = ascend::GetWorkspacePool().Ensure(
        stream, 32, "fused_add_rms_norm_short_rstd");
    Tensor row_input{input_arena.buf, Tensor::Shape{1, dim_}, dtype_,
                     input.device()};
    Tensor row_residual{residual_arena.buf, Tensor::Shape{1, dim_}, dtype_,
                        input.device()};
    Tensor rstd{rstd_arena.buf, Tensor::Shape{1, 1}, DataType::kFloat32,
                input.device()};
    auto t_input = ascend::OwnAclTensor(ascend::BuildAclTensor(row_input));
    auto t_residual =
        ascend::OwnAclTensor(ascend::BuildAclTensor(row_residual));
    auto t_rstd = ascend::OwnAclTensor(ascend::BuildAclTensor(rstd));
    for (Tensor::Size row = 0; row < num_tokens_; ++row) {
      auto* input_row = static_cast<uint8_t*>(input_data) + row * row_bytes_;
      auto* residual_row =
          static_cast<uint8_t*>(residual_data) + row * row_bytes_;
      ascend::CheckAcl(
          aclrtMemcpyAsync(input_arena.buf, 32, input_row, row_bytes_,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
          "FusedAddRmsNorm short input copy");
      ascend::CheckAcl(
          aclrtMemcpyAsync(residual_arena.buf, 32, residual_row, row_bytes_,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
          "FusedAddRmsNorm short residual copy");
      uint64_t workspace_size = 0;
      aclOpExecutor* executor = nullptr;
      ascend::CheckAcl(aclnnInplaceAddRmsNormGetWorkspaceSize(
                           t_input.get(), t_residual.get(), t_weight,
                           static_cast<double>(epsilon), t_rstd.get(),
                           &workspace_size, &executor),
                       "FusedAddRmsNorm short-row workspace query");
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
      ascend::CheckAcl(
          aclnnInplaceAddRmsNorm(arena.buf, workspace_size, executor, stream),
          "FusedAddRmsNorm short-row execution");
      ascend::CheckAcl(
          aclrtMemcpyAsync(input_row, row_bytes_, input_arena.buf, row_bytes_,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
          "FusedAddRmsNorm short input writeback");
      ascend::CheckAcl(
          aclrtMemcpyAsync(residual_row, row_bytes_, residual_arena.buf,
                           row_bytes_, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
          "FusedAddRmsNorm short residual writeback");
    }
  }

  int64_t RowOffset(int64_t row, const Tensor::Strides& strides) const {
    int64_t remaining = row;
    int64_t offset = 0;
    for (int64_t axis = static_cast<int64_t>(shape_.size()) - 2; axis >= 0;
         --axis) {
      auto coordinate = remaining % static_cast<int64_t>(shape_[axis]);
      remaining /= static_cast<int64_t>(shape_[axis]);
      offset += coordinate * static_cast<int64_t>(strides[axis]);
    }
    return offset;
  }

  void PackRows(void* dst, const void* src, const Tensor::Strides& strides,
                aclrtStream stream) const {
    for (int64_t row = 0; row < static_cast<int64_t>(num_tokens_); ++row) {
      auto offset = RowOffset(row, strides);
      auto ret = aclrtMemcpyAsync(
          static_cast<uint8_t*>(dst) + row * row_bytes_, row_bytes_,
          static_cast<const uint8_t*>(src) + offset * element_size_, row_bytes_,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      ascend::CheckAcl(ret, "FusedAddRmsNorm input pack");
    }
  }

  void UnpackRows(void* dst, const void* src, const Tensor::Strides& strides,
                  aclrtStream stream) const {
    for (int64_t row = 0; row < static_cast<int64_t>(num_tokens_); ++row) {
      auto offset = RowOffset(row, strides);
      auto ret = aclrtMemcpyAsync(
          static_cast<uint8_t*>(dst) + offset * element_size_, row_bytes_,
          static_cast<const uint8_t*>(src) + row * row_bytes_, row_bytes_,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      ascend::CheckAcl(ret, "FusedAddRmsNorm output unpack");
    }
  }

  Tensor::Shape shape_;

  DataType dtype_;

  uint64_t element_size_{0};

  uint64_t row_bytes_{0};

  uint64_t tensor_bytes_{0};

  bool needs_input_staging_{false};

  bool needs_residual_staging_{false};

  bool use_small_fallback_{false};

  void* unit_weight_data_{nullptr};

  std::optional<Tensor> unit_weight_;

  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache residual_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  std::vector<int64_t> rstd_shape_;

  uint64_t rstd_size_{0};

  mutable aclTensor* rstd_tensor_{nullptr};

  mutable aclOpExecutor* executor_{nullptr};

  mutable uint64_t ws_size_{0};
};

}  // namespace infini::ops

#endif
