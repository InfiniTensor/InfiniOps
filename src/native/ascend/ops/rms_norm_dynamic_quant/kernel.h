#ifndef INFINI_OPS_ASCEND_RMS_NORM_DYNAMIC_QUANT_KERNEL_H_
#define INFINI_OPS_ASCEND_RMS_NORM_DYNAMIC_QUANT_KERNEL_H_

#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_rms_norm_dynamic_quant.h"
#include "base/rms_norm_dynamic_quant.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<RmsNormDynamicQuant, Device::Type::kAscend, 0>
    : public RmsNormDynamicQuant {
 public:
  static bool supports(Tensor packed, Tensor packed2, Tensor scale,
                       Tensor scale2, const Tensor input, const Tensor gamma,
                       double epsilon) {
    const auto rows = input.shape()[0];
    const auto cols = input.shape()[1];
    return input.device().type() == Device::Type::kAscend &&
           input.ndim() == 2 && rows > 0 && cols > 0 &&
           (input.dtype() == DataType::kFloat16 ||
            input.dtype() == DataType::kBFloat16) &&
           input.IsContiguous() && gamma.device() == input.device() &&
           gamma.dtype() == input.dtype() &&
           gamma.shape() == Tensor::Shape{cols} && gamma.IsContiguous() &&
           packed.device() == input.device() &&
           packed.dtype() == DataType::kInt8 &&
           packed.shape() == input.shape() && packed.IsContiguous() &&
           packed2.device() == input.device() &&
           packed2.dtype() == DataType::kInt8 &&
           packed2.shape() == Tensor::Shape{1} && packed2.IsContiguous() &&
           scale.device() == input.device() &&
           scale.dtype() == DataType::kFloat32 &&
           scale.shape() == Tensor::Shape{rows} && scale.IsContiguous() &&
           scale2.device() == input.device() &&
           scale2.dtype() == DataType::kFloat32 &&
           scale2.shape() == scale.shape() && scale2.IsContiguous() &&
           epsilon > 0.0;
  }

  Operator(Tensor packed, Tensor packed2, Tensor scale, Tensor scale2,
           const Tensor input, const Tensor gamma, double epsilon)
      : RmsNormDynamicQuant(packed, packed2, scale, scale2, input, gamma,
                            epsilon),
        packed_cache_(packed), packed2_cache_(packed2), scale_cache_(scale),
        scale2_cache_(scale2), input_cache_(input), gamma_cache_(gamma),
        epsilon_(epsilon) {
    if (!supports(packed, packed2, scale, scale2, input, gamma, epsilon)) {
      throw std::invalid_argument(
          "unsupported Ascend RmsNormDynamicQuant operands");
    }
    t_packed_ = packed_cache_.get(packed.data());
    t_packed2_ = packed2_cache_.get(packed2.data());
    t_scale_ = scale_cache_.get(scale.data());
    t_scale2_ = scale2_cache_.get(scale2.data());
    t_input_ = input_cache_.get(const_cast<void *>(input.data()));
    t_gamma_ = gamma_cache_.get(const_cast<void *>(gamma.data()));
    ascend::CheckAcl(
        aclnnRmsNormDynamicQuantGetWorkspaceSize(
            t_input_, t_gamma_, nullptr, nullptr, nullptr, epsilon_, nullptr,
            int64_t{0}, t_packed_, t_packed2_, t_scale_, t_scale2_,
            &workspace_size_, &executor_),
        "RmsNormDynamicQuant workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "RmsNormDynamicQuant repeatable executor setup");
  }

  ~Operator() override {
    if (!ascend::IsAclRuntimeAlive()) return;
    packed_cache_.release();
    packed2_cache_.release();
    scale_cache_.release();
    scale2_cache_.release();
    input_cache_.release();
    gamma_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor packed, Tensor packed2, Tensor scale, Tensor scale2,
                  const Tensor input, const Tensor gamma,
                  double epsilon) const override {
    if (!supports(packed, packed2, scale, scale2, input, gamma, epsilon)) {
      throw std::invalid_argument(
          "unsupported Ascend RmsNormDynamicQuant operands");
    }
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 0, t_input_,
                         const_cast<void *>(input.data())),
                     "RmsNormDynamicQuant input address update");
    ascend::CheckAcl(
        AclSetInputTensorAddr(executor_, 1, t_gamma_,
                              const_cast<void *>(gamma.data())),
        "RmsNormDynamicQuant gamma address update");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 0, t_packed_, packed.data()),
        "RmsNormDynamicQuant packed address update");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 1, t_packed2_, packed2.data()),
        "RmsNormDynamicQuant packed2 address update");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 2, t_scale_, scale.data()),
        "RmsNormDynamicQuant scale address update");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 3, t_scale2_, scale2.data()),
        "RmsNormDynamicQuant scale2 address update");

    auto stream = static_cast<aclrtStream>(stream_);
    auto &arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "rms_norm_dynamic_quant");
    ascend::CheckAcl(
        aclnnRmsNormDynamicQuant(arena.buf, workspace_size_, executor_,
                                 stream),
        "RmsNormDynamicQuant execution");
  }

 private:
  mutable ascend::AclTensorCache packed_cache_;
  mutable ascend::AclTensorCache packed2_cache_;
  mutable ascend::AclTensorCache scale_cache_;
  mutable ascend::AclTensorCache scale2_cache_;
  mutable ascend::AclTensorCache input_cache_;
  mutable ascend::AclTensorCache gamma_cache_;
  double epsilon_;
  aclTensor *t_packed_{nullptr};
  aclTensor *t_packed2_{nullptr};
  aclTensor *t_scale_{nullptr};
  aclTensor *t_scale2_{nullptr};
  aclTensor *t_input_{nullptr};
  aclTensor *t_gamma_{nullptr};
  mutable aclOpExecutor *executor_{nullptr};
  mutable uint64_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
