#ifndef INFINI_OPS_ASCEND_I8_GEMM_KERNEL_H_
#define INFINI_OPS_ASCEND_I8_GEMM_KERNEL_H_

#include <optional>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"
#include "aclnnop/aclnn_quant_matmul_weight_nz.h"
#include "base/i8_gemm.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

namespace {
Tensor nd_weight_view(const Tensor& weight) {
  return Tensor{const_cast<void*>(weight.data()),
                {weight.shape()[1] * 16, weight.shape()[0] * 32},
                weight.dtype()};
}
}

template <>
class Operator<I8Gemm, Device::Type::kAscend, 0> : public I8Gemm {
 public:
  static bool supports(Tensor c, const Tensor a, const Tensor as,
                       const Tensor b, const Tensor bs,
                       const std::optional<Tensor> bias) {
    if (c.device().type() != Device::Type::kAscend || c.ndim() != 2 ||
        a.ndim() != 2 || (b.ndim() != 2 && b.ndim() != 4) ||
        a.dtype() != DataType::kInt8 ||
        b.dtype() != DataType::kInt8 ||
        (c.dtype() != DataType::kFloat16 &&
         c.dtype() != DataType::kBFloat16 &&
         c.dtype() != DataType::kFloat32) ||
        as.dtype() != DataType::kFloat32 || bs.dtype() != DataType::kFloat32 ||
        c.device() != a.device() || c.device() != b.device() ||
        c.device() != as.device() || c.device() != bs.device()) {
      return false;
    }

    const auto m = c.shape()[0];
    const auto n = c.shape()[1];
    const auto k = a.shape()[1];
    const bool nz_weight = b.ndim() == 4;
    const bool weight_geometry =
        nz_weight
            ? (n % 32 == 0 && k % 16 == 0 && b.shape()[0] == n / 32 &&
               b.shape()[1] == k / 16 && b.shape()[2] == 16 &&
               b.shape()[3] == 32 && b.IsContiguous())
            : (b.shape() == Tensor::Shape{k, n} &&
               ((b.stride(0) == 1 &&
                 b.stride(1) >= static_cast<int64_t>(k)) ||
                (b.stride(1) == 1 &&
                 b.stride(0) >= static_cast<int64_t>(n))));
    if (!m || !n || !k || a.shape()[0] != m ||
        !weight_geometry ||
        as.shape() != Tensor::Shape{m, 1} ||
        bs.shape() != Tensor::Shape{n, 1} || !c.IsContiguous() ||
        !a.IsContiguous() || !as.IsContiguous() || !bs.IsContiguous()) {
      return false;
    }

    return !bias ||
           (bias->device() == c.device() && bias->dtype() == c.dtype() &&
            bias->shape() == c.shape() && bias->stride(0) == 0 &&
            bias->stride(1) == 1);
  }

  Operator(Tensor c, const Tensor a, const Tensor as, const Tensor b,
           const Tensor bs, const std::optional<Tensor> bias)
      : I8Gemm(c, a, as, b, bs, bias),
        c_cache_(c),
        a_cache_(a),
        as_cache_({static_cast<int64_t>(c.shape()[0])}, ACL_FLOAT,
                  const_cast<void*>(as.data())),
        b_cache_(b.ndim() == 4 ? nd_weight_view(b) : b,
                 b.ndim() == 4 ? false : b.stride(0) == 1,
                 b.ndim() == 4
                     ? std::vector<int64_t>(b.shape().begin(), b.shape().end())
                     : std::vector<int64_t>{},
                 b.ndim() == 4 ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND),
        bs_cache_({static_cast<int64_t>(c.shape()[1])}, ACL_FLOAT,
                  const_cast<void*>(bs.data())),
        weight_nz_(b.ndim() == 4),
        has_bias_(bias.has_value()),
        transpose_x2_(b.ndim() != 4 && b.stride(0) == 1),
        t_c_(c_cache_.get(c.data())),
        t_a_(a_cache_.get(const_cast<void*>(a.data()))),
        t_as_(as_cache_.get(const_cast<void*>(as.data()))),
        t_b_(b_cache_.get(const_cast<void*>(b.data()))),
        t_bs_(bs_cache_.get(const_cast<void*>(bs.data()))) {
    if (!supports(c, a, as, b, bs, bias)) {
      throw std::invalid_argument("unsupported Ascend I8Gemm operands");
    }
    if (has_bias_) {
      bias_cache_ = ascend::AclTensorCache(
          {static_cast<int64_t>(c.shape()[1])},
          ascend::ToAclDtype(c.dtype()), const_cast<void*>(bias->data()));
      t_bias_ = bias_cache_.get(const_cast<void*>(bias->data()));
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    c_cache_.release();
    a_cache_.release();
    as_cache_.release();
    b_cache_.release();
    bs_cache_.release();
    bias_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor c, const Tensor a, const Tensor as, const Tensor b,
                  const Tensor bs,
                  const std::optional<Tensor> bias) const override {
    if (!supports(c, a, as, b, bs, bias) || bias.has_value() != has_bias_) {
      throw std::invalid_argument("unsupported Ascend I8Gemm operands");
    }

    auto stream = static_cast<aclrtStream>(stream_);

    if (!executor_) {
      auto query = weight_nz_ ? &aclnnQuantMatmulWeightNzGetWorkspaceSize
          : &aclnnQuantMatmulV5GetWorkspaceSize;
      ascend::CheckAcl(
          query(
              t_a_, t_b_, t_as_, t_bs_, nullptr, nullptr, nullptr, nullptr,
              t_bias_, false, transpose_x2_, 0, t_c_, &workspace_size_, &executor_),
          "I8Gemm workspace query");
      ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                       "I8Gemm repeatable executor");
    } else {
      ascend::CheckAcl(AclSetInputTensorAddr(
                           executor_, 0, t_a_, const_cast<void*>(a.data())),
                       "I8Gemm activation address");
      ascend::CheckAcl(AclSetInputTensorAddr(
                           executor_, 1, t_b_, const_cast<void*>(b.data())),
                       "I8Gemm weight address");
      ascend::CheckAcl(AclSetInputTensorAddr(
                           executor_, 2, t_as_, const_cast<void*>(as.data())),
                       "I8Gemm activation scale address");
      ascend::CheckAcl(AclSetInputTensorAddr(
                           executor_, 3, t_bs_, const_cast<void*>(bs.data())),
                       "I8Gemm weight scale address");
      if (has_bias_) {
        ascend::CheckAcl(
            AclSetInputTensorAddr(executor_, 4, t_bias_,
                                  const_cast<void*>(bias->data())),
            "I8Gemm bias address");
      }
      ascend::CheckAcl(AclSetOutputTensorAddr(executor_, 0, t_c_, c.data()),
                       "I8Gemm output address");
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size_,
                                                     "i8_gemm");
    ascend::CheckAcl(
        aclnnQuantMatmulV5(arena.buf, workspace_size_, executor_, stream),
        "I8Gemm execution");
  }

 private:
  mutable ascend::AclTensorCache c_cache_;
  mutable ascend::AclTensorCache a_cache_;
  mutable ascend::AclTensorCache as_cache_;
  mutable ascend::AclTensorCache b_cache_;
  mutable ascend::AclTensorCache bs_cache_;
  mutable ascend::AclTensorCache bias_cache_;
  bool weight_nz_{false};
  bool has_bias_{false};
  bool transpose_x2_{true};
  aclTensor* t_c_{nullptr};
  aclTensor* t_a_{nullptr};
  aclTensor* t_as_{nullptr};
  aclTensor* t_b_{nullptr};
  aclTensor* t_bs_{nullptr};
  aclTensor* t_bias_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  mutable uint64_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
