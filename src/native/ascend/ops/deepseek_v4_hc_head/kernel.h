#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_HC_HEAD_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_HC_HEAD_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "base/deepseek_v4_hc_head.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// Implements the checkpoint HC head exactly in FP32:
// sigmoid(linear(rms_norm(flatten(x))) * scale + base) + eps,
// followed by a weighted sum over the four residual streams.
template <>
class Operator<DeepseekV4HcHead, Device::Type::kAscend, 0>
    : public DeepseekV4HcHead {
 public:
  static bool supports(Tensor out, const Tensor x, const Tensor hc_fn,
                       const Tensor hc_scale, const Tensor hc_base,
                       const Tensor rms_weight, double eps) {
    if (x.device().type() != Device::Type::kAscend ||
        x.dtype() != DataType::kBFloat16 || x.ndim() != 3 ||
        !x.IsContiguous() || x.shape()[1] != 4 ||
        out.device() != x.device() || out.dtype() != DataType::kBFloat16 ||
        out.ndim() != 2 || !out.IsContiguous() ||
        hc_fn.device() != x.device() || hc_fn.dtype() != DataType::kFloat32 ||
        hc_fn.ndim() != 2 || !hc_fn.IsContiguous() ||
        hc_scale.device() != x.device() ||
        hc_scale.dtype() != DataType::kFloat32 || hc_scale.numel() != 1 ||
        !hc_scale.IsContiguous() || hc_base.device() != x.device() ||
        hc_base.dtype() != DataType::kFloat32 || hc_base.ndim() != 1 ||
        !hc_base.IsContiguous() || rms_weight.device() != x.device() ||
        rms_weight.dtype() != DataType::kFloat32 ||
        rms_weight.ndim() != 1 || !rms_weight.IsContiguous() || eps <= 0.0) {
      return false;
    }
    const auto rows = x.shape()[0];
    const auto streams = x.shape()[1];
    const auto hidden = x.shape()[2];
    return out.shape() == Tensor::Shape{rows, hidden} &&
           hc_fn.shape() == Tensor::Shape{streams, streams * hidden} &&
           hc_base.shape() == Tensor::Shape{streams} &&
           rms_weight.shape() == Tensor::Shape{streams * hidden};
  }

  Operator(Tensor out, const Tensor x, const Tensor hc_fn,
           const Tensor hc_scale, const Tensor hc_base,
           const Tensor rms_weight, double eps)
      : DeepseekV4HcHead(out, x, hc_fn, hc_scale, hc_base, rms_weight, eps),
        out_cache_(out),
        hc_scale_cache_(hc_scale),
        hc_base_cache_(hc_base),
        rms_weight_cache_(rms_weight),
        rows_(x.shape()[0]),
        streams_(x.shape()[1]),
        hidden_(x.shape()[2]),
        flat_(streams_ * hidden_),
        eps_storage_(static_cast<float>(eps)) {
    if (!supports(out, x, hc_fn, hc_scale, hc_base, rms_weight, eps)) {
      throw std::invalid_argument("unsupported Ascend DeepseekV4HcHead operands");
    }

    Tensor x_flat{const_cast<void*>(x.data()), {rows_, flat_}, x.dtype(),
                  x.device()};
    Tensor fn_t{const_cast<void*>(hc_fn.data()), {flat_, streams_},
                hc_fn.dtype(), hc_fn.device(), {1, flat_}};
    x_flat_cache_ = ascend::AclTensorCache(x_flat);
    fn_t_cache_ = ascend::AclTensorCache(fn_t);
    x_f32_cache_ = ascend::AclTensorCache({rows_, flat_}, ACL_FLOAT, nullptr);
    normalized_cache_ =
        ascend::AclTensorCache({rows_, flat_}, ACL_FLOAT, nullptr);
    rstd_cache_ = ascend::AclTensorCache({rows_}, ACL_FLOAT, nullptr);
    mixes_cache_ = ascend::AclTensorCache({rows_, streams_}, ACL_FLOAT, nullptr);
    scaled_cache_ = ascend::AclTensorCache({rows_, streams_}, ACL_FLOAT, nullptr);
    biased_cache_ = ascend::AclTensorCache({rows_, streams_}, ACL_FLOAT, nullptr);
    gates_cache_ = ascend::AclTensorCache({rows_, streams_}, ACL_FLOAT, nullptr);
    gates_eps_cache_ =
        ascend::AclTensorCache({rows_, streams_}, ACL_FLOAT, nullptr);
    gates_batch_cache_ =
        ascend::AclTensorCache({rows_, 1, streams_}, ACL_FLOAT, nullptr);
    x_batch_cache_ =
        ascend::AclTensorCache({rows_, streams_, hidden_}, ACL_FLOAT, nullptr);
    summed_cache_ = ascend::AclTensorCache({rows_, hidden_}, ACL_FLOAT, nullptr);
    summed_batch_cache_ =
        ascend::AclTensorCache({rows_, 1, hidden_}, ACL_FLOAT, nullptr);

    reserve_scratch(x_f32_offset_, rows_ * flat_ * sizeof(float));
    reserve_scratch(normalized_offset_, rows_ * flat_ * sizeof(float));
    reserve_scratch(rstd_offset_, rows_ * sizeof(float));
    reserve_scratch(mixes_offset_, rows_ * streams_ * sizeof(float));
    reserve_scratch(scaled_offset_, rows_ * streams_ * sizeof(float));
    reserve_scratch(biased_offset_, rows_ * streams_ * sizeof(float));
    reserve_scratch(gates_offset_, rows_ * streams_ * sizeof(float));
    reserve_scratch(gates_eps_offset_, rows_ * streams_ * sizeof(float));
    reserve_scratch(summed_offset_, rows_ * hidden_ * sizeof(float));
    scratch_size_ = align64(scratch_size_);

    one_storage_ = 1.0f;
    one_scalar_ = aclCreateScalar(&one_storage_, ACL_FLOAT);
    eps_scalar_ = aclCreateScalar(&eps_storage_, ACL_FLOAT);
    if (!one_scalar_ || !eps_scalar_) {
      throw std::runtime_error("DeepseekV4HcHead scalar creation failed");
    }

    auto* t_x = x_flat_cache_.get(const_cast<void*>(x.data()));
    auto* t_fn = fn_t_cache_.get(const_cast<void*>(hc_fn.data()));
    auto* t_scale = hc_scale_cache_.get(const_cast<void*>(hc_scale.data()));
    auto* t_base = hc_base_cache_.get(const_cast<void*>(hc_base.data()));
    auto* t_gamma = rms_weight_cache_.get(const_cast<void*>(rms_weight.data()));
    auto* t_out = out_cache_.get(out.data());
    auto* t_x_f32 = x_f32_cache_.get(nullptr);
    auto* t_normalized = normalized_cache_.get(nullptr);
    auto* t_rstd = rstd_cache_.get(nullptr);
    auto* t_mixes = mixes_cache_.get(nullptr);
    auto* t_scaled = scaled_cache_.get(nullptr);
    auto* t_biased = biased_cache_.get(nullptr);
    auto* t_gates = gates_cache_.get(nullptr);
    auto* t_gates_eps = gates_eps_cache_.get(nullptr);
    auto* t_gates_batch = gates_batch_cache_.get(nullptr);
    auto* t_x_batch = x_batch_cache_.get(nullptr);
    auto* t_summed = summed_cache_.get(nullptr);
    auto* t_summed_batch = summed_batch_cache_.get(nullptr);

    check_workspace(aclnnCastGetWorkspaceSize(
                        t_x, ACL_FLOAT, t_x_f32, &cast_in_ws_, &cast_in_exec_),
                    cast_in_exec_, cast_in_ws_, "input cast");
    check_workspace(aclnnRmsNormGetWorkspaceSize(
                        t_x_f32, t_gamma, eps, t_normalized, t_rstd,
                        &rms_ws_, &rms_exec_),
                    rms_exec_, rms_ws_, "rms norm");
    check_workspace(aclnnMatmulGetWorkspaceSize(
                        t_normalized, t_fn, t_mixes, 1, &linear_ws_,
                        &linear_exec_),
                    linear_exec_, linear_ws_, "gate linear");
    check_workspace(aclnnMulGetWorkspaceSize(
                        t_mixes, t_scale, t_scaled, &scale_ws_, &scale_exec_),
                    scale_exec_, scale_ws_, "gate scale");
    check_workspace(aclnnAddGetWorkspaceSize(
                        t_scaled, t_base, one_scalar_, t_biased, &bias_ws_,
                        &bias_exec_),
                    bias_exec_, bias_ws_, "gate bias");
    check_workspace(aclnnSigmoidGetWorkspaceSize(
                        t_biased, t_gates, &sigmoid_ws_, &sigmoid_exec_),
                    sigmoid_exec_, sigmoid_ws_, "gate sigmoid");
    check_workspace(aclnnAddsGetWorkspaceSize(
                        t_gates, eps_scalar_, one_scalar_, t_gates_eps,
                        &eps_ws_, &eps_exec_),
                    eps_exec_, eps_ws_, "gate epsilon");
    check_workspace(aclnnMatmulGetWorkspaceSize(
                        t_gates_batch, t_x_batch, t_summed_batch, 1,
                        &combine_ws_, &combine_exec_),
                    combine_exec_, combine_ws_, "stream combine");
    check_workspace(aclnnCastGetWorkspaceSize(
                        t_summed, ACL_BF16, t_out, &cast_out_ws_,
                        &cast_out_exec_),
                    cast_out_exec_, cast_out_ws_, "output cast");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    x_flat_cache_.release();
    fn_t_cache_.release();
    out_cache_.release();
    hc_scale_cache_.release();
    hc_base_cache_.release();
    rms_weight_cache_.release();
    x_f32_cache_.release();
    normalized_cache_.release();
    rstd_cache_.release();
    mixes_cache_.release();
    scaled_cache_.release();
    biased_cache_.release();
    gates_cache_.release();
    gates_eps_cache_.release();
    gates_batch_cache_.release();
    x_batch_cache_.release();
    summed_cache_.release();
    summed_batch_cache_.release();
    if (one_scalar_) aclDestroyScalar(one_scalar_);
    if (eps_scalar_) aclDestroyScalar(eps_scalar_);
  }

  bool capture_safe() const override { return false; }

  void operator()(Tensor out, const Tensor x, const Tensor hc_fn,
                  const Tensor hc_scale, const Tensor hc_base,
                  const Tensor rms_weight, double) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, scratch_size_ + max_workspace_, "deepseek_v4_hc_head");
    auto* base = static_cast<uint8_t*>(arena.buf);
    auto* workspace = base + scratch_size_;
    auto* x_f32 = base + x_f32_offset_;
    auto* normalized = base + normalized_offset_;
    auto* rstd = base + rstd_offset_;
    auto* mixes = base + mixes_offset_;
    auto* scaled = base + scaled_offset_;
    auto* biased = base + biased_offset_;
    auto* gates = base + gates_offset_;
    auto* gates_eps = base + gates_eps_offset_;
    auto* summed = base + summed_offset_;

    auto* t_x = x_flat_cache_.get(const_cast<void*>(x.data()));
    auto* t_fn = fn_t_cache_.get(const_cast<void*>(hc_fn.data()));
    auto* t_scale = hc_scale_cache_.get(const_cast<void*>(hc_scale.data()));
    auto* t_base = hc_base_cache_.get(const_cast<void*>(hc_base.data()));
    auto* t_gamma = rms_weight_cache_.get(const_cast<void*>(rms_weight.data()));
    auto* t_out = out_cache_.get(out.data());
    auto* t_x_f32 = x_f32_cache_.get(x_f32);
    auto* t_normalized = normalized_cache_.get(normalized);
    auto* t_rstd = rstd_cache_.get(rstd);
    auto* t_mixes = mixes_cache_.get(mixes);
    auto* t_scaled = scaled_cache_.get(scaled);
    auto* t_biased = biased_cache_.get(biased);
    auto* t_gates = gates_cache_.get(gates);
    auto* t_gates_eps = gates_eps_cache_.get(gates_eps);
    auto* t_gates_batch = gates_batch_cache_.get(gates_eps);
    auto* t_x_batch = x_batch_cache_.get(x_f32);
    auto* t_summed = summed_cache_.get(summed);
    auto* t_summed_batch = summed_batch_cache_.get(summed);

    bind(cast_in_exec_, {{0, t_x, const_cast<void*>(x.data())}},
         {{0, t_x_f32, x_f32}});
    run(aclnnCast, cast_in_exec_, cast_in_ws_, workspace, stream, "input cast");

    bind(rms_exec_, {{0, t_x_f32, x_f32},
                     {1, t_gamma, const_cast<void*>(rms_weight.data())}},
         {{0, t_normalized, normalized}, {1, t_rstd, rstd}});
    run(aclnnRmsNorm, rms_exec_, rms_ws_, workspace, stream, "rms norm");

    bind(linear_exec_, {{0, t_normalized, normalized},
                        {1, t_fn, const_cast<void*>(hc_fn.data())}},
         {{0, t_mixes, mixes}});
    run(aclnnMatmul, linear_exec_, linear_ws_, workspace, stream, "gate linear");

    bind(scale_exec_, {{0, t_mixes, mixes},
                       {1, t_scale, const_cast<void*>(hc_scale.data())}},
         {{0, t_scaled, scaled}});
    run(aclnnMul, scale_exec_, scale_ws_, workspace, stream, "gate scale");

    bind(bias_exec_, {{0, t_scaled, scaled},
                      {1, t_base, const_cast<void*>(hc_base.data())}},
         {{0, t_biased, biased}});
    run(aclnnAdd, bias_exec_, bias_ws_, workspace, stream, "gate bias");

    bind(sigmoid_exec_, {{0, t_biased, biased}}, {{0, t_gates, gates}});
    run(aclnnSigmoid, sigmoid_exec_, sigmoid_ws_, workspace, stream,
        "gate sigmoid");

    bind(eps_exec_, {{0, t_gates, gates}}, {{0, t_gates_eps, gates_eps}});
    run(aclnnAdds, eps_exec_, eps_ws_, workspace, stream, "gate epsilon");

    bind(combine_exec_, {{0, t_gates_batch, gates_eps},
                         {1, t_x_batch, x_f32}},
         {{0, t_summed_batch, summed}});
    run(aclnnMatmul, combine_exec_, combine_ws_, workspace, stream,
        "stream combine");

    bind(cast_out_exec_, {{0, t_summed, summed}}, {{0, t_out, out.data()}});
    run(aclnnCast, cast_out_exec_, cast_out_ws_, workspace, stream,
        "output cast");
  }

 private:
  struct Binding {
    int index;
    aclTensor* tensor;
    void* data;
  };

  static uint64_t align64(uint64_t value) { return (value + 63) & ~uint64_t{63}; }

  void reserve_scratch(uint64_t& offset, uint64_t bytes) {
    scratch_size_ = align64(scratch_size_);
    offset = scratch_size_;
    scratch_size_ += bytes;
  }

  void check_workspace(aclnnStatus status, aclOpExecutor* executor,
                       uint64_t workspace, const char* stage) {
    const auto workspace_message =
        std::string("DeepseekV4HcHead ") + stage + " workspace query";
    ascend::CheckAcl(status, workspace_message.c_str());
    const auto repeatable_message =
        std::string("DeepseekV4HcHead ") + stage + " repeatable executor";
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor),
                     repeatable_message.c_str());
    max_workspace_ = std::max(max_workspace_, workspace);
  }

  static void bind(aclOpExecutor* executor,
                   std::initializer_list<Binding> inputs,
                   std::initializer_list<Binding> outputs) {
    for (const auto& binding : inputs) {
      ascend::CheckAcl(aclSetInputTensorAddr(
                           executor, binding.index, binding.tensor,
                           binding.data),
                       "DeepseekV4HcHead input address");
    }
    for (const auto& binding : outputs) {
      ascend::CheckAcl(aclSetOutputTensorAddr(
                           executor, binding.index, binding.tensor,
                           binding.data),
                       "DeepseekV4HcHead output address");
    }
  }

  template <typename Fn>
  static void run(Fn fn, aclOpExecutor* executor, uint64_t workspace_size,
                  void* workspace, aclrtStream stream, const char* stage) {
    const auto message =
        std::string("DeepseekV4HcHead ") + stage + " execution";
    ascend::CheckAcl(fn(workspace, workspace_size, executor, stream),
                     message.c_str());
  }

  mutable ascend::AclTensorCache x_flat_cache_;
  mutable ascend::AclTensorCache fn_t_cache_;
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache hc_scale_cache_;
  mutable ascend::AclTensorCache hc_base_cache_;
  mutable ascend::AclTensorCache rms_weight_cache_;
  mutable ascend::AclTensorCache x_f32_cache_;
  mutable ascend::AclTensorCache normalized_cache_;
  mutable ascend::AclTensorCache rstd_cache_;
  mutable ascend::AclTensorCache mixes_cache_;
  mutable ascend::AclTensorCache scaled_cache_;
  mutable ascend::AclTensorCache biased_cache_;
  mutable ascend::AclTensorCache gates_cache_;
  mutable ascend::AclTensorCache gates_eps_cache_;
  mutable ascend::AclTensorCache gates_batch_cache_;
  mutable ascend::AclTensorCache x_batch_cache_;
  mutable ascend::AclTensorCache summed_cache_;
  mutable ascend::AclTensorCache summed_batch_cache_;

  int64_t rows_{0};
  int64_t streams_{0};
  int64_t hidden_{0};
  int64_t flat_{0};
  uint64_t scratch_size_{0};
  uint64_t max_workspace_{0};
  uint64_t x_f32_offset_{0};
  uint64_t normalized_offset_{0};
  uint64_t rstd_offset_{0};
  uint64_t mixes_offset_{0};
  uint64_t scaled_offset_{0};
  uint64_t biased_offset_{0};
  uint64_t gates_offset_{0};
  uint64_t gates_eps_offset_{0};
  uint64_t summed_offset_{0};

  float one_storage_{1.0f};
  float eps_storage_{0.0f};
  aclScalar* one_scalar_{nullptr};
  aclScalar* eps_scalar_{nullptr};

  mutable aclOpExecutor* cast_in_exec_{nullptr};
  mutable aclOpExecutor* rms_exec_{nullptr};
  mutable aclOpExecutor* linear_exec_{nullptr};
  mutable aclOpExecutor* scale_exec_{nullptr};
  mutable aclOpExecutor* bias_exec_{nullptr};
  mutable aclOpExecutor* sigmoid_exec_{nullptr};
  mutable aclOpExecutor* eps_exec_{nullptr};
  mutable aclOpExecutor* combine_exec_{nullptr};
  mutable aclOpExecutor* cast_out_exec_{nullptr};
  uint64_t cast_in_ws_{0};
  uint64_t rms_ws_{0};
  uint64_t linear_ws_{0};
  uint64_t scale_ws_{0};
  uint64_t bias_ws_{0};
  uint64_t sigmoid_ws_{0};
  uint64_t eps_ws_{0};
  uint64_t combine_ws_{0};
  uint64_t cast_out_ws_{0};
};

}  // namespace infini::ops

#endif
