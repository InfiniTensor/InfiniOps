#ifndef INFINI_OPS_ASCEND_CUMSUM_KERNEL_H_
#define INFINI_OPS_ASCEND_CUMSUM_KERNEL_H_

#include <optional>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cumsum.h"
#include "base/cumsum.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Cumsum, Device::Type::kAscend> : public Cumsum {
 public:
  Operator(const Tensor input, const int64_t dim,
           const std::optional<DataType> dtype, Tensor out)
      : Cumsum(input, dim, dtype, out), input_cache_(input),
        output_cache_(out), dim_(dim),
        output_dtype_(ascend::ToAclDtype(dtype.value_or(out.dtype()))) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    input_cache_.release();
    output_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const int64_t dim,
                  const std::optional<DataType> dtype,
                  Tensor output) const override {
    auto *t_input = input_cache_.get(const_cast<void *>(input.data()));
    auto *t_output = output_cache_.get(output.data());
    if (!executor_) {
      ascend::CheckAcl(
          aclnnCumsumGetWorkspaceSize(
              t_input, dim_, output_dtype_, t_output, &workspace_size_,
              &executor_),
          "Cumsum workspace query");
      ascend::CheckAcl(
          aclSetAclOpExecutorRepeatable(executor_),
          "Cumsum repeatable executor");
    } else {
      ascend::CheckAcl(
          aclSetInputTensorAddr(
              executor_, 0, t_input, const_cast<void *>(input.data())),
          "Cumsum input address update");
      ascend::CheckAcl(
          aclSetOutputTensorAddr(
              executor_, 0, t_output, output.data()),
          "Cumsum output address update");
    }
    auto stream = static_cast<aclrtStream>(stream_);
    auto &arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "cumsum");
    ascend::CheckAcl(
        aclnnCumsum(arena.buf, workspace_size_, executor_, stream),
        "Cumsum execution");
  }

 private:
  mutable ascend::AclTensorCache input_cache_;
  mutable ascend::AclTensorCache output_cache_;
  int64_t dim_{};
  aclDataType output_dtype_{};
  mutable aclOpExecutor *executor_{nullptr};
  mutable uint64_t workspace_size_{0};
};

} // namespace infini::ops

#endif
