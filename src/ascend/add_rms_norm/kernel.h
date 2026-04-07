#ifndef INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add_rms_norm.h"
#include "ascend/device.h"
#include "base/add_rms_norm.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kAscend> : public AddRmsNorm {
 public:
  Operator(const Tensor x1, const Tensor x2, const Tensor gamma, float eps,
           Tensor y_out, Tensor x_out)
      : AddRmsNorm(x1, x2, gamma, eps, y_out, x_out) {
    // aclnnAddRmsNorm writes rstd as a required side output.
    // Allocate a persistent device buffer for it.
    size_t rstd_bytes = batch_size_ * nhead_ * sizeof(float);
    aclrtMalloc(&rstd_data_, rstd_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
  }

  ~Operator() {
    if (rstd_data_) aclrtFree(rstd_data_);
  }

  void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                  float eps, Tensor y_out, Tensor x_out) const override {
    auto t_x1    = ascend::buildAclTensor(x1);
    auto t_x2    = ascend::buildAclTensor(x2);
    auto t_gamma = ascend::buildAclTensor(gamma);
    auto t_y_out = ascend::buildAclTensor(y_out);
    auto t_x_out = ascend::buildAclTensor(x_out);
    // rstd is always float32 regardless of input dtype.
    auto t_rstd  = aclCreateTensor(
        rstd_shape_.data(), 2, ACL_FLOAT,
        /*strides=*/nullptr, 0, ACL_FORMAT_ND,
        rstd_shape_.data(), 2, rstd_data_);
    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddRmsNormGetWorkspaceSize(t_x1, t_x2, t_gamma, eps, t_y_out, t_rstd,
                                   t_x_out, &ws_needed, &executor);
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclnnAddRmsNorm(arena.buf, ws_needed, executor, stream);
    aclDestroyTensor(t_x1);
    aclDestroyTensor(t_x2);
    aclDestroyTensor(t_gamma);
    aclDestroyTensor(t_y_out);
    aclDestroyTensor(t_rstd);
    aclDestroyTensor(t_x_out);
  }

 private:
  void* rstd_data_ = nullptr;
};

}  // namespace infini::ops

#endif
