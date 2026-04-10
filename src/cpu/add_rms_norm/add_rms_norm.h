#ifndef INFINI_OPS_CPU_ADD_RMS_NORM_ADD_RMS_NORM_H_
#define INFINI_OPS_CPU_ADD_RMS_NORM_ADD_RMS_NORM_H_

#include "base/add_rms_norm.h"
#include "cpu/add/add.h"
#include "cpu/rms_norm/rms_norm.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kCpu> : public AddRmsNorm {
 public:
  Operator(const Tensor x1, const Tensor x2, const Tensor gamma, float eps,
           Tensor y_out, Tensor x_out)
      : AddRmsNorm(x1, x2, gamma, eps, y_out, x_out),
        add_(x1, x2, x_out),
        rms_norm_(x_out, gamma, eps, y_out) {}

  void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                  float eps, Tensor y_out, Tensor x_out) const override {
    add_.set_handle(handle_);
    add_.set_config(config_);
    add_.set_stream(stream_);
    add_.set_workspace(workspace_);
    add_.set_workspace_size_in_bytes(workspace_size_in_bytes_);
    add_(x1, x2, x_out);

    rms_norm_.set_handle(handle_);
    rms_norm_.set_config(config_);
    rms_norm_.set_stream(stream_);
    rms_norm_.set_workspace(workspace_);
    rms_norm_.set_workspace_size_in_bytes(workspace_size_in_bytes_);
    rms_norm_(x_out, gamma, eps, y_out);
  }

 private:
  mutable Operator<Add, Device::Type::kCpu> add_;

  mutable Operator<RmsNorm, Device::Type::kCpu> rms_norm_;
};

}  // namespace infini::ops

#endif
