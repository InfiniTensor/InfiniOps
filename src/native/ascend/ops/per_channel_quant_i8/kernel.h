#ifndef INFINI_OPS_ASCEND_PER_CHANNEL_QUANT_I8_KERNEL_H_
#define INFINI_OPS_ASCEND_PER_CHANNEL_QUANT_I8_KERNEL_H_

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "aclrtlaunch_per_token_quant_i8.h"
#include "base/per_channel_quant_i8.h"
#include "native/ascend/error_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<PerChannelQuantI8, Device::Type::kAscend, 0>
    : public PerChannelQuantI8 {
 public:
  static bool supports(const Tensor x, Tensor packed, Tensor scale) {
    return x.device().type() == Device::Type::kAscend &&
           x.device() == packed.device() && x.device() == scale.device() &&
           x.ndim() == 2 && x.shape() == packed.shape() &&
           scale.shape() == Tensor::Shape{x.shape()[0], 1} &&
           (x.dtype() == DataType::kFloat16 ||
            x.dtype() == DataType::kBFloat16) &&
           packed.dtype() == DataType::kInt8 &&
           scale.dtype() == DataType::kFloat32 && x.IsContiguous() &&
           packed.IsContiguous() && scale.IsContiguous() && x.shape()[0] > 0 &&
           x.shape()[1] > 0 && x.shape()[1] % 32 == 0;
  }

  Operator(const Tensor x, Tensor packed, Tensor scale)
      : PerChannelQuantI8(x, packed, scale),
        rows_(static_cast<int64_t>(x.shape()[0])),
        cols_(static_cast<int64_t>(x.shape()[1])),
        dtype_code_(x.dtype() == DataType::kFloat16 ? 0 : 1) {
    if (!supports(x, packed, scale)) {
      throw std::invalid_argument(
          "unsupported Ascend per-token INT8 operands");
    }

    int32_t device_id = 0;
    int64_t vector_cores = 1;
    ascend::CheckAcl(aclrtGetDevice(&device_id),
                     "PerChannelQuantI8 device query");
    ascend::CheckAcl(
        aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
                           ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_cores),
        "PerChannelQuantI8 vector core query");
    const int64_t max_scale_blocks = (rows_ + 7) / 8;
    block_dim_ = static_cast<uint32_t>(std::min<int64_t>(
        max_scale_blocks, std::max<int64_t>(1, vector_cores)));
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor x, Tensor packed, Tensor scale) const override {
    if (!supports(x, packed, scale)) {
      throw std::invalid_argument(
          "unsupported Ascend per-token INT8 operands");
    }

    auto stream = static_cast<aclrtStream>(stream_);
    ascend::CheckAcl(
        aclrtlaunch_per_token_quant_i8(
            block_dim_, stream, const_cast<void*>(x.data()), packed.data(),
            scale.data(), rows_, cols_, dtype_code_),
        "PerChannelQuantI8 custom kernel launch");
  }

 private:
  int64_t rows_{0};
  int64_t cols_{0};
  int64_t dtype_code_{0};
  uint32_t block_dim_{1};
};

}  // namespace infini::ops

#endif
