#ifndef INFINI_OPS_CAMBRICON_LOG_KERNEL_H_
#define INFINI_OPS_CAMBRICON_LOG_KERNEL_H_

#include "base/log.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

inline cnnlStatus_t NaturalLog(cnnlHandle_t handle, cnnlTensorDescriptor_t input_desc,
                               const void* input, cnnlTensorDescriptor_t out_desc, void* out) {
  return cnnlLog_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, CNNL_LOG_E,
                    input_desc, input, out_desc, out);
}

template <>
class Operator<Log, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Log, NaturalLog> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
