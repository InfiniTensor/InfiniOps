#ifndef INFINI_OPS_ASCEND_MATMUL_KERNEL_H_
#define INFINI_OPS_ASCEND_MATMUL_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_matmul.h"
#include "ascend/device.h"
#include "base/matmul.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

// Build an aclTensor descriptor, optionally transposing the last two
// dimensions by swapping their shape and stride entries.  This avoids an
// explicit .contiguous() copy on the caller side.
inline aclTensor* buildAclTensorMaybeTransposed(const Tensor& t,
                                                bool transpose_last2) {
    std::vector<int64_t> shape(t.shape().begin(), t.shape().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());

    if (transpose_last2 && shape.size() >= 2) {
        auto n = shape.size();
        std::swap(shape[n - 2], shape[n - 1]);
        std::swap(strides[n - 2], strides[n - 1]);
    }

    int64_t storage_elems = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) { storage_elems = 0; break; }
        if (strides[i] > 0 && shape[i] > 1)
            storage_elems += static_cast<int64_t>(shape[i] - 1) * strides[i];
    }
    std::vector<int64_t> storage_shape = {storage_elems};

    return aclCreateTensor(
        shape.data(), static_cast<int64_t>(shape.size()),
        ascend::toAclDtype(t.dtype()),
        strides.data(), 0, ACL_FORMAT_ND,
        storage_shape.data(), static_cast<int64_t>(storage_shape.size()),
        const_cast<void*>(t.data()));
}

}  // namespace detail


template <>
class Operator<Matmul, Device::Type::kAscend> : public Matmul {
 public:
  Operator(const Tensor a, const Tensor b, Tensor c,
           bool trans_a, bool trans_b)
      : Matmul(a, b, c, trans_a, trans_b) {}

  void operator()(const Tensor a, const Tensor b, Tensor c,
                  bool trans_a, bool trans_b) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_a    = detail::buildAclTensorMaybeTransposed(a, trans_a);
    auto t_b    = detail::buildAclTensorMaybeTransposed(b, trans_b);
    auto t_out  = ascend::buildAclTensor(c);
    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    // cube_math_type = 1: allow fp16 accumulation.
    int8_t cube_math_type = 1;
    aclnnMatmulGetWorkspaceSize(t_a, t_b, t_out, cube_math_type, &ws_needed,
                                &executor);
    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclnnMatmul(arena.buf, ws_needed, executor, stream);
    aclDestroyTensor(t_a);
    aclDestroyTensor(t_b);
    aclDestroyTensor(t_out);
  }
};

}  // namespace infini::ops

#endif
