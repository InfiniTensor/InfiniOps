#ifndef INFINI_OPS_ASCEND_GEMM_KERNEL_H_
#define INFINI_OPS_ASCEND_GEMM_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_baddbmm.h"
#include "ascend/device.h"
#include "base/gemm.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

// Build an aclTensor descriptor, optionally transposing the last two
// dimensions.  This is the same as `ascend::buildAclTensor` but swaps the
// trailing dims/strides when `transpose_last2` is true.
inline aclTensor* buildAclTensorTransposed(const Tensor& t,
                                           bool transpose_last2 = false) {
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
class Operator<Gemm, Device::Type::kAscend> : public Gemm {
 public:
    Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
             std::optional<float> beta, std::optional<int> trans_a,
             std::optional<int> trans_b, Tensor c)
        : Gemm(a, b, alpha, beta, trans_a, trans_b, c),
          batched_{batch_count_ > 1},
          alpha_val_{alpha.value_or(1.0f)},
          beta_val_{beta.value_or(1.0f)} {
        alpha_scalar_ = aclCreateScalar(&alpha_val_, ACL_FLOAT);
        beta_scalar_  = aclCreateScalar(&beta_val_, ACL_FLOAT);
    }

    ~Operator() {
        aclDestroyScalar(alpha_scalar_);
        aclDestroyScalar(beta_scalar_);
    }

    void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                    std::optional<float> beta, std::optional<int> trans_a,
                    std::optional<int> trans_b, Tensor c) const override {
        auto stream = static_cast<aclrtStream>(stream_);

        auto t_self = ascend::buildAclTensor(c);
        auto t_a    = detail::buildAclTensorTransposed(a, trans_a_);
        auto t_b    = detail::buildAclTensorTransposed(b, trans_b_);
        auto t_out  = ascend::buildAclTensor(c);

        uint64_t ws_needed = 0;
        aclOpExecutor* executor = nullptr;

        if (batched_) {
            aclnnBaddbmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                         alpha_scalar_, t_out, 0, &ws_needed,
                                         &executor);
        } else {
            aclnnAddmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                       alpha_scalar_, t_out, 0, &ws_needed,
                                       &executor);
        }

        auto& arena = ascend::workspacePool().ensure(stream, ws_needed);

        if (batched_) {
            aclnnBaddbmm(arena.buf, ws_needed, executor, stream);
        } else {
            aclnnAddmm(arena.buf, ws_needed, executor, stream);
        }

        aclDestroyTensor(t_self);
        aclDestroyTensor(t_a);
        aclDestroyTensor(t_b);
        aclDestroyTensor(t_out);
    }

 private:
    bool           batched_;
    float          alpha_val_;
    float          beta_val_;
    aclScalar*     alpha_scalar_  = nullptr;
    aclScalar*     beta_scalar_   = nullptr;
};

}  // namespace infini::ops

#endif
