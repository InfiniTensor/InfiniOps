#ifndef INFINI_OPS_CUDA_ADD_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_KERNEL_H_

#include "base/add.h"
#include "cuda/add/kernel.cuh"
#include "cuda/templates/binary_elementwise.cuh"

namespace infini::ops {

// CudaAdd uses BinaryElementwiseBrick for automatic vectorized dispatch
// on contiguous tensors (128-bit coalesced load/store).
template <typename Backend>
class CudaAdd : public Add {
 public:
  CudaAdd(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, out},
        brick_{input, other, out, ndim_} {}

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    brick_.template Run<AllTypes, AddOp>(
        stream_, input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        out_type_);
  }

 private:
  BinaryElementwiseBrick<Backend> brick_;
};

}  // namespace infini::ops

#endif
