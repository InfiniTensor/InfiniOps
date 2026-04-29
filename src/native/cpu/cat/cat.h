#ifndef INFINI_OPS_CPU_CAT_CAT_H_
#define INFINI_OPS_CPU_CAT_CAT_H_

#include <cstring>
#include <vector>

#include "base/cat.h"
#include "cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kCpu> : public Cat {
 public:
  Operator(const Tensor first_input, std::vector<Tensor> rest_inputs,
           int64_t dim, Tensor out)
      : Cat{first_input, rest_inputs, dim, out} {}

  void operator()(const Tensor first_input, std::vector<Tensor> rest_inputs,
                  int64_t /*dim*/, Tensor out) const override {
    // Collect all input tensors.
    std::vector<const Tensor*> inputs;
    inputs.reserve(input_count_);
    inputs.push_back(&first_input);
    for (const auto& t : rest_inputs) {
      inputs.push_back(&t);
    }

    // Use normalized `dim_` from base class (handles negative dim).
    auto dim = dim_;
    auto elem_size = kDataTypeToSize.at(out.dtype());
    auto ndim = out.ndim();
    auto out_shape = out.shape();

    // Compute outer and inner sizes relative to the cat dimension.
    Tensor::Size outer = 1;
    for (int64_t i = 0; i < dim; ++i) {
      outer *= out_shape[i];
    }

    Tensor::Size inner = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < ndim; ++i) {
      inner *= out_shape[i];
    }

    auto* out_ptr = static_cast<char*>(out.data());
    Tensor::Size out_dim_size = out_shape[dim];

    // For each outer index, copy slices from each input along the cat dim.
    for (Tensor::Size o = 0; o < outer; ++o) {
      Tensor::Size offset_in_dim = 0;

      for (size_t t = 0; t < input_count_; ++t) {
        auto in_dim = inputs[t]->shape()[dim];
        auto in_ptr = static_cast<const char*>(inputs[t]->data());

        auto src_offset = (o * in_dim) * inner * elem_size;
        auto dst_offset =
            (o * out_dim_size + offset_in_dim) * inner * elem_size;
        auto copy_size = in_dim * inner * elem_size;

        std::memcpy(out_ptr + dst_offset, in_ptr + src_offset, copy_size);
        offset_in_dim += in_dim;
      }
    }
  }
};

}  // namespace infini::ops

#endif
