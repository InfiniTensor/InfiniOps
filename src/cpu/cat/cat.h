#ifndef INFINI_OPS_CPU_CAT_CAT_H_
#define INFINI_OPS_CPU_CAT_CAT_H_

#include <cstring>
#include <utility>

#include "base/cat.h"
#include "cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kCpu> : public Cat,
                                          Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor first_input, std::vector<Tensor> rest_inputs,
           int64_t dim, Tensor out)
      : Cat{first_input, std::move(rest_inputs), dim, out} {}

  void operator()(const Tensor first_input, std::vector<Tensor> rest_inputs,
                  int64_t dim, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllTypes>(
        dtype_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(out);
        },
        "`Operator<Cat, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(Tensor out) const {
    auto* out_ptr = static_cast<T*>(out.data());

    for (size_t outer = 0; outer < outer_size_; ++outer) {
      size_t out_offset = 0;

      for (size_t i = 0; i < input_count_; ++i) {
        const auto* in_ptr = static_cast<const T*>(inputs_[i].data());
        size_t dim_size = inputs_[i].size(dim_);
        size_t copy_count = dim_size * inner_size_;

        std::memcpy(
            out_ptr + outer * cum_dim_sizes_.back() * inner_size_ + out_offset,
            in_ptr + outer * dim_size * inner_size_,
            copy_count * sizeof(T));

        out_offset += copy_count;
      }
    }
  }
};

}  // namespace infini::ops

#endif
