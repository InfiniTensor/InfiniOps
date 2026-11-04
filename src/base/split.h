#ifndef INFINI_OPS_BASE_SPLIT_H_
#define INFINI_OPS_BASE_SPLIT_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "operator.h"

namespace infini::ops {

class Split : public Operator<Split> {
 public:
  Split(const Tensor input, const int64_t split_size, const int64_t dim,
        std::vector<Tensor> outs)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        split_size_{split_size},
        dim_{NormalizeDim(dim, input.ndim())},
        outs_{std::move(outs)},
        device_index_{outs_.empty() ? 0 : outs_.front().device().index()} {
    assert(split_size_ > 0 && "`Split` requires positive `split_size`");
    assert(!outs_.empty() && "`Split` requires at least one output tensor");
    const auto expected =
        (input.size(dim_) + split_size_ - 1) / split_size_;
    assert(static_cast<int64_t>(outs_.size()) == expected &&
           "`Split` output count must equal ceil(input_size(dim) / split_size)");

    for (std::size_t i = 0; i < outs_.size(); ++i) {
      assert(outs_[i].ndim() == input.ndim() && "`Split` output rank mismatch");
      assert(outs_[i].dtype() == input_type_ && "`Split` dtype mismatch");
      for (Tensor::Size d = 0; d < input.ndim(); ++d) {
        if (d != dim_) {
          assert(outs_[i].size(d) == input.size(d) && "`Split` output shape mismatch");
        }
      }
      const auto expect_dim = std::min<int64_t>(
          split_size_,
          static_cast<int64_t>(input.size(dim_)) -
              static_cast<int64_t>(i) * split_size_);
      assert(outs_[i].size(dim_) == expect_dim && "`Split` output dim size mismatch");

      out_shapes_.emplace_back(outs_[i].shape().begin(), outs_[i].shape().end());
      out_strides_.emplace_back(outs_[i].strides().begin(), outs_[i].strides().end());
      out_types_.emplace_back(outs_[i].dtype());
    }
  }

  virtual void operator()(const Tensor input, const int64_t split_size,
                          const int64_t dim,
                          std::vector<Tensor> outs) const = 0;

 protected:
  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    assert(ndim > 0 && "`Split` requires ndim > 0");
    if (ndim == 0) return 0;
    const auto d = dim % static_cast<int64_t>(ndim);
    return d < 0 ? d + static_cast<int64_t>(ndim) : d;
  }

  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;
  int64_t split_size_{0};
  int64_t dim_{0};
  std::vector<Tensor> outs_{};
  std::vector<Tensor::Shape> out_shapes_;
  std::vector<Tensor::Strides> out_strides_;
  std::vector<DataType> out_types_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SPLIT_H_
