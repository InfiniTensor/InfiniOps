#ifndef INFINI_OPS_TORCH_GEMM_H_
#define INFINI_OPS_TORCH_GEMM_H_

#include "base/gemm.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Gemm, kDev, 2> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c);

  using Gemm::operator();

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override;

 private:
  Tensor::Shape a_shape_;

  Tensor::Shape b_shape_;

  Tensor::Shape c_shape_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
