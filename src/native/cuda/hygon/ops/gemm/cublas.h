#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include "native/cuda/hygon/blas.h"
#include "native/cuda/hygon/ops/gemm/handle_owner.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kHygon, 0>
    : public BlasGemm<Blas<Device::Type::kHygon>> {
 public:
  template <typename... Tail>
  static bool supports(const Config& config, const Tensor a, const Tensor b,
                       const Tail&... tail) {
    static_assert(sizeof...(Tail) == 1 || sizeof...(Tail) == 6);
    const auto& y = std::get<sizeof...(Tail) - 1>(std::tie(tail...));
    return GemmConfig::SupportsMetadata(config, a, b, y);
  }

  Operator(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : BlasGemm<Blas<Device::Type::kHygon>>{a,    b,       c,       alpha,
                                             beta, trans_a, trans_b, y},
        handle_{HygonGemmHandle::Acquire(y.device().index())} {}

  Operator(const Tensor a, const Tensor b, Tensor y)
      : Operator{a,
                 b,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 y} {}

  bool capture_safe() const override { return true; }

 protected:
  cublasHandle_t GetHandle() const override { return handle_->get(); }

 private:
  std::shared_ptr<HygonGemmHandle> handle_;
};

}  // namespace infini::ops

#endif
