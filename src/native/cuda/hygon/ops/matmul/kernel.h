#ifndef INFINI_OPS_HYGON_MATMUL_KERNEL_H_
#define INFINI_OPS_HYGON_MATMUL_KERNEL_H_
#include <climits>
#include <memory>

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/ops/gemm/cublas.h"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/matmul/kernel.h"

namespace infini::ops {
template <>
class Operator<Matmul, Device::Type::kHygon>
    : public CudaMatmul<Runtime<Device::Type::kHygon>> {
  using Base = CudaMatmul<Runtime<Device::Type::kHygon>>;
  using Fast = Operator<Gemm, Device::Type::kHygon>;

 public:
  Operator(Tensor a, Tensor b, Tensor y) : Base(a, b, y) {
    const bool reduced =
        y.dtype() == DataType::kFloat16 || y.dtype() == DataType::kBFloat16;
    if (reduced && Matrix(a) && Matrix(b) && Matrix(y) && a.size(0) >= 16 &&
        a.size(1) >= 512 && b.size(1) >= 512 && a.size(1) == b.size(0) &&
        y.size(0) == a.size(0) && y.size(1) == b.size(1)) {
      fast_ = std::make_unique<Fast>(a, b, y);
    }
  }
  bool capture_safe() const override { return true; }
  void operator()(Tensor a, Tensor b, Tensor y) const override {
    if (fast_ && !Overlap(a, y) && !Overlap(b, y)) {
      fast_->set_stream(stream_);
      (*fast_)(a, b, y);
    } else {
      Base::operator()(a, b, y);
    }
  }

 private:
  static bool Matrix(const Tensor& t) {
    if (t.shape().size() != 2) return false;
    for (int i = 0; i < 2; ++i)
      if (!t.size(i) || t.size(i) > INT_MAX || t.stride(i) <= 0 ||
          t.stride(i) > INT_MAX)
        return false;
    // The existing BLAS provider selects row-major whenever the last stride
    // is one. A degenerate column-major view may have both strides equal to
    // one; keep its valid generic path rather than supplying an invalid lda.
    if (t.stride(1) == 1) return t.stride(0) >= t.size(1);
    return t.stride(0) == 1 && t.stride(1) >= t.size(0);
  }
  static bool Overlap(const Tensor& a, const Tensor& b) {
    const auto bytes = [](const Tensor& t) -> uintptr_t {
      return 2 * (1 + (t.size(0) - 1) * t.stride(0) +
                  (t.size(1) - 1) * t.stride(1));
    };
    const auto ap = reinterpret_cast<uintptr_t>(a.data());
    const auto bp = reinterpret_cast<uintptr_t>(b.data());
    return ap <= bp ? bp - ap < bytes(a) : ap - bp < bytes(b);
  }
  std::unique_ptr<Fast> fast_;
};
}  // namespace infini::ops

#endif
