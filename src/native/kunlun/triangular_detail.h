#ifndef INFINI_OPS_KUNLUN_TRIANGULAR_DETAIL_H_
#define INFINI_OPS_KUNLUN_TRIANGULAR_DETAIL_H_

#include <stdexcept>

#include "base/tril.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

namespace detail {
// Shared kunlun triangular launch (used by both Triu and Tril wrappers).
inline void KunlunLaunchTriangular(bool upper, const Tensor input, Tensor out,
                                   int64_t diagonal, Tensor::Size rows,
                                   void* stream) {
  if (out.numel() == 0) {
    return;
  }
  if (input.ndim() < 2 || input.ndim() > 8) {
    throw std::runtime_error("kunlun triangular: rank must be 2..8");
  }
  const int element_size = input.element_size();
  if (element_size != 2 && element_size != 4) {
    throw std::runtime_error("kunlun triangular: unsupported element size");
  }
  long long shape[8];
  long long in_strides[8];
  long long out_strides[8];
  for (std::size_t i = 0; i < input.ndim(); ++i) {
    shape[i] = (long long)input.shape()[i];
    in_strides[i] = (long long)input.strides()[i];
    out_strides[i] = (long long)out.strides()[i];
  }
  kunlun_launch::LaunchTriangular(
      upper, element_size, out.data(), input.data(), (int)input.ndim(), shape,
      in_strides, out_strides, (int)rows, (unsigned long long)out.numel(),
      (int)diagonal, static_cast<XPUStream>(stream));
}
}  // namespace detail

}  // namespace infini::ops

#endif  // INFINI_OPS_KUNLUN_TRIANGULAR_DETAIL_H_
