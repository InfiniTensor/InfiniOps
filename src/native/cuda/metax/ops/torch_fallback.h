#ifndef INFINI_OPS_METAX_TORCH_FALLBACK_H_
#define INFINI_OPS_METAX_TORCH_FALLBACK_H_

#ifdef WITH_TORCH

#include <optional>
#include <vector>

#include "tensor.h"
#include "torch/tensor_.h"

namespace infini::ops::metax_torch_fallback {

inline at::Tensor ToAten(const Tensor& tensor) {
  return ToAtenTensor<Device::Type::kMetax>(tensor);
}

inline std::optional<at::Tensor> ToAten(const std::optional<Tensor>& tensor) {
  if (!tensor.has_value()) {
    return std::nullopt;
  }

  return ToAten(*tensor);
}

inline std::vector<at::Tensor> ToAten(const std::vector<Tensor>& tensors) {
  std::vector<at::Tensor> result;
  result.reserve(tensors.size());

  for (const auto& tensor : tensors) {
    result.push_back(ToAten(tensor));
  }

  return result;
}

inline void CopyToOutput(at::Tensor out, at::Tensor result) {
  if (result.scalar_type() != out.scalar_type()) {
    result = result.to(out.scalar_type());
  }

  if (result.device() != out.device()) {
    result = result.to(out.device());
  }

  out.copy_(result);
}

}  // namespace infini::ops::metax_torch_fallback

#endif

#endif
