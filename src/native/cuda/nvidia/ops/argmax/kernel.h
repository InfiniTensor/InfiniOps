#ifndef INFINI_OPS_NVIDIA_ARGMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_ARGMAX_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "base/argmax.h"

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kNvidia> : public Argmax {
 public:
  Operator(const Tensor input, const std::optional<int64_t> dim,
           const bool keepdim, Tensor out);

  ~Operator() override;

  std::size_t workspace_size_in_bytes() const override;

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override;

 private:
  static std::size_t DispatchWorkspaceSize(DataType dtype,
                                           std::size_t numel);

  std::size_t numel_{0};

  std::size_t workspace_size_{0};

  void* default_workspace_{nullptr};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_ARGMAX_KERNEL_H_
