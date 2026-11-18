#ifndef INFINI_OPS_KUNLUN_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
#define INFINI_OPS_KUNLUN_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_

#include <stdexcept>

#include "base/moe_align_block_size.h"

namespace infini::ops {

// Placeholder: `MoeAlignBlockSize` has no Kunlun implementation yet. The
// class must exist so downstream consumers (InfiniLM) that reference the
// operator type at compile time can build; running it throws.
template <>
class Operator<MoeAlignBlockSize, Device::Type::kKunlun>
    : public MoeAlignBlockSize {
 public:
  using MoeAlignBlockSize::MoeAlignBlockSize;

 protected:
  [[noreturn]] void Run(const Tensor, const std::optional<Tensor>,
                        const int64_t, const int64_t, Tensor, Tensor,
                        Tensor) const override {
    throw std::runtime_error("MoeAlignBlockSize is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
