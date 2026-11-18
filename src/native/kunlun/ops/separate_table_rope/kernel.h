#ifndef INFINI_OPS_KUNLUN_SEPARATE_TABLE_ROPE_KERNEL_H_
#define INFINI_OPS_KUNLUN_SEPARATE_TABLE_ROPE_KERNEL_H_

#include <stdexcept>

#include "base/separate_table_rope.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun separate-table rope. Mirrors the CPU reference semantics: rank 3/4
// inputs, interleaved (GPT-J, is_neox=false) or split-half (NeoX,
// is_neox=true) pairs, per-sequence or per-batch positions, in-place safe.
// Layout contract from SeparateTableRope::supports(): last-dim contiguous,
// tables [rows, half] with row stride == half.
template <>
class Operator<SeparateTableRope, Device::Type::kKunlun>
    : public SeparateTableRope {
 public:
  using SeparateTableRope::SeparateTableRope;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const Tensor positions,
                  const Tensor sin_table, const Tensor cos_table, bool is_neox,
                  Tensor out) const override {
    if (input.numel() == 0) {
      return;
    }
    const auto rank = input.ndim();
    const bool batched = rank == 4;
    const unsigned long long batch = batched ? input.shape()[0] : 1ULL;
    const unsigned long long seq = input.shape()[rank - 3];
    const unsigned long long heads = input.shape()[rank - 2];
    const int half = (int)sin_table.shape()[1];

    // Strides: batch / seq / head (elements).
    const long long x_stride_seq = batched ? (long long)input.stride(1)
                                           : (long long)input.stride(rank - 3);
    const long long x_stride_head = (long long)input.stride(rank - 2);
    const long long y_stride_seq =
        batched ? (long long)out.stride(1) : (long long)out.stride(rank - 3);
    const long long y_stride_head = (long long)out.stride(rank - 2);
    const long long x_stride_batch = batched ? (long long)input.stride(0) : 0;
    const long long y_stride_batch = batched ? (long long)out.stride(0) : 0;
    const bool batched_pos = positions.ndim() == 2;

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(input.dtype());
    kunlun_launch::LaunchRope(
        tag, /*interleaved=*/!is_neox, out.data(), input.data(),
        positions.data(), sin_table.data(), cos_table.data(), half,
        x_stride_seq, x_stride_head, y_stride_seq, y_stride_head, seq, heads,
        batch, x_stride_batch, y_stride_batch, batched_pos, stream);
  }
};

}  // namespace infini::ops

#endif
