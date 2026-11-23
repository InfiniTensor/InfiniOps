#ifndef INFINI_OPS_BASE_COMPRESSOR_H_
#define INFINI_OPS_BASE_COMPRESSOR_H_

#include <cstdint>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class Compressor : public Operator<Compressor> {
 public:
  static constexpr bool kAutomaticDefault = true;

  Compressor(Tensor cmp_kv, Tensor state_cache, const Tensor x,
             const Tensor wkv, const Tensor wgate, const Tensor ape,
             const Tensor norm_weight, const Tensor rope_sin,
             const Tensor rope_cos, const Tensor state_block_table,
             const Tensor cu_seqlens, const Tensor start_pos,
             int64_t rope_head_dim, int64_t compress_ratio, int64_t coff,
             double norm_eps, int64_t rotary_mode, int64_t cache_mode) {}

  virtual void operator()(
      Tensor cmp_kv, Tensor state_cache, const Tensor x, const Tensor wkv,
      const Tensor wgate, const Tensor ape, const Tensor norm_weight,
      const Tensor rope_sin, const Tensor rope_cos,
      const Tensor state_block_table, const Tensor cu_seqlens,
      const Tensor start_pos, int64_t rope_head_dim, int64_t compress_ratio,
      int64_t coff, double norm_eps, int64_t rotary_mode,
      int64_t cache_mode) const = 0;
};

}  // namespace infini::ops

#endif
