#ifndef INFINI_OPS_BASE_COMPRESSOR_METADATA_H_
#define INFINI_OPS_BASE_COMPRESSOR_METADATA_H_

#include <cstdint>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class CompressorMetadata : public Operator<CompressorMetadata> {
 public:
  static constexpr bool kAutomaticDefault = true;

  CompressorMetadata(Tensor compress_cos, Tensor compress_sin,
                     Tensor slot_mapping, const Tensor rope_cos,
                     const Tensor rope_sin, const Tensor cu_seqlens,
                     const Tensor start_pos, const Tensor kv_block_table,
                     int64_t kv_block_size, int64_t slot_mapping_format,
                     int64_t compress_ratio, int64_t num_reqs_actual) {}

  virtual void operator()(Tensor compress_cos, Tensor compress_sin,
                          Tensor slot_mapping, const Tensor rope_cos,
                          const Tensor rope_sin, const Tensor cu_seqlens,
                          const Tensor start_pos,
                          const Tensor kv_block_table,
                          int64_t kv_block_size,
                          int64_t slot_mapping_format,
                          int64_t compress_ratio,
                          int64_t num_reqs_actual) const = 0;
};

}  // namespace infini::ops

#endif
