#ifndef INFINI_OPS_BASE_VLLM_QUANT_LIGHTNING_INDEXER_METADATA_H_
#define INFINI_OPS_BASE_VLLM_QUANT_LIGHTNING_INDEXER_METADATA_H_
#include <cstdint>
#include <string>
#include "operator.h"
#include "tensor.h"
namespace infini::ops {
class VllmQuantLightningIndexerMetadata : public Operator<VllmQuantLightningIndexerMetadata> {
 public:
  static constexpr bool kAutomaticDefault = true;
  VllmQuantLightningIndexerMetadata(Tensor output, const Tensor actual_seq_lengths_query,
      const Tensor actual_seq_lengths_key, int64_t num_heads_q, int64_t num_heads_k,
      int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode,
      int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k,
      std::string layout_query, std::string layout_key, int64_t sparse_count,
      int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio) {}
  virtual void operator()(Tensor output, const Tensor actual_seq_lengths_query,
      const Tensor actual_seq_lengths_key, int64_t num_heads_q, int64_t num_heads_k,
      int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode,
      int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k,
      std::string layout_query, std::string layout_key, int64_t sparse_count,
      int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio) const = 0;
};
}
#endif
