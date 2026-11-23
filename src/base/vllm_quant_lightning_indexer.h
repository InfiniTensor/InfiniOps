#ifndef INFINI_OPS_BASE_VLLM_QUANT_LIGHTNING_INDEXER_H_
#define INFINI_OPS_BASE_VLLM_QUANT_LIGHTNING_INDEXER_H_
#include <cstdint>
#include <string>
#include "operator.h"
#include "tensor.h"
namespace infini::ops {
class VllmQuantLightningIndexer : public Operator<VllmQuantLightningIndexer> {
 public:
  static constexpr bool kAutomaticDefault = true;
  VllmQuantLightningIndexer(Tensor sparse_indices, Tensor sparse_values, const Tensor query, const Tensor key, const Tensor weights, const Tensor query_scale, const Tensor key_scale, const Tensor qlens, const Tensor klens, const Tensor block_table, const Tensor metadata, int64_t query_quant_mode, int64_t key_quant_mode, std::string layout_query, std::string layout_key, int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_values) {}
  virtual void operator()(Tensor sparse_indices, Tensor sparse_values, const Tensor query, const Tensor key, const Tensor weights, const Tensor query_scale, const Tensor key_scale, const Tensor qlens, const Tensor klens, const Tensor block_table, const Tensor metadata, int64_t query_quant_mode, int64_t key_quant_mode, std::string layout_query, std::string layout_key, int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_values) const = 0;
};
}
#endif
