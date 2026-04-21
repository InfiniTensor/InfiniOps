#ifndef INFINI_OPS_TORCH_MHA_VARLEN_H_
#define INFINI_OPS_TORCH_MHA_VARLEN_H_

#include "base/mha_varlen.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<MhaVarlen, kDev, 1> : public MhaVarlen {
 public:
  Operator(const Tensor q, const Tensor k_cache, const Tensor v_cache,
           const Tensor cum_seqlens_q, const Tensor cum_seqlens_k,
           const Tensor block_table, float scale, Tensor out);

  void operator()(const Tensor q, const Tensor k_cache, const Tensor v_cache,
                  const Tensor cum_seqlens_q, const Tensor cum_seqlens_k,
                  const Tensor block_table, float scale,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
