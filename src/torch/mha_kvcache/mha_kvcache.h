#ifndef INFINI_OPS_TORCH_MHA_KVCACHE_H_
#define INFINI_OPS_TORCH_MHA_KVCACHE_H_

#include <cstdint>

#include "base/mha_kvcache.h"

namespace infini::ops {

// RAII hint: while alive, `MhaKvcache`'s Torch backend reads `seqlens_k`
// values from the given host pointer instead of issuing a D2H sync on the
// device tensor. The caller owns the buffer; it must outlive the hint.
//
// Rationale: `seqlens_k` drives per-sequence loop bounds / scratch shape,
// so the backend needs host-side scalars. With a paged KV cache in a 32-
// layer model every decode step would otherwise do 32 D2H syncs, each
// serializing the GPU stream. One host copy per forward step is plenty.
class MhaKvcacheHostSeqlensHint {
 public:
  explicit MhaKvcacheHostSeqlensHint(const std::int64_t* host_ptr) noexcept;
  ~MhaKvcacheHostSeqlensHint() noexcept;

  MhaKvcacheHostSeqlensHint(const MhaKvcacheHostSeqlensHint&) = delete;
  MhaKvcacheHostSeqlensHint& operator=(const MhaKvcacheHostSeqlensHint&) =
      delete;

 private:
  const std::int64_t* previous_;
};

namespace mha_kvcache_internal {
const std::int64_t* current_host_seqlens_ptr() noexcept;
}  // namespace mha_kvcache_internal

template <Device::Type kDev>
class Operator<MhaKvcache, kDev, 1> : public MhaKvcache {
 public:
  Operator(const Tensor q, const Tensor k_cache, const Tensor v_cache,
           const Tensor seqlens_k, const Tensor block_table, float scale,
           Tensor out);

  void operator()(const Tensor q, const Tensor k_cache, const Tensor v_cache,
                  const Tensor seqlens_k, const Tensor block_table,
                  float scale, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
