#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_COMPRESS_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_COMPRESS_H_

#include "operator.h"

namespace infini::ops {

// sources encodes input rows (>=0), old state rows (-2-row), or padding (-1).
// C4 has 8 sources ordered previous group then current group. C128 has 128.
// C4 APE can be the prepared [8, dim] or raw checkpoint [4, 2 * dim].
// An all-padding output row is zero. state_slots uses unique nonnegative
// destinations, or -1. Pooling observes the state from before the launch.
class DeepseekV4Compress : public Operator<DeepseekV4Compress> {
 public:
  DeepseekV4Compress(const Tensor projection, const Tensor ape, const Tensor sources,
                     const Tensor state_slots, Tensor state, Tensor out) {
    (void)projection;
    (void)ape;
    (void)sources;
    (void)state_slots;
    (void)state;
    (void)out;
  }

  virtual void operator()(const Tensor projection, const Tensor ape, const Tensor sources,
                          const Tensor state_slots, Tensor state, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
