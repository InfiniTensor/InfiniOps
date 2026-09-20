#ifndef INFINI_OPS_POOLING_H_
#define INFINI_OPS_POOLING_H_

#include <infini/rt.h>

#include <cstdint>
#include <vector>

namespace infini::ops {

// Value-only adapter for providers whose pooling indices have a private
// representation. Returns false when the regular with-indices API should be
// used instead. The caller must retain provider capture memory across replays.
bool TryMaxPool2dValues(infini::rt::TensorView input,
                        const std::vector<int64_t>& kernel_size,
                        const std::vector<int64_t>& stride,
                        const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, bool ceil_mode,
                        infini::rt::TensorView output, void* stream);

}  // namespace infini::ops

#endif  // INFINI_OPS_POOLING_H_
