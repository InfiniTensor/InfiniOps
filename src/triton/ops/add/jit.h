#ifndef INFINI_OPS_TRITON_OPS_ADD_JIT_H_
#define INFINI_OPS_TRITON_OPS_ADD_JIT_H_

#include <string>
#include <vector>

#include "base/add.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Add, kDev, 10> : public JitOperatorBase<Add, kDev> {
 public:
  using JitOperatorBase<Add, kDev>::JitOperatorBase;

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const;

  static JitConfig DefaultConfig() { return {4u, 3u, {{"BLOCK_SIZE", 1024}}}; }

  static std::vector<std::string> DefaultKey() { return {"n_elements"}; }

  static std::vector<JitConfig> AutotuneConfigs() {
    return {
        {4u, 3u, {{"BLOCK_SIZE", 256}}},
        {4u, 3u, {{"BLOCK_SIZE", 512}}},
        {8u, 4u, {{"BLOCK_SIZE", 1024}}},
        {8u, 4u, {{"BLOCK_SIZE", 2048}}},
    };
  }
};

}  // namespace infini::ops

#endif
