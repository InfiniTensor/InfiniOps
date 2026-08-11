#include "triton/jit/config_.h"

namespace infini::ops::triton::jit {

const AutoTuningOptions* Config::auto_tuning_options() const { return nullptr; }

const AutoTuningOptions* AutoTuningConfig::auto_tuning_options() const {
  return &options_;
}

}  // namespace infini::ops::triton::jit
