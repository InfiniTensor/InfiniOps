#ifndef INFINI_OPS_NVIDIA_RANDOM_SAMPLE_KERNEL_H_
#define INFINI_OPS_NVIDIA_RANDOM_SAMPLE_KERNEL_H_

#include <utility>

#include "cuda/nvidia/caster.cuh"
#include "cuda/nvidia/runtime_.h"
#include "cuda/random_sample/kernel.h"

namespace infini::ops {

template <>
class Operator<RandomSample, Device::Type::kNvidia>
    : public CudaRandomSample<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRandomSample<Runtime<Device::Type::kNvidia>>::CudaRandomSample;
};

}  // namespace infini::ops

#endif
