#ifndef INFINI_OPS_CONVOLUTION_LAYOUT_H_
#define INFINI_OPS_CONVOLUTION_LAYOUT_H_

#include <infini/rt.h>

namespace infini::ops {

// Logical NCHW tensors may use packed channels-last strides.
bool SupportsConvolutionChannelsLast(const infini::rt::Device& device);

}  // namespace infini::ops

#endif  // INFINI_OPS_CONVOLUTION_LAYOUT_H_
