#include <infini/convolution_layout.h>

namespace infini::ops {

bool SupportsConvolutionChannelsLast(const infini::rt::Device& device) {
#ifdef WITH_CAMBRICON
  return device.type() == infini::rt::Device::Type::kCambricon;
#else
  (void)device;
  return false;
#endif
}

}  // namespace infini::ops
