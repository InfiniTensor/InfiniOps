#ifndef INFINI_OPS_BASE_ROI_ALIGN_H_
#define INFINI_OPS_BASE_ROI_ALIGN_H_

#include "operator.h"

namespace infini::ops {

// Average pooling over NHWC feature maps. Boxes are [batch, x1, y1, x2, y2].
class RoiAlign : public Operator<RoiAlign> {
 public:
  RoiAlign(const Tensor input, const Tensor boxes, int64_t sampling_ratio,
           float spatial_scale, bool aligned, Tensor out) {}
  virtual void operator()(const Tensor input, const Tensor boxes,
                          int64_t sampling_ratio, float spatial_scale,
                          bool aligned, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
