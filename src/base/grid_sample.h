#ifndef INFINI_OPS_BASE_GRID_SAMPLE_H_
#define INFINI_OPS_BASE_GRID_SAMPLE_H_

#include "operator.h"

namespace infini::ops {

// Spatial sampling in NHWC layout; the last grid dimension stores (x, y).
class GridSample : public Operator<GridSample> {
 public:
  GridSample(const Tensor input, const Tensor grid, bool nearest,
             bool reflection, bool align_corners, Tensor out) {}
  virtual void operator()(const Tensor input, const Tensor grid, bool nearest,
                          bool reflection, bool align_corners, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
