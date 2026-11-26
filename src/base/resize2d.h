#ifndef INFINI_OPS_BASE_RESIZE2D_H_
#define INFINI_OPS_BASE_RESIZE2D_H_

#include "operator.h"

namespace infini::ops {

class Resize2d : public Operator<Resize2d> {
 public:
  enum Mode : int64_t { kNearest, kLinear };
  enum Coordinates : int64_t { kHalfPixel, kPytorchHalfPixel, kAlignCorners, kAsymmetric };
  enum Rounding : int64_t { kPreferFloor, kPreferCeil, kFloor, kCeil };

  Resize2d(const Tensor input, int64_t mode, int64_t coordinates,
           int64_t rounding, float height_scale, float width_scale, Tensor out) {}
  virtual void operator()(const Tensor input, int64_t mode, int64_t coordinates,
                          int64_t rounding, float height_scale, float width_scale,
                          Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
