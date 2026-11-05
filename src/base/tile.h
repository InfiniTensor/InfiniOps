#ifndef INFINI_OPS_BASE_TILE_H_
#define INFINI_OPS_BASE_TILE_H_

#include "operator.h"

namespace infini::ops {

class Tile : public Operator<Tile> {
 public:
  Tile(const Tensor input, Tensor out) {}
  virtual void operator()(const Tensor input, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
