#ifndef INFINI_OPS_CAMBRICON_TILE_KERNEL_H_
#define INFINI_OPS_CAMBRICON_TILE_KERNEL_H_

#include "base/tile.h"
#include "native/cambricon/cnnl_utils.h"

namespace infini::ops {

template <>
class Operator<Tile, Device::Type::kCambricon> : public Tile {
 public:
  Operator(const Tensor input, Tensor out) : Tile{input, out} {
    if (input.dtype() != out.dtype() || input.device() != out.device() ||
        !input.IsContiguous() || !out.IsContiguous() || input.ndim() != out.ndim())
      throw std::invalid_argument("Cambricon Tile requires matching contiguous tensors");
    if (out.numel() == 0) return;
    for (size_t i = 0; i < out.ndim(); ++i) {
      if (input.size(i) == 0 || out.size(i) % input.size(i) != 0)
        throw std::invalid_argument("Cambricon Tile output must be a multiple of input");
    }
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input);
    out_ = cnnl_utils::MakeTensorDescriptor(out);
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    INFINI_OPS_CNNL_CHECK(cnnlTile(handle_.get(), input_.get(), input.data(),
                                  out_.get(), out.data()));
  }

 private:
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, out_;
};

}  // namespace infini::ops

#endif
