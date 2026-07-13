#ifndef INFINI_OPS_TENSOR_H_
#define INFINI_OPS_TENSOR_H_

#include <infini/rt.h>

#include <memory>

namespace infini::ops {

class Tensor : public infini::rt::TensorView {
 public:
  using infini::rt::TensorView::TensorView;

  Tensor(const infini::rt::TensorView& tensor)
      : infini::rt::TensorView(tensor) {}

  // Opaque handle to the original backend tensor, attached by the pybind layer
  // so the PyTorch fallback can reuse its existing tensor wrapper. Type-erased
  // (`shared_ptr<const void>`) so the core Tensor has no torch dependency.
  void set_source_handle(std::shared_ptr<const void> handle) {
    source_handle_ = std::move(handle);
  }

  const std::shared_ptr<const void>& source_handle() const {
    return source_handle_;
  }

 private:
  std::shared_ptr<const void> source_handle_;
};

}  // namespace infini::ops

template <>
struct std::hash<infini::ops::Tensor> {
  std::size_t operator()(const infini::ops::Tensor& tensor) const {
    return std::hash<infini::rt::TensorView>{}(
        static_cast<const infini::rt::TensorView&>(tensor));
  }
};

template <>
struct std::equal_to<infini::ops::Tensor> {
  bool operator()(const infini::ops::Tensor& a,
                  const infini::ops::Tensor& b) const {
    return std::equal_to<infini::rt::TensorView>{}(
        static_cast<const infini::rt::TensorView&>(a),
        static_cast<const infini::rt::TensorView&>(b));
  }
};

#endif
