#ifndef INFINI_OPS_BASE_SWIGLU_H_
#define INFINI_OPS_BASE_SWIGLU_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Swiglu : public Operator<Swiglu> {
 public:
  Swiglu(const Tensor input, const Tensor gate, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        gate_type_{gate.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        gate_shape_{gate.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        gate_strides_{gate.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_gate_contiguous_{gate.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(
        input_type_ == gate_type_ && gate_type_ == out_type_ &&
        "Operator `Swiglu` requires all input and output tensors to have the "
        "same dtype.");
  }

  virtual void operator()(const Tensor input, const Tensor gate,
                          Tensor out) const = 0;

 protected:
  struct WorkspaceLayout {
    Tensor::Size* input_shape{nullptr};

    Tensor::Size* gate_shape{nullptr};

    Tensor::Size* out_shape{nullptr};

    Tensor::Stride* input_strides{nullptr};

    Tensor::Stride* gate_strides{nullptr};

    Tensor::Stride* out_strides{nullptr};
  };

  std::size_t MetadataWorkspaceSizeInBytes() const {
    return 3 * ndim_ * sizeof(Tensor::Size) +
           3 * ndim_ * sizeof(Tensor::Stride);
  }

  WorkspaceLayout ResolveWorkspaceLayout(void* workspace) const {
    WorkspaceLayout layout;

    if (workspace == nullptr) {
      return layout;
    }

    auto* cursor = static_cast<char*>(workspace);

    layout.input_shape = reinterpret_cast<Tensor::Size*>(cursor);
    cursor += ndim_ * sizeof(Tensor::Size);
    layout.gate_shape = reinterpret_cast<Tensor::Size*>(cursor);
    cursor += ndim_ * sizeof(Tensor::Size);
    layout.out_shape = reinterpret_cast<Tensor::Size*>(cursor);
    cursor += ndim_ * sizeof(Tensor::Size);
    layout.input_strides = reinterpret_cast<Tensor::Stride*>(cursor);
    cursor += ndim_ * sizeof(Tensor::Stride);
    layout.gate_strides = reinterpret_cast<Tensor::Stride*>(cursor);
    cursor += ndim_ * sizeof(Tensor::Stride);
    layout.out_strides = reinterpret_cast<Tensor::Stride*>(cursor);

    return layout;
  }

  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType gate_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape gate_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides gate_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_gate_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
