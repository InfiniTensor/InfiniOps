#ifndef INFINI_OPS_BASE_FRACTIONAL_MAX_POOL3D_OUTPUT_H_
#define INFINI_OPS_BASE_FRACTIONAL_MAX_POOL3D_OUTPUT_H_

#include "operator.h"

namespace infini::ops {

class FractionalMaxPool3dOutput : public Operator<FractionalMaxPool3dOutput> {
 public:
  FractionalMaxPool3dOutput(const Tensor self,
                            const std::vector<int64_t> kernel_size,
                            const std::vector<int64_t> output_size,
                            const Tensor random_samples, Tensor output,
                            Tensor indices)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        random_samples_shape_{random_samples.shape()},
        random_samples_strides_{random_samples.strides()},
        random_samples_type_{random_samples.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor self,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> output_size,
                          const Tensor random_samples, Tensor output,
                          Tensor indices) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape random_samples_shape_;
  Tensor::Strides random_samples_strides_;
  DataType random_samples_type_;
  Tensor::Shape output_shape_;
  Tensor::Strides output_strides_;
  DataType output_type_;
  Tensor::Shape indices_shape_;
  Tensor::Strides indices_strides_;
  DataType indices_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
