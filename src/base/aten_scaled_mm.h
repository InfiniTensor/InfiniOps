#ifndef INFINI_OPS_BASE_ATEN_SCALED_MM_H_
#define INFINI_OPS_BASE_ATEN_SCALED_MM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class AtenScaledMm : public Operator<AtenScaledMm> {
 public:
  AtenScaledMm(const Tensor input, const Tensor mat2, const Tensor scale_a,
               const Tensor scale_b, const std::optional<Tensor> bias,
               const std::optional<Tensor> scale_result,
               const std::optional<DataType> out_dtype,
               const bool use_fast_accum, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mat2_shape_{mat2.shape()},
        mat2_strides_{mat2.strides()},
        mat2_type_{mat2.dtype()},
        scale_a_shape_{scale_a.shape()},
        scale_a_strides_{scale_a.strides()},
        scale_a_type_{scale_a.dtype()},
        scale_b_shape_{scale_b.shape()},
        scale_b_strides_{scale_b.strides()},
        scale_b_type_{scale_b.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_bias_{bias.has_value()},
        bias_shape_{bias ? bias->shape() : Tensor::Shape{}},
        bias_strides_{bias ? bias->strides() : Tensor::Strides{}},
        bias_type_{bias ? bias->dtype() : DataType::kFloat32},
        has_scale_result_{scale_result.has_value()},
        scale_result_shape_{scale_result ? scale_result->shape()
                                         : Tensor::Shape{}},
        scale_result_strides_{scale_result ? scale_result->strides()
                                           : Tensor::Strides{}},
        scale_result_type_{scale_result ? scale_result->dtype()
                                        : DataType::kFloat32},
        out_dtype_{out_dtype},
        use_fast_accum_{use_fast_accum},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor mat2,
                          const Tensor scale_a, const Tensor scale_b,
                          const std::optional<Tensor> bias,
                          const std::optional<Tensor> scale_result,
                          const std::optional<DataType> out_dtype,
                          const bool use_fast_accum, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mat2_shape_;

  Tensor::Strides mat2_strides_;

  DataType mat2_type_;

  Tensor::Shape scale_a_shape_;

  Tensor::Strides scale_a_strides_;

  DataType scale_a_type_;

  Tensor::Shape scale_b_shape_;

  Tensor::Strides scale_b_strides_;

  DataType scale_b_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool has_bias_{false};

  Tensor::Shape bias_shape_;

  Tensor::Strides bias_strides_;

  DataType bias_type_{DataType::kFloat32};

  bool has_scale_result_{false};

  Tensor::Shape scale_result_shape_;

  Tensor::Strides scale_result_strides_;

  DataType scale_result_type_{DataType::kFloat32};

  std::optional<DataType> out_dtype_{};

  bool use_fast_accum_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
