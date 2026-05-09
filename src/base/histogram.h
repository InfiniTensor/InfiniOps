#ifndef INFINI_OPS_BASE_HISTOGRAM_H_
#define INFINI_OPS_BASE_HISTOGRAM_H_

#include "operator.h"

namespace infini::ops {

class Histogram : public Operator<Histogram> {
 public:
  Histogram(const Tensor input, const Tensor bins, const bool density,
            Tensor hist, Tensor bin_edges)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        bins_shape_{bins.shape()},
        bins_strides_{bins.strides()},
        bins_type_{bins.dtype()},
        hist_shape_{hist.shape()},
        hist_strides_{hist.strides()},
        hist_type_{hist.dtype()},
        bin_edges_shape_{bin_edges.shape()},
        bin_edges_strides_{bin_edges.strides()},
        bin_edges_type_{bin_edges.dtype()},
        density_{density},
        device_index_{hist.device().index()} {}

  Histogram(const Tensor input, const int64_t bins, const bool density,
            Tensor hist, Tensor bin_edges)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        hist_shape_{hist.shape()},
        hist_strides_{hist.strides()},
        hist_type_{hist.dtype()},
        bin_edges_shape_{bin_edges.shape()},
        bin_edges_strides_{bin_edges.strides()},
        bin_edges_type_{bin_edges.dtype()},
        density_{density},
        bins_{bins},
        device_index_{hist.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor bins,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;

  virtual void operator()(const Tensor input, const int64_t bins,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape bins_shape_;

  Tensor::Strides bins_strides_;

  DataType bins_type_;

  Tensor::Shape hist_shape_;

  Tensor::Strides hist_strides_;

  DataType hist_type_;

  Tensor::Shape bin_edges_shape_;

  Tensor::Strides bin_edges_strides_;

  DataType bin_edges_type_;

  bool density_{};

  int64_t bins_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
