#ifndef INFINI_OPS_BASE_HISTOGRAM_BINS_TENSOR_H_
#define INFINI_OPS_BASE_HISTOGRAM_BINS_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class HistogramBinsTensor : public Operator<HistogramBinsTensor> {
 public:
  HistogramBinsTensor(const Tensor self, const Tensor bins, Tensor hist,
                      Tensor bin_edges)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        bins_shape_{bins.shape()},
        bins_strides_{bins.strides()},
        bins_type_{bins.dtype()},
        hist_shape_{hist.shape()},
        hist_strides_{hist.strides()},
        hist_type_{hist.dtype()},
        bin_edges_shape_{bin_edges.shape()},
        bin_edges_strides_{bin_edges.strides()},
        bin_edges_type_{bin_edges.dtype()},
        device_index_{hist.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor bins, Tensor hist,
                          Tensor bin_edges) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape bins_shape_;
  Tensor::Strides bins_strides_;
  DataType bins_type_;
  Tensor::Shape hist_shape_;
  Tensor::Strides hist_strides_;
  DataType hist_type_;
  Tensor::Shape bin_edges_shape_;
  Tensor::Strides bin_edges_strides_;
  DataType bin_edges_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
