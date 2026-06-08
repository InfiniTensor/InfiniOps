#ifndef INFINI_OPS_TRITON_ADD_H_
#define INFINI_OPS_TRITON_ADD_H_

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <vector>

#include "add/infini_ops_triton_add.h"
#include "base/add.h"
#include "data_type.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia, 8> : public Add {
 public:
  using Add::Add;
  using Add::operator();

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    load_infini_ops_triton_add(out.dtype());

    const int ndim = static_cast<int>(ndim_);

    std::vector<int64_t> h_meta(4 * std::max(ndim, 1), 0);
    for (int i = 0; i < ndim; ++i) {
      h_meta[0 * ndim + i] = static_cast<int64_t>(out_shape_[i]);
      h_meta[1 * ndim + i] = static_cast<int64_t>(input_strides_[i]);
      h_meta[2 * ndim + i] = static_cast<int64_t>(other_strides_[i]);
      h_meta[3 * ndim + i] = static_cast<int64_t>(out_strides_[i]);
    }
    const size_t meta_bytes = h_meta.size() * sizeof(int64_t);
    const size_t stride_bytes = ndim * sizeof(int64_t);

    CUdeviceptr d_meta;
    cuMemAlloc(&d_meta, meta_bytes);
    cuMemcpyHtoD(d_meta, h_meta.data(), meta_bytes);

    CUstream stream = static_cast<CUstream>(stream_);
    auto x = reinterpret_cast<CUdeviceptr>(const_cast<void*>(input.data()));
    auto y = reinterpret_cast<CUdeviceptr>(const_cast<void*>(other.data()));
    auto o = reinterpret_cast<CUdeviceptr>(out.data());

    int32_t n = static_cast<int32_t>(out.numel());
    int32_t ndim_val = static_cast<int32_t>(ndim);
    int32_t x_contig = static_cast<int32_t>(is_input_contiguous_);
    int32_t y_contig = static_cast<int32_t>(is_other_contiguous_);
    int32_t out_contig = static_cast<int32_t>(is_out_contiguous_);

    auto result = launch_infini_ops_triton_add(
        out.dtype(), stream, x, y, o,
        d_meta + 0 * stride_bytes,
        d_meta + 1 * stride_bytes,
        d_meta + 2 * stride_bytes,
        d_meta + 3 * stride_bytes,
        x_contig, y_contig, out_contig, ndim_val, n);

    cuMemFree(d_meta);

    assert(result == CUDA_SUCCESS && "Triton `Add` launch failed");
  }
};

}  // namespace infini::ops

#endif
