#ifndef INFINI_OPS_TRITON_ADD_H_
#define INFINI_OPS_TRITON_ADD_H_

#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <vector>

#include "base/add.h"
#include "data_type.h"

extern "C" {
#include "add/infini_ops_triton_add_bf16.h"
#include "add/infini_ops_triton_add_fp16.h"
#include "add/infini_ops_triton_add_fp32.h"
#include "add/infini_ops_triton_add_fp64.h"
#include "add/infini_ops_triton_add_i16.h"
#include "add/infini_ops_triton_add_i32.h"
#include "add/infini_ops_triton_add_i64.h"
#include "add/infini_ops_triton_add_i8.h"
#include "add/infini_ops_triton_add_u16.h"
#include "add/infini_ops_triton_add_u32.h"
#include "add/infini_ops_triton_add_u64.h"
#include "add/infini_ops_triton_add_u8.h"
}

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia, 8> : public Add {
 public:
  using Add::operator();

  Operator(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, out} {
    const int ndim = static_cast<int>(ndim_);
    std::vector<int64_t> h_metadata(4 * std::max(ndim, 1), 0);
    for (int i = 0; i < ndim; ++i) {
      h_metadata[0 * ndim + i] = static_cast<int64_t>(out_shape_[i]);
      h_metadata[1 * ndim + i] = static_cast<int64_t>(input_strides_[i]);
      h_metadata[2 * ndim + i] = static_cast<int64_t>(other_strides_[i]);
      h_metadata[3 * ndim + i] = static_cast<int64_t>(out_strides_[i]);
    }

    const size_t bytes = h_metadata.size() * sizeof(int64_t);
    cuMemAlloc(&d_metadata_, bytes);
    cuMemcpyHtoD(d_metadata_, h_metadata.data(), bytes);

    const size_t stride_bytes = ndim * sizeof(int64_t);
    d_out_shape_ = d_metadata_ + 0 * stride_bytes;
    d_input_stride_ = d_metadata_ + 1 * stride_bytes;
    d_other_stride_ = d_metadata_ + 2 * stride_bytes;
    d_out_stride_ = d_metadata_ + 3 * stride_bytes;
  }

  ~Operator() {
    if (d_metadata_) {
      cuMemFree(d_metadata_);
    }
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    EnsureLoaded(out.dtype());

    CUstream stream = static_cast<CUstream>(stream_);
    auto x = reinterpret_cast<CUdeviceptr>(const_cast<void*>(input.data()));
    auto y = reinterpret_cast<CUdeviceptr>(const_cast<void*>(other.data()));
    auto o = reinterpret_cast<CUdeviceptr>(out.data());

    int32_t n = static_cast<int32_t>(out.numel());
    int32_t ndim_val = static_cast<int32_t>(ndim_);
    int32_t x_contig = static_cast<int32_t>(is_input_contiguous_);
    int32_t y_contig = static_cast<int32_t>(is_other_contiguous_);
    int32_t out_contig = static_cast<int32_t>(is_out_contiguous_);

    CUresult rc = CUDA_ERROR_INVALID_VALUE;
    switch (out.dtype()) {
      case DataType::kFloat16:
        rc = infini_ops_triton_add_fp16_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kBFloat16:
        rc = infini_ops_triton_add_bf16_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kFloat32:
        rc = infini_ops_triton_add_fp32_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kFloat64:
        rc = infini_ops_triton_add_fp64_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kInt8:
        rc = infini_ops_triton_add_i8_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kInt16:
        rc = infini_ops_triton_add_i16_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kInt32:
        rc = infini_ops_triton_add_i32_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kInt64:
        rc = infini_ops_triton_add_i64_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kUInt8:
        rc = infini_ops_triton_add_u8_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kUInt16:
        rc = infini_ops_triton_add_u16_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kUInt32:
        rc = infini_ops_triton_add_u32_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      case DataType::kUInt64:
        rc = infini_ops_triton_add_u64_default(
            stream, x, y, o, d_out_shape_, d_input_stride_, d_other_stride_,
            d_out_stride_, x_contig, y_contig, out_contig, ndim_val, n);
        break;
      default:
        assert(false && "triton `Add` does not support this dtype");
    }

    assert(rc == CUDA_SUCCESS && "Triton `Add` launch failed");
  }

 private:
  CUdeviceptr d_metadata_{0};

  CUdeviceptr d_out_shape_{0};

  CUdeviceptr d_input_stride_{0};

  CUdeviceptr d_other_stride_{0};

  CUdeviceptr d_out_stride_{0};

  static void EnsureLoaded(DataType dtype) {
    static std::once_flag fp16, bf16, fp32, fp64, i8, i16, i32, i64, u8, u16,
        u32, u64;
    switch (dtype) {
      case DataType::kFloat16:
        std::call_once(fp16, &load_infini_ops_triton_add_fp16);
        break;
      case DataType::kBFloat16:
        std::call_once(bf16, &load_infini_ops_triton_add_bf16);
        break;
      case DataType::kFloat32:
        std::call_once(fp32, &load_infini_ops_triton_add_fp32);
        break;
      case DataType::kFloat64:
        std::call_once(fp64, &load_infini_ops_triton_add_fp64);
        break;
      case DataType::kInt8:
        std::call_once(i8, &load_infini_ops_triton_add_i8);
        break;
      case DataType::kInt16:
        std::call_once(i16, &load_infini_ops_triton_add_i16);
        break;
      case DataType::kInt32:
        std::call_once(i32, &load_infini_ops_triton_add_i32);
        break;
      case DataType::kInt64:
        std::call_once(i64, &load_infini_ops_triton_add_i64);
        break;
      case DataType::kUInt8:
        std::call_once(u8, &load_infini_ops_triton_add_u8);
        break;
      case DataType::kUInt16:
        std::call_once(u16, &load_infini_ops_triton_add_u16);
        break;
      case DataType::kUInt32:
        std::call_once(u32, &load_infini_ops_triton_add_u32);
        break;
      case DataType::kUInt64:
        std::call_once(u64, &load_infini_ops_triton_add_u64);
        break;
      default:
        break;
    }
  }
};

}  // namespace infini::ops

#endif
