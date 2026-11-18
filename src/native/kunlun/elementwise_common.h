#ifndef INFINI_OPS_KUNLUN_ELEMENTWISE_COMMON_H_
#define INFINI_OPS_KUNLUN_ELEMENTWISE_COMMON_H_

// Host-side shared infrastructure for the Kunlun backend.
//
// The device kernels live in  (compiled by the XTDK clang++);
// this header only declares their host launch entry points.

#include <cstring>
#include <stdexcept>
#include <vector>

#include "native/kunlun/runtime_.h"
#include "native/kunlun/runtime_utils.h"

namespace infini::ops {

namespace detail {

// dtype tags shared with kernels.cu.
inline int KunlunDtypeTag(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return 0;
    case DataType::kFloat16:
      return 1;
    case DataType::kBFloat16:
      return 2;
    case DataType::kInt16:
      return 5;
    case DataType::kInt32:
      return 3;
    case DataType::kInt64:
      return 4;
    default:
      throw std::runtime_error("kunlun: unsupported dtype " +
                               std::to_string(static_cast<int>(dtype)));
  }
}

}  // namespace detail

namespace kunlun_launch {

// dtype tag: 0=f32, 1=f16, 2=bf16.
void LaunchRelu(int tag, void* out, const void* in,
                const unsigned long long* out_shape,
                const unsigned long long* in_shape,
                const long long* out_strides, const long long* in_strides,
                unsigned long long output_size, unsigned long long ndim,
                bool out_c, bool in_c, void* stream);
void LaunchGelu(int tag, void* out, const void* in,
                const unsigned long long* out_shape,
                const unsigned long long* in_shape,
                const long long* out_strides, const long long* in_strides,
                unsigned long long output_size, unsigned long long ndim,
                bool out_c, bool in_c, int approximate_tanh, void* stream);
void LaunchSilu(int tag, void* out, const void* in,
                const unsigned long long* out_shape,
                const unsigned long long* in_shape,
                const long long* out_strides, const long long* in_strides,
                unsigned long long output_size, unsigned long long ndim,
                bool out_c, bool in_c, void* stream);
void LaunchAdd(int tag, void* out, const void* a, const void* b,
               const unsigned long long* out_shape,
               const unsigned long long* a_shape,
               const unsigned long long* b_shape, const long long* out_strides,
               const long long* a_strides, const long long* b_strides,
               unsigned long long output_size, unsigned long long ndim,
               bool out_c, bool a_c, bool b_c, double alpha, void* stream);
void LaunchSub(int tag, void* out, const void* a, const void* b,
               const unsigned long long* out_shape,
               const unsigned long long* a_shape,
               const unsigned long long* b_shape, const long long* out_strides,
               const long long* a_strides, const long long* b_strides,
               unsigned long long output_size, unsigned long long ndim,
               bool out_c, bool a_c, bool b_c, void* stream);
void LaunchMul(int tag, void* out, const void* a, const void* b,
               const unsigned long long* out_shape,
               const unsigned long long* a_shape,
               const unsigned long long* b_shape, const long long* out_strides,
               const long long* a_strides, const long long* b_strides,
               unsigned long long output_size, unsigned long long ndim,
               bool out_c, bool a_c, bool b_c, void* stream);
void LaunchSiluAndMul(int tag, void* out, const void* in,
                      unsigned long long rows, unsigned long long cols,
                      long long in_row_stride, long long out_row_stride,
                      bool in_c, bool out_c, void* stream);
void LaunchSiluAndMulNd(int tag, void* out, const void* in,
                        const unsigned long long* out_shape,
                        const long long* out_strides,
                        const long long* in_strides, unsigned long long ndim,
                        unsigned long long cols, void* stream);

}  // namespace kunlun_launch

}  // namespace infini::ops

#endif
