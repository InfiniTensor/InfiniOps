#include <musa_runtime.h>

#include "data_type.h"
#include "moore/swiglu/launch.h"
#include "moore/swiglu/swiglu_moore_kernel.h"

namespace infini::ops::swiglu::moore {

__device__ __forceinline__ size_t IndexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t* shape,
    const ptrdiff_t* strides) {
  size_t res = 0;
  for (size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }

  return res;
}

template <typename T, unsigned int BLOCK_SIZE>
__global__ void SwigluKernel(
    T* out, const T* input, const T* gate, const size_t* out_shape,
    const size_t* input_shape, const size_t* gate_shape,
    const ptrdiff_t* out_strides, const ptrdiff_t* input_strides,
    const ptrdiff_t* gate_strides, size_t output_size, size_t ndim,
    size_t offset, bool out_contiguous, bool input_contiguous,
    bool gate_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
  if (idx >= output_size) {
    return;
  }

  size_t out_idx = out_contiguous ? idx
                                  : IndexToOffset(idx, ndim, out_shape,
                                                 out_strides);
  size_t input_idx = input_contiguous ? idx
                                      : IndexToOffset(idx, ndim, input_shape,
                                                     input_strides);
  size_t gate_idx = gate_contiguous ? idx
                                    : IndexToOffset(idx, ndim, gate_shape,
                                                   gate_strides);

  SwiGLUOp op;
  out[out_idx] = op(input[input_idx], gate[gate_idx]);
}

template <unsigned int BLOCK_SIZE, typename T>
musaError_t LaunchKernel(
    void* out, const void* input, const void* gate, const size_t* out_shape,
    const size_t* input_shape, const size_t* gate_shape,
    const ptrdiff_t* out_strides, const ptrdiff_t* input_strides,
    const ptrdiff_t* gate_strides, size_t output_size, size_t ndim,
    bool out_contiguous, bool input_contiguous, bool gate_contiguous,
    musaStream_t stream) {
  if (output_size == 0) {
    return musaSuccess;
  }

  dim3 block(BLOCK_SIZE);
  dim3 grid((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
  size_t step = static_cast<size_t>(grid.x) * block.x;

  for (size_t offset = 0; offset < output_size; offset += step) {
    SwigluKernel<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        static_cast<T*>(out), static_cast<const T*>(input),
        static_cast<const T*>(gate), out_shape, input_shape, gate_shape,
        out_strides, input_strides, gate_strides, output_size, ndim, offset,
        out_contiguous, input_contiguous, gate_contiguous);
    auto err = musaGetLastError();
    if (err != musaSuccess) {
      return err;
    }
  }

  return musaSuccess;
}

musaError_t LaunchSwiglu(
    const void* input, const void* gate, void* out, const size_t* out_shape,
    const size_t* input_shape, const size_t* gate_shape,
    const ptrdiff_t* out_strides, const ptrdiff_t* input_strides,
    const ptrdiff_t* gate_strides, size_t output_size, size_t ndim,
    bool out_contiguous, bool input_contiguous, bool gate_contiguous, int dtype,
    musaStream_t stream) {
  switch (static_cast<DataType>(dtype)) {
    case DataType::kFloat16:
      return LaunchKernel<256, half>(out, input, gate, out_shape, input_shape,
                                     gate_shape, out_strides, input_strides,
                                     gate_strides, output_size, ndim,
                                     out_contiguous, input_contiguous,
                                     gate_contiguous, stream);
    case DataType::kBFloat16:
      return LaunchKernel<256, cuda_bfloat16>(
          out, input, gate, out_shape, input_shape, gate_shape, out_strides,
          input_strides, gate_strides, output_size, ndim, out_contiguous,
          input_contiguous, gate_contiguous, stream);
    case DataType::kFloat32:
      return LaunchKernel<256, float>(
          out, input, gate, out_shape, input_shape, gate_shape, out_strides,
          input_strides, gate_strides, output_size, ndim, out_contiguous,
          input_contiguous, gate_contiguous, stream);
    case DataType::kFloat64:
      return LaunchKernel<256, double>(
          out, input, gate, out_shape, input_shape, gate_shape, out_strides,
          input_strides, gate_strides, output_size, ndim, out_contiguous,
          input_contiguous, gate_contiguous, stream);
    default:
      return musaErrorInvalidValue;
  }
}

}  // namespace infini::ops::swiglu::moore
