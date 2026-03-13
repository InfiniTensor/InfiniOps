#ifndef INFINI_OPS_MOORE_SWIGLU_LAUNCH_H_
#define INFINI_OPS_MOORE_SWIGLU_LAUNCH_H_

#include <cstddef>
#include <cstdint>

#include <musa_runtime_api.h>

namespace infini::ops::swiglu::moore {

musaError_t LaunchSwiglu(const void* input, const void* gate, void* out,
                         const size_t* out_shape, const size_t* input_shape,
                         const size_t* gate_shape, const ptrdiff_t* out_strides,
                         const ptrdiff_t* input_strides,
                         const ptrdiff_t* gate_strides, size_t output_size,
                         size_t ndim, bool out_contiguous,
                         bool input_contiguous, bool gate_contiguous, int dtype,
                         musaStream_t stream);

}  // namespace infini::ops::swiglu::moore

#endif  // INFINI_OPS_MOORE_SWIGLU_LAUNCH_H_
