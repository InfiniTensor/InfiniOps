#ifndef INFINI_OPS_CUDA_SWIGLU_KERNEL_H_
#define INFINI_OPS_CUDA_SWIGLU_KERNEL_H_

#include <cstdint>

#include "base/swiglu.h"
#include "common/generic_utils.h"
#include "cuda/swiglu/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaSwiglu : public Swiglu {
 public:
  CudaSwiglu(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu{input, gate, out} {
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);

    Backend::malloc((void**)&d_input_shape_, shape_size);
    Backend::malloc((void**)&d_gate_shape_, shape_size);
    Backend::malloc((void**)&d_out_shape_, shape_size);
    Backend::malloc((void**)&d_input_strides_, strides_size);
    Backend::malloc((void**)&d_gate_strides_, strides_size);
    Backend::malloc((void**)&d_out_strides_, strides_size);

    Backend::memcpy(d_input_shape_, input_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_gate_shape_, gate_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_out_shape_, out_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_input_strides_, input_strides_.data(), strides_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_gate_strides_, gate_strides_.data(), strides_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_out_strides_, out_strides_.data(), strides_size,
                    Backend::memcpyH2D);
  }

  ~CudaSwiglu() {
    Backend::free(d_input_shape_);
    Backend::free(d_gate_shape_);
    Backend::free(d_out_shape_);
    Backend::free(d_input_strides_);
    Backend::free(d_gate_strides_);
    Backend::free(d_out_strides_);
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    int block_size = GetOptimalBlockSize();
    DispatchFunc<AllFloatTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::stream_t>(stream_ ? stream_ : 0);
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          const T* d_gate = reinterpret_cast<const T*>(gate.data());

          SwigluKernel<T, kBlockSize><<<gridDims, blockDims, 0, cuda_stream>>>(
              d_out, d_input, d_gate, d_out_shape_, d_input_shape_,
              d_gate_shape_, d_out_strides_, d_input_strides_, d_gate_strides_,
              output_size_, ndim_, is_out_contiguous_, is_input_contiguous_,
              is_gate_contiguous_);
        },
        "CudaSwiglu::operator()");
  }

 private:
  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_gate_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_gate_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
