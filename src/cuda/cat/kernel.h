#ifndef INFINI_OPS_CUDA_CAT_KERNEL_H_
#define INFINI_OPS_CUDA_CAT_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "base/cat.h"
#include "common/generic_utils.h"
#include "cuda/cat/kernel.cuh"
#include "cuda/kernel_commons.cuh"
#include "cuda/runtime_utils.h"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

template <typename Backend>
class CudaCat : public Cat {
 public:
  CudaCat(const Tensor first_input, std::vector<Tensor> rest_inputs,
          int64_t dim, Tensor out)
      : Cat{first_input, rest_inputs, dim, out},
        out_shape_{out.shape()},
        out_dtype_{out.dtype()},
        output_size_{out.numel()} {
    assert(out.IsContiguous() &&
           "CudaCat currently requires contiguous output");
    assert(first_input.IsContiguous() &&
           "CudaCat currently requires contiguous inputs");
    assert(first_input.dtype() == out_dtype_);

    input_dim_sizes_.reserve(input_count_);
    input_dim_offsets_.reserve(input_count_);

    input_dim_offsets_.push_back(0);
    input_dim_sizes_.push_back(first_input.shape()[dim_]);
    for (const auto& t : rest_inputs) {
      assert(t.IsContiguous() &&
             "CudaCat currently requires contiguous inputs");
      assert(t.dtype() == out_dtype_);
      input_dim_offsets_.push_back(input_dim_offsets_.back() +
                                   input_dim_sizes_.back());
      input_dim_sizes_.push_back(t.shape()[dim_]);
    }

    inner_ = 1;
    for (size_t i = static_cast<size_t>(dim_) + 1; i < out_shape_.size(); ++i) {
      inner_ *= out_shape_[i];
    }
    out_dim_size_ = out_shape_[dim_];

    size_t count_bytes = input_count_ * sizeof(*d_input_ptrs_);
    size_t dim_size_bytes = input_count_ * sizeof(*d_input_dim_sizes_);
    size_t dim_offset_bytes = input_count_ * sizeof(*d_input_dim_offsets_);
    const size_t metadata_size =
        count_bytes + dim_size_bytes + dim_offset_bytes;
    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_input_ptrs_ = reinterpret_cast<const void**>(d_metadata_ + offset);
    offset += count_bytes;

    d_input_dim_sizes_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_dim_sizes_.data(),
                dim_size_bytes);
    offset += dim_size_bytes;

    d_input_dim_offsets_ =
        reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_dim_offsets_.data(),
                dim_offset_bytes);

    Backend::Memcpy(d_metadata_ + count_bytes, metadata.data() + count_bytes,
                    dim_size_bytes + dim_offset_bytes,
                    Backend::MemcpyHostToDevice);
  }

  ~CudaCat() { Backend::Free(d_metadata_); }

  void operator()(const Tensor first_input, std::vector<Tensor> rest_inputs,
                  int64_t /*dim*/, Tensor out) const override {
    std::vector<const void*> input_ptrs;
    input_ptrs.reserve(input_count_);
    input_ptrs.push_back(first_input.data());
    for (const auto& t : rest_inputs) {
      input_ptrs.push_back(t.data());
    }

    Backend::Memcpy(d_input_ptrs_, input_ptrs.data(),
                    input_count_ * sizeof(*d_input_ptrs_),
                    Backend::MemcpyHostToDevice);

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
    DispatchFunc<AllTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(out_dtype_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));

          CatKernel<T, kBlockSize><<<gridDims, blockDims, 0, cuda_stream>>>(
              reinterpret_cast<T*>(out.data()),
              reinterpret_cast<const T* const*>(d_input_ptrs_),
              d_input_dim_sizes_, d_input_dim_offsets_, input_count_,
              out_dim_size_, inner_, output_size_);
        },
        "CudaCat::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  const void** d_input_ptrs_{nullptr};

  Tensor::Size* d_input_dim_sizes_{nullptr};

  Tensor::Size* d_input_dim_offsets_{nullptr};

  Tensor::Shape out_shape_;

  DataType out_dtype_;

  Tensor::Size output_size_{0};

  Tensor::Size inner_{0};

  Tensor::Size out_dim_size_{0};

  std::vector<Tensor::Size> input_dim_sizes_;

  std::vector<Tensor::Size> input_dim_offsets_;
};

}  // namespace infini::ops

#endif
