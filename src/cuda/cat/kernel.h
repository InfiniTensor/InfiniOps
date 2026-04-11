#ifndef INFINI_OPS_CUDA_CAT_KERNEL_H_
#define INFINI_OPS_CUDA_CAT_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "base/cat.h"
#include "common/generic_utils.h"
#include "cuda/cat/kernel.cuh"
#include "cuda/kernel_commons.cuh"
#include "cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaCat : public Cat {
 public:
  CudaCat(const Tensor first_input, std::vector<Tensor> rest_inputs,
          int64_t dim, Tensor out)
      : Cat{first_input, std::move(rest_inputs), dim, out} {
    // Allocate device memory for input pointers and cumulative sizes.
    size_t ptrs_size = input_count_ * sizeof(const void*);
    size_t cum_size = input_count_ * sizeof(size_t);
    size_t metadata_size = ptrs_size + cum_size;

    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    // Copy input data pointers.
    std::vector<const void*> input_ptrs(input_count_);

    for (size_t i = 0; i < input_count_; ++i) {
      input_ptrs[i] = inputs_[i].data();
    }

    std::memcpy(metadata.data(), input_ptrs.data(), ptrs_size);

    // Copy cumulative dimension sizes.
    std::memcpy(metadata.data() + ptrs_size, cum_dim_sizes_.data(), cum_size);

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::MemcpyHostToDevice);

    d_inputs_ = reinterpret_cast<const void**>(d_metadata_);
    d_cum_sizes_ = reinterpret_cast<size_t*>(d_metadata_ + ptrs_size);
  }

  ~CudaCat() { Backend::Free(d_metadata_); }

  void operator()(const Tensor first_input, std::vector<Tensor> rest_inputs,
                  int64_t dim, Tensor out) const override {
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
    DispatchFunc<AllTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(dtype_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
          dim3 blockDims(
              std::min(static_cast<size_t>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));

          T* d_out = reinterpret_cast<T*>(out.data());
          size_t total_dim_size = cum_dim_sizes_.back();

          CatKernel<T>
              <<<gridDims, blockDims, 0, cuda_stream>>>(
                  d_out, d_inputs_, d_cum_sizes_, input_count_, outer_size_,
                  inner_size_, total_dim_size, output_size_);
        },
        "CudaCat::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  const void** d_inputs_{nullptr};

  size_t* d_cum_sizes_{nullptr};
};

}  // namespace infini::ops

#endif
