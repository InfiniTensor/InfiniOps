#ifndef INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_H_

#include "base/clover_coreset.h"
#include "data_type.h"
#include "dispatcher.h"
#include "common/generic_utils.h"
#include "native/cuda/ops/clover_coreset/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverCoreset : public CloverCoreset {
 public:
  // Workspace: mass [blocks,256] fp32 + allow [blocks,256] int8, laid out
  // exactly like the Triton provider declares.
  CudaCloverCoreset(const Tensor logits, size_t block_size, double top_p,
                    Tensor out)
      : CloverCoreset(logits, block_size, top_p, out),
        blocks_(logits.size(0) / block_size),
        tokens_(logits.size(0)) {}

  size_t workspace_size_in_bytes() const override {
    // mass [blocks,256] fp32 + allow [blocks,256] i8 + row_top8 [tokens,16] fp32.
    return blocks_ * kCloverCoresetExperts * 5 + tokens_ * 16 * sizeof(float);
  }

  void operator()(const Tensor logits, size_t block_size, double top_p,
                  Tensor out) const override {
    if (blocks_ == 0) return;
    if (!workspace_ || workspace_size_in_bytes_ < workspace_size_in_bytes())
      throw std::invalid_argument("CudaCloverCoreset requires its workspace.");
    const auto tokens = logits.size(0);
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    float* mass = reinterpret_cast<float*>(workspace_);
    int8_t* allow =
        reinterpret_cast<int8_t*>(workspace_) + blocks_ * kCloverCoresetExperts * 4;
    float* row_top8 =
        reinterpret_cast<float*>(workspace_) + blocks_ * kCloverCoresetExperts * 5;

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(logits.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;

          // Stage A: one block per row (parallel across rows); stage B:
          // one block per routing block, one thread per expert.
          CloverCoresetRowKernel<Backend::kDeviceType, Data>
              <<<static_cast<unsigned>(tokens),
                 kCloverCoresetExperts,
                 (kCloverCoresetExperts + 2 * 8) * sizeof(float) +
                     2 * 8 * sizeof(int),
                 cuda_stream>>>(
                  reinterpret_cast<const Data*>(logits.data()), row_top8,
                  static_cast<int>(block_size),
                  static_cast<int64_t>(tokens));
          CloverCoresetMassKernel<Backend::kDeviceType>
              <<<static_cast<unsigned>(blocks_), kCloverCoresetExperts, 0,
                 cuda_stream>>>(
                  row_top8, mass, static_cast<int64_t>(tokens),
                  static_cast<int64_t>(blocks_),
                  static_cast<int>(block_size));

          CloverCoresetAllowKernel<Backend::kDeviceType>
              <<<static_cast<unsigned>(blocks_), kCloverCoresetExperts,
                 kCloverCoresetExperts * (sizeof(float) + sizeof(int)),
                 cuda_stream>>>(
                  mass, allow, static_cast<int64_t>(blocks_),
                  static_cast<float>(top_p));

          CloverCoresetApplyKernel<Backend::kDeviceType, Data>
              <<<static_cast<unsigned>(tokens), 256, 0, cuda_stream>>>(
                  reinterpret_cast<const Data*>(logits.data()), allow,
                  reinterpret_cast<Data*>(out.data()),
                  static_cast<int>(block_size), static_cast<int64_t>(tokens));
        },
        "CudaCloverCoreset::operator()");
  }

 private:
  int64_t blocks_;
  int64_t tokens_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_CORESET_KERNEL_H_
