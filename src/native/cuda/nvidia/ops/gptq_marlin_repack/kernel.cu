// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM at commit ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab:
// csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>

#include "native/cuda/nvidia/ops/gptq_marlin_repack/kernel.cuh"
#include "native/cuda/nvidia/ops/gptq_marlin_repack/kernel.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`GptqMarlinRepack` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`GptqMarlinRepack` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`GptqMarlinRepack` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

template <int kNumBits, bool kHasPerm, bool kIsA8Bit>
void Launch(const uint32_t* b_q_weight, const uint32_t* perm, uint32_t* out,
            int size_k, int size_n, int blocks, int shared_memory_bytes,
            cudaStream_t stream) {
  const auto attribute_status = cudaFuncSetAttribute(
      gptq_marlin_repack_detail::GptqMarlinRepackKernel<kNumBits, kHasPerm,
                                                        kIsA8Bit>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
  assert(attribute_status == cudaSuccess &&
         "`GptqMarlinRepack` failed to configure dynamic shared memory");

  gptq_marlin_repack_detail::GptqMarlinRepackKernel<kNumBits, kHasPerm,
                                                    kIsA8Bit>
      <<<blocks, gptq_marlin_repack_detail::kRepackThreads, shared_memory_bytes,
         stream>>>(b_q_weight, perm, out, size_k, size_n);
}

}  // namespace

void Operator<GptqMarlinRepack, Device::Type::kNvidia, 0>::operator()(
    const Tensor b_q_weight, const Tensor perm, const int64_t size_k,
    const int64_t size_n, const int64_t num_bits, const bool is_a_8bit,
    Tensor out) const {
  ValidateCallMetadata(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit,
                       out);

  DeviceGuard device_guard{device_index_};
  int blocks = 0;
  auto status = cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount,
                                       device_index_);
  assert(status == cudaSuccess && blocks > 0 &&
         "`GptqMarlinRepack` failed to query CUDA multiprocessor count");

  int shared_memory_bytes = 0;
  status = cudaDeviceGetAttribute(&shared_memory_bytes,
                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                  device_index_);
  assert(status == cudaSuccess && shared_memory_bytes > 0 &&
         "`GptqMarlinRepack` failed to query CUDA shared memory capacity");

  const auto* b_q_weight_ptr =
      reinterpret_cast<const uint32_t*>(b_q_weight.data());
  const auto* perm_ptr =
      has_perm_ ? reinterpret_cast<const uint32_t*>(perm.data()) : nullptr;
  auto* out_ptr = reinterpret_cast<uint32_t*>(out.data());
  const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
  const auto kernel_size_k = static_cast<int>(size_k_);
  const auto kernel_size_n = static_cast<int>(size_n_);

  if (is_a_8bit_) {
    if (num_bits_ == 4) {
      Launch<4, false, true>(b_q_weight_ptr, perm_ptr, out_ptr, kernel_size_k,
                             kernel_size_n, blocks, shared_memory_bytes,
                             stream);
    } else {
      Launch<8, false, true>(b_q_weight_ptr, perm_ptr, out_ptr, kernel_size_k,
                             kernel_size_n, blocks, shared_memory_bytes,
                             stream);
    }
  } else if (has_perm_) {
    if (num_bits_ == 4) {
      Launch<4, true, false>(b_q_weight_ptr, perm_ptr, out_ptr, kernel_size_k,
                             kernel_size_n, blocks, shared_memory_bytes,
                             stream);
    } else {
      Launch<8, true, false>(b_q_weight_ptr, perm_ptr, out_ptr, kernel_size_k,
                             kernel_size_n, blocks, shared_memory_bytes,
                             stream);
    }
  } else if (num_bits_ == 4) {
    Launch<4, false, false>(b_q_weight_ptr, perm_ptr, out_ptr, kernel_size_k,
                            kernel_size_n, blocks, shared_memory_bytes, stream);
  } else {
    Launch<8, false, false>(b_q_weight_ptr, perm_ptr, out_ptr, kernel_size_k,
                            kernel_size_n, blocks, shared_memory_bytes, stream);
  }

  status = cudaGetLastError();
  assert(status == cudaSuccess &&
         "`GptqMarlinRepack` CUDA kernel launch failed");
}

}  // namespace infini::ops
