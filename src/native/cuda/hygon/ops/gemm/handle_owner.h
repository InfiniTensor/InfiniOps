#pragma once
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace infini::ops {
// DTK's CUDA driver shim does not expose the HIP primary context through
// cuCtxGetCurrent. Bind handles to the runtime device and owner thread instead.
// As with Runtime, callers must not reset a device while a handle lease lives.
class HygonGemmHandle {
 public:
  static std::shared_ptr<HygonGemmHandle> Acquire(
      int device, cublasMath_t mode = CUBLAS_DEFAULT_MATH) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != device)
      throw std::invalid_argument("Hygon GEMM current device mismatch");
    static thread_local std::map<std::pair<int, cublasMath_t>,
                                 std::weak_ptr<HygonGemmHandle>>
        owners;
    for (auto it = owners.begin(); it != owners.end();) {
      if (it->second.expired())
        it = owners.erase(it);
      else
        ++it;
    }
    const auto key = std::make_pair(device, mode);
    if (auto owner = owners[key].lock()) return owner;
    auto owner =
        std::shared_ptr<HygonGemmHandle>(new HygonGemmHandle(device, mode));
    owners[key] = owner;
    return owner;
  }
  HygonGemmHandle(const HygonGemmHandle&) = delete;
  HygonGemmHandle& operator=(const HygonGemmHandle&) = delete;
  ~HygonGemmHandle() noexcept {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        (previous != device_ && cudaSetDevice(device_) != cudaSuccess)) {
      std::fprintf(stderr, "Hygon GEMM cleanup: cannot select owning device\n");
      return;
    }
    if (cublasDestroy(handle_) != CUBLAS_STATUS_SUCCESS)
      std::fprintf(stderr, "Hygon GEMM cleanup: handle destruction failed\n");
    if (previous != device_ && cudaSetDevice(previous) != cudaSuccess)
      std::fprintf(stderr, "Hygon GEMM cleanup: device restoration failed\n");
  }
  cublasHandle_t get() const {
    int current = -1;
    if (worker_ != std::this_thread::get_id() ||
        cudaGetDevice(&current) != cudaSuccess || current != device_)
      throw std::invalid_argument("Hygon GEMM owner thread/device mismatch");
    return handle_;
  }

 private:
  HygonGemmHandle(int device, cublasMath_t mode)
      : device_(device), worker_(std::this_thread::get_id()) {
    const auto status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Hygon GEMM handle creation failed");
    if (mode != CUBLAS_DEFAULT_MATH &&
        cublasSetMathMode(handle_, mode) != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      throw std::runtime_error("Hygon GEMM math mode setup failed");
    }
  }
  int device_;
  std::thread::id worker_;
  cublasHandle_t handle_{};
};
}  // namespace infini::ops
