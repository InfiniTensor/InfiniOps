#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/gemm.h"
#include "data_type.h"
#include "handle.h"
#include "infiniccl.h"
#include "runtime_api.h"
#include "tensor.h"

#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
#include <cuda_runtime.h>
#elif defined(WITH_METAX)
#include <mcr/mc_runtime.h>
#elif defined(WITH_MOORE)
#include <musa_runtime.h>
#elif defined(WITH_CAMBRICON)
#include <cnrt.h>
#elif defined(WITH_ASCEND)
#include <acl/acl.h>
#endif

namespace {

#define CHECK_CCL(cmd)                                                       \
  do {                                                                       \
    auto status = (cmd);                                                     \
    if (status != infinicclSuccess) {                                        \
      throw std::runtime_error(std::string("InfiniCCL failure at line ") +   \
                               std::to_string(__LINE__));                   \
    }                                                                        \
  } while (0)

template <typename Status>
void CheckRuntime(Status status, const char *expr, int line) {
#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + " failed at line " +
                             std::to_string(line) + ": " +
                             cudaGetErrorString(status));
  }
#elif defined(WITH_METAX)
  if (status != mcSuccess) {
    throw std::runtime_error(std::string(expr) + " failed at line " +
                             std::to_string(line) + " status=" +
                             std::to_string(static_cast<long long>(status)));
  }
#elif defined(WITH_MOORE)
  if (status != musaSuccess) {
    throw std::runtime_error(std::string(expr) + " failed at line " +
                             std::to_string(line) + ": " +
                             musaGetErrorString(status));
  }
#elif defined(WITH_CAMBRICON)
  if (status != cnrtSuccess) {
    throw std::runtime_error(std::string(expr) + " failed at line " +
                             std::to_string(line) + " status=" +
                             std::to_string(static_cast<long long>(status)));
  }
#elif defined(WITH_ASCEND)
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(expr) + " failed at line " +
                             std::to_string(line) + " status=" +
                             std::to_string(static_cast<long long>(status)));
  }
#else
  (void)status;
  (void)expr;
  (void)line;
#endif
}

#define CHECK_RT(cmd) CheckRuntime((cmd), #cmd, __LINE__)

int LocalRank() {
  const char *names[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                         "SLURM_LOCALID", "LOCAL_RANK"};
  for (auto *name : names) {
    const char *value = std::getenv(name);
    if (value) return std::atoi(value);
  }
  return 0;
}

float MakeA(int global_row, int k) {
  return static_cast<float>((global_row % 13) - 6) * 0.01f +
         static_cast<float>((k % 7) + 1) * 0.001f;
}

float MakeB(int k, int col) {
  return static_cast<float>(((k + col) % 17) - 8) * 0.02f;
}

float Reference(int row, int col, int k_size) {
  float acc = 0.0f;
  for (int k = 0; k < k_size; ++k) acc += MakeA(row, k) * MakeB(k, col);
  return acc;
}

class DeviceContext {
 public:
  explicit DeviceContext(int local_rank) : local_rank_(local_rank) {
#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
    CHECK_RT(cudaSetDevice(local_rank));
    CHECK_RT(cudaStreamCreate(&stream_));
#elif defined(WITH_METAX)
    CHECK_RT(mcSetDevice(local_rank));
    CHECK_RT(mcStreamCreate(&stream_));
#elif defined(WITH_MOORE)
    CHECK_RT(musaSetDevice(local_rank));
    CHECK_RT(musaStreamCreate(&stream_));
#elif defined(WITH_CAMBRICON)
    CHECK_RT(cnrtSetDevice(local_rank));
    CHECK_RT(cnrtQueueCreate(&stream_));
#elif defined(WITH_ASCEND)
    CHECK_RT(aclInit(nullptr));
    CHECK_RT(aclrtSetDevice(local_rank));
    CHECK_RT(aclrtCreateStream(&stream_));
#else
    (void)local_rank;
#endif
  }

  ~DeviceContext() {
#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
    if (stream_) cudaStreamDestroy(stream_);
#elif defined(WITH_METAX)
    if (stream_) mcStreamDestroy(stream_);
#elif defined(WITH_MOORE)
    if (stream_) musaStreamDestroy(stream_);
#elif defined(WITH_CAMBRICON)
    if (stream_) cnrtQueueDestroy(stream_);
#elif defined(WITH_ASCEND)
    if (stream_) aclrtDestroyStream(stream_);
    aclrtResetDevice(local_rank_);
    aclFinalize();
#endif
  }

  void *stream() const { return reinterpret_cast<void *>(stream_); }

  void Synchronize() {
#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
    CHECK_RT(cudaStreamSynchronize(stream_));
#elif defined(WITH_METAX)
    CHECK_RT(mcStreamSynchronize(stream_));
#elif defined(WITH_MOORE)
    CHECK_RT(musaStreamSynchronize(stream_));
#elif defined(WITH_CAMBRICON)
    CHECK_RT(cnrtQueueSync(stream_));
#elif defined(WITH_ASCEND)
    CHECK_RT(aclrtSynchronizeStream(stream_));
#endif
  }

 private:
  int local_rank_{0};
#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
  cudaStream_t stream_{nullptr};
#elif defined(WITH_METAX)
  mcStream_t stream_{nullptr};
#elif defined(WITH_MOORE)
  musaStream_t stream_{nullptr};
#elif defined(WITH_CAMBRICON)
  cnrtQueue_t stream_{nullptr};
#elif defined(WITH_ASCEND)
  aclrtStream stream_{nullptr};
#else
  void *stream_{nullptr};
#endif
};

const char *BackendName() {
#if defined(WITH_NVIDIA)
  return "nvidia";
#elif defined(WITH_ILUVATAR)
  return "iluvatar";
#elif defined(WITH_METAX)
  return "metax";
#elif defined(WITH_MOORE)
  return "moore";
#elif defined(WITH_CAMBRICON)
  return "cambricon";
#elif defined(WITH_ASCEND)
  return "ascend";
#else
  return "cpu";
#endif
}

}  // namespace

int main(int argc, char **argv) {
  try {
    CHECK_CCL(infinicclInit(&argc, &argv));

    int rank = 0;
    int world = 1;
    CHECK_CCL(infinicclGetRank(&rank));
    CHECK_CCL(infinicclGetSize(&world));

    const int rows_per_rank = argc > 1 ? std::atoi(argv[1]) : 64;
    const int k_size = argc > 2 ? std::atoi(argv[2]) : 128;
    const int n_size = argc > 3 ? std::atoi(argv[3]) : 96;
    if (rows_per_rank <= 0 || k_size <= 0 || n_size <= 0) {
      throw std::runtime_error("matrix sizes must be positive");
    }

    const int local_rank = LocalRank();
    DeviceContext device(local_rank);

    char host[256];
    gethostname(host, sizeof(host));
    std::cout << "[rank " << rank << "/" << world << "] host=" << host
              << " local_rank=" << local_rank
              << " backend=" << BackendName() << std::endl;

    std::vector<float> h_a(static_cast<size_t>(rows_per_rank) * k_size);
    std::vector<float> h_b(static_cast<size_t>(k_size) * n_size);
    std::vector<float> h_c(static_cast<size_t>(rows_per_rank) * n_size, 0.0f);

    for (int i = 0; i < rows_per_rank; ++i) {
      int global_row = rank * rows_per_rank + i;
      for (int k = 0; k < k_size; ++k) {
        h_a[static_cast<size_t>(i) * k_size + k] = MakeA(global_row, k);
      }
    }
    for (int k = 0; k < k_size; ++k) {
      for (int j = 0; j < n_size; ++j) {
        h_b[static_cast<size_t>(k) * n_size + j] = MakeB(k, j);
      }
    }

    using Rt = infini::ops::DefaultRuntimeUtils;
    const size_t a_bytes = h_a.size() * sizeof(float);
    const size_t b_bytes = h_b.size() * sizeof(float);
    const size_t c_bytes = h_c.size() * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CHECK_RT(Rt::Malloc(reinterpret_cast<void **>(&d_a), a_bytes));
    CHECK_RT(Rt::Malloc(reinterpret_cast<void **>(&d_b), b_bytes));
    CHECK_RT(Rt::Malloc(reinterpret_cast<void **>(&d_c), c_bytes));

    CHECK_RT(Rt::Memcpy(d_a, h_a.data(), a_bytes, Rt::MemcpyHostToDevice));
    CHECK_RT(Rt::Memcpy(d_b, h_b.data(), b_bytes, Rt::MemcpyHostToDevice));
    CHECK_RT(Rt::Memcpy(d_c, h_c.data(), c_bytes, Rt::MemcpyHostToDevice));

    infini::ops::Handle handle;
    handle.set_stream(device.stream());

    infini::ops::Tensor a(
        d_a, std::vector<size_t>{static_cast<size_t>(rows_per_rank),
                                 static_cast<size_t>(k_size)},
        infini::ops::DataType::kFloat32,
        infini::ops::Device{Rt::kDeviceType, local_rank});
    infini::ops::Tensor b(
        d_b, std::vector<size_t>{static_cast<size_t>(k_size),
                                 static_cast<size_t>(n_size)},
        infini::ops::DataType::kFloat32,
        infini::ops::Device{Rt::kDeviceType, local_rank});
    infini::ops::Tensor c(
        d_c, std::vector<size_t>{static_cast<size_t>(rows_per_rank),
                                 static_cast<size_t>(n_size)},
        infini::ops::DataType::kFloat32,
        infini::ops::Device{Rt::kDeviceType, local_rank});

    infini::ops::Gemm::Call(handle, {}, a, b, std::optional<float>{1.0f},
                            std::optional<float>{0.0f},
                            std::optional<int>{0}, std::optional<int>{0}, c);
    device.Synchronize();
    CHECK_RT(Rt::Memcpy(h_c.data(), d_c, c_bytes, Rt::MemcpyDeviceToHost));
    device.Synchronize();

    float max_local_error = 0.0f;
    for (int i = 0; i < rows_per_rank; ++i) {
      int global_row = rank * rows_per_rank + i;
      for (int j = 0; j < n_size; ++j) {
        float expected = Reference(global_row, j, k_size);
        float actual = h_c[static_cast<size_t>(i) * n_size + j];
        max_local_error = std::max(max_local_error, std::fabs(actual - expected));
      }
    }

    infinicclComm_t comm = nullptr;
    CHECK_CCL(infinicclCommInitAll(&comm, world, nullptr));

    std::vector<float> gathered(static_cast<size_t>(world) * h_c.size());
    CHECK_CCL(infinicclAllGather(h_c.data(), gathered.data(), h_c.size(),
                                 infinicclFloat32, comm, nullptr));

    if (rank == 0) {
      float max_global_error = 0.0f;
      for (int r = 0; r < world; ++r) {
        for (int i = 0; i < rows_per_rank; ++i) {
          int global_row = r * rows_per_rank + i;
          for (int j = 0; j < n_size; ++j) {
            size_t idx = (static_cast<size_t>(r) * rows_per_rank + i) * n_size + j;
            max_global_error = std::max(
                max_global_error,
                std::fabs(gathered[idx] - Reference(global_row, j, k_size)));
          }
        }
      }
      std::cout << "global_shape=[" << world * rows_per_rank << ", " << n_size
                << "] k=" << k_size << " max_error=" << max_global_error
                << " sample_c00=" << gathered[0] << std::endl;
    }

    std::cout << "[rank " << rank << "] local max_error=" << max_local_error
              << std::endl;

    CHECK_CCL(infinicclCommDestroy(comm));
    CHECK_RT(Rt::Free(d_c));
    CHECK_RT(Rt::Free(d_b));
    CHECK_RT(Rt::Free(d_a));

    CHECK_CCL(infinicclFinalize());
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "distributed_matmul failed: " << e.what() << std::endl;
    return 1;
  }
}
