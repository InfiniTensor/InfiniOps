#include <infini/ops.h>
#include <infini/rt/device_guard.h>
#ifdef WITH_HYGON
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace infini::ops;
#ifdef WITH_HYGON
static void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error("GPU operation failed");
}
struct Buffer {
  void* pointer = nullptr;
  explicit Buffer(size_t bytes) {
    if (bytes) checked(cudaMalloc(&pointer, bytes));
  }
  ~Buffer() {
    if (pointer) cudaFree(pointer);
  }
};
#endif
int main() {
  size_t checks = 0;
  try {
    std::vector<Device::Type> devices{Device::Type::kCpu};
#ifdef WITH_HYGON
    devices.push_back(Device::Type::kHygon);
#endif
    for (auto type : devices) {
      Device device{type, 0};
      infini::rt::DeviceGuard guard{type, 0, true};
      for (int layout = 0; layout < 4; ++layout) {
        Tensor::Shape shape = layout == 0   ? Tensor::Shape{17, 1}
                              : layout == 1 ? Tensor::Shape{2, 3}
                              : layout == 2 ? Tensor::Shape{}
                                            : Tensor::Shape{0, 4};
        Tensor::Strides xs = layout == 0   ? Tensor::Strides{1, 1}
                             : layout == 1 ? Tensor::Strides{5, 1}
                             : layout == 2 ? Tensor::Strides{}
                                           : Tensor::Strides{4, 1};
        Tensor::Strides ys = layout == 1 ? Tensor::Strides{1, 2} : xs;
        for (int seed = 0; seed < 2; ++seed) {
          std::vector<float> input(32), output(32, -999.0f), expected = output;
          for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(int(i) - 9 + seed) / 8;
          void* xp = input.data();
          void* yp = output.data();
#ifdef WITH_HYGON
          const bool gpu = type == Device::Type::kHygon;
          Buffer xgpu(gpu ? 128 : 0), ygpu(gpu ? 128 : 0);
          if (gpu) {
            xp = xgpu.pointer;
            yp = ygpu.pointer;
          }
#endif
          for (double alpha : {0.5, -2.0, 0.5}) {
            std::fill(output.begin(), output.end(), -999.0f);
            expected = output;
#ifdef WITH_HYGON
            if (gpu) {
              checked(
                  cudaMemcpy(xp, input.data(), 128, cudaMemcpyHostToDevice));
              checked(
                  cudaMemcpy(yp, output.data(), 128, cudaMemcpyHostToDevice));
            }
#endif
            Tensor x(xp, shape, DataType::kFloat32, device, xs);
            Tensor y(yp, shape, DataType::kFloat32, device, ys);
            Handle handle;
            Config config;
            config.set_implementation_index(8);
            Mul::Call(handle, config, x, alpha, y);
#ifdef WITH_HYGON
            if (gpu)
              checked(
                  cudaMemcpy(output.data(), yp, 128, cudaMemcpyDeviceToHost));
#endif
            const size_t rows = layout == 0   ? 17
                                : layout == 1 ? 2
                                : layout == 2 ? 1
                                              : 0;
            const size_t cols = layout == 1 ? 3 : 1;
            for (size_t r = 0; r < rows; ++r)
              for (size_t c = 0; c < cols; ++c) {
                const auto ix = layout == 2 ? 0 : r * xs[0] + c * xs[1];
                const auto iy = layout == 2 ? 0 : r * ys[0] + c * ys[1];
                expected[iy] = input[ix] * static_cast<float>(alpha);
              }
            if (output != expected)
              throw std::runtime_error(
                  "scalar Mul result or output canary mismatch");
            ++checks;
          }
        }
      }
    }
    std::cout << "{\"passed\":true,\"checks\":" << checks << "}\n";
  } catch (const std::exception&) {
    std::cout << "{\"passed\":false,\"checks\":" << checks << "}\n";
    return 1;
  }
}
