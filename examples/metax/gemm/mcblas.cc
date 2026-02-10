#include "metax/gemm/mcblas.h"

#include <mcr/mc_runtime.h>

#include <iostream>
#include <numeric>

#include "tensor.h"

int main() {
  using namespace infini::ops;

  constexpr auto m{2};
  constexpr auto k{3};
  constexpr auto n{4};

  std::vector<Tensor::Size> a_shape{m, k};
  std::vector<Tensor::Size> b_shape{k, n};
  std::vector<Tensor::Size> c_shape{m, n};

  const auto a_num_elements{std::accumulate(a_shape.cbegin(), a_shape.cend(), 1,
                                            std::multiplies<int>())};
  const auto b_num_elements{std::accumulate(b_shape.cbegin(), b_shape.cend(), 1,
                                            std::multiplies<int>())};
  const auto c_num_elements{std::accumulate(c_shape.cbegin(), c_shape.cend(), 1,
                                            std::multiplies<int>())};

  std::vector<float> a_vec(a_num_elements);
  std::vector<float> b_vec(b_num_elements);
  std::vector<float> c_vec(c_num_elements);

  std::iota(a_vec.begin(), a_vec.end(), 0);
  std::iota(b_vec.begin(), b_vec.end(), 0);

  Tensor a_host{a_vec.data(), a_shape, Device{Device::Type::kMetax}};
  Tensor b_host{b_vec.data(), b_shape, Device{Device::Type::kMetax}};
  Tensor c_host{c_vec.data(), c_shape, Device{Device::Type::kMetax}};

  Tensor a_device{nullptr, a_host.shape(), a_host.dtype(), a_host.device(),
                  a_host.strides()};
  Tensor b_device{nullptr, b_host.shape(), b_host.dtype(), a_host.device(),
                  b_host.strides()};
  Tensor c_device{nullptr, c_host.shape(), c_host.dtype(), a_host.device(),
                  c_host.strides()};

  const auto a_size{a_num_elements * a_device.dtype().element_size()};
  const auto b_size{b_num_elements * b_device.dtype().element_size()};
  const auto c_size{c_num_elements * c_device.dtype().element_size()};

  mcMalloc(&a_device.data(), a_size);
  mcMalloc(&b_device.data(), b_size);
  mcMalloc(&c_device.data(), c_size);

  mcMemcpy(a_device.data(), a_vec.data(), a_size, mcMemcpyHostToDevice);
  mcMemcpy(b_device.data(), b_vec.data(), b_size, mcMemcpyHostToDevice);
  mcMemset(c_device.data(), 0, c_size);

  Gemm::call(nullptr, a_device, b_device, c_device);

  mcMemcpy(a_host.data(), a_device.data(), a_size, mcMemcpyDeviceToHost);
  mcMemcpy(b_host.data(), b_device.data(), b_size, mcMemcpyDeviceToHost);
  mcMemcpy(c_host.data(), c_device.data(), c_size, mcMemcpyDeviceToHost);

  mcFree(a_device.data());
  mcFree(b_device.data());
  mcFree(c_device.data());

  std::cout << "A: " << a_host.ToString() << "\n";
  std::cout << "B: " << b_host.ToString() << "\n";
  std::cout << "C: " << c_host.ToString() << "\n";

  return 0;
}
