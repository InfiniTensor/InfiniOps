#include <iostream>
#include <numeric>
#include <variant>

#ifdef USE_CPU
#include "cpu/gemm/gemm.h"
#elif USE_NVIDIA
#include "nvidia/gemm/cublas.h"
#elif USE_METAX
#include "metax/gemm/mcblas.h"
#endif

#include "../runtime_api.h"
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

  Device dev{DEFAULT_DEVICE_TYPE};

  Tensor a_host{a_vec.data(), a_shape, dev};
  Tensor b_host{b_vec.data(), b_shape, dev};
  Tensor c_host{c_vec.data(), c_shape, dev};

  const auto a_size{a_num_elements * kDataTypeToSize.at(a_host.dtype())};
  const auto b_size{b_num_elements * kDataTypeToSize.at(b_host.dtype())};
  const auto c_size{c_num_elements * kDataTypeToSize.at(c_host.dtype())};

  void *a_ptr, *b_ptr, *c_ptr;

#ifdef USE_CPU
  a_ptr = a_vec.data();
  b_ptr = b_vec.data();
  c_ptr = c_vec.data();
#else
  DEVICE_MALLOC(&a_ptr, a_size);
  DEVICE_MALLOC(&b_ptr, b_size);
  DEVICE_MALLOC(&c_ptr, c_size);

  DEVICE_MEMCPY(a_ptr, a_vec.data(), a_size, DEVICE_MEMCPY_HOST_TO_DEVICE);
  DEVICE_MEMCPY(b_ptr, b_vec.data(), b_size, DEVICE_MEMCPY_HOST_TO_DEVICE);
  DEVICE_MEMSET(c_ptr, 0, c_size);
#endif

  Tensor a_device{a_ptr, a_host.shape(), a_host.dtype(), a_host.device(),
                  a_host.strides()};
  Tensor b_device{b_ptr, b_host.shape(), b_host.dtype(), a_host.device(),
                  b_host.strides()};
  Tensor c_device{c_ptr, c_host.shape(), c_host.dtype(), a_host.device(),
                  c_host.strides()};

  Gemm::call(nullptr, a_device, b_device, c_device);

#ifndef USE_CPU
  DEVICE_MEMCPY(c_vec.data(), c_ptr, c_size, DEVICE_MEMCPY_DEVICE_TO_HOST);
  DEVICE_FREE(a_ptr);
  DEVICE_FREE(b_ptr);
  DEVICE_FREE(c_ptr);
#endif

  std::cout << "A: " << a_host.ToString() << "\n";
  std::cout << "B: " << b_host.ToString() << "\n";
  std::cout << "C: " << c_host.ToString() << "\n";

  return 0;
}
