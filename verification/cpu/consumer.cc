#include <infini/ops.h>
#include <link.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "returning_support.h"

bool CheckReturningCalls() {
  OwningTensor x({1.0f, -2.0f, 3.0f}, {3});
  OwningTensor y({4.0f, 5.0f, -6.0f}, {3});
  auto sum = infini::ops::Add::Call(x, y);
  if (sum.shape() != OwningTensor::Shape{3}) return false;
  const auto* sum_data = static_cast<const float*>(sum.data());
  const float expected_sum[] = {5.0f, 3.0f, -3.0f};
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(sum_data[i]) || sum_data[i] != expected_sum[i])
      return false;
  }
  OwningTensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  OwningTensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  auto product = infini::ops::Gemm::Call(a, b);
  if (product.shape() != OwningTensor::Shape{2, 2}) return false;
  const auto* product_data = static_cast<const float*>(product.data());
  const float expected_product[] = {19.0f, 22.0f, 43.0f, 50.0f};
  for (int i = 0; i < 4; ++i) {
    if (!std::isfinite(product_data[i]) ||
        product_data[i] != expected_product[i])
      return false;
  }
  return true;
}

int CollectLibraries(dl_phdr_info* info, std::size_t, void* opaque) {
  const std::string name = info->dlpi_name;
  if (name.find("libinfiniops") != std::string::npos ||
      name.find("libinfinirt") != std::string::npos) {
    static_cast<std::vector<std::string>*>(opaque)->push_back(name);
  }
  return 0;
}

template <typename Op>
void PrintSelection(const char* name) {
  constexpr auto device = infini::ops::Device::Type::kCpu;
  std::cout << "\"" << name << "\":{\"default\":"
            << Op::DefaultConfig(device).implementation_index()
            << ",\"active\":[";
  bool first = true;
  for (auto index : Op::active_implementation_indices(device)) {
    if (!first) std::cout << ",";
    std::cout << index;
    first = false;
  }
  std::cout << "]}";
}

int main() {
  if (!CheckReturningCalls()) return 1;
  std::cout << "{\"selection\":{";
  PrintSelection<infini::ops::Add>("add");
  std::cout << ",";
  PrintSelection<infini::ops::Gemm>("gemm");
  std::cout << "},\"loaded_libraries\":[";
  std::vector<std::string> libraries;
  dl_iterate_phdr(CollectLibraries, &libraries);
  for (std::size_t i = 0; i < libraries.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << std::quoted(libraries[i]);
  }
  std::cout << "]}\n";
}
