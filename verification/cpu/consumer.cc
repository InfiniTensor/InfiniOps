#include <infini/ops.h>

#include <iostream>

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
  std::cout << "{";
  PrintSelection<infini::ops::Add>("add");
  std::cout << ",";
  PrintSelection<infini::ops::Gemm>("gemm");
  std::cout << "}\n";
}
