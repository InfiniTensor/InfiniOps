#include "data_type.h"

#include <iomanip>
#include <iostream>
#include <vector>

static void PrintDataTypeInfo(const infini::ops::DataType& dtype) {}

int main() {
  using namespace infini::ops;

  static const std::vector<DataType> kDataTypes{
      kInt8,   kInt16,  kInt32,   kInt64,    kUInt8,   kUInt16,
      kUInt32, kUInt64, kFloat16, kBFloat16, kFloat32, kFloat64};

  std::cout << std::left << std::setw(10) << "Name" << std::left
            << std::setw(10) << "Element Size\n";

  for (const auto& dtype : kDataTypes) {
    std::cout << std::left << std::setw(10) << dtype.name() << std::left
              << std::setw(10) << dtype.element_size() << '\n';
  }

  return 0;
}
