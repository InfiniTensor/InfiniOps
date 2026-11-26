#include <infini/ops.h>

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace infini::ops;
int main() {
  size_t checks = 0;
  try {
    const Device cpu{Device::Type::kCpu, 0};
    Config config;
    config.set_implementation_index(0);
    Handle handle;
    std::array<float, 16> input{}, output{};
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(i) - 3;
    const auto original = input;
    auto call = [&](const Tensor& x, const Tensor& y) {
      Cast::Call(handle, config, x, y);
    };
    Tensor x(input.data(), {2, 3}, DataType::kFloat32, cpu, {5, 1});
    Tensor y(output.data(), {6}, DataType::kFloat32, cpu, {2});
    output.fill(-999);
    call(x, y);
    for (size_t i = 0; i < output.size(); ++i) {
      float expected =
          i < 12 && i % 2 == 0 ? input[(i / 2) / 3 * 5 + (i / 2) % 3] : -999;
      if (output[i] != expected)
        throw std::runtime_error("strided rank-changing Cast mismatch");
    }
    ++checks;
    call(x, x);
    if (input != original)
      throw std::runtime_error("in-place identity changed input");
    ++checks;
    auto rejected = [&](const Tensor& src, const Tensor& dst) {
      const auto before = input, before_out = output;
      bool caught = false;
      try {
        call(src, dst);
      } catch (const std::invalid_argument&) {
        caught = true;
      }
      if (!caught || std::memcmp(input.data(), before.data(), sizeof(input)) ||
          std::memcmp(output.data(), before_out.data(), sizeof(output)))
        throw std::runtime_error(
            "invalid Cast was not rejected before writing");
      ++checks;
    };
    Tensor flat(input.data(), {4}, DataType::kFloat32, cpu);
    rejected(flat, Tensor(input.data(), {4}, DataType::kFloat16, cpu));
    rejected(flat, Tensor(input.data() + 1, {4}, DataType::kFloat16, cpu));
    rejected(flat, Tensor(input.data() + 1, {4}, DataType::kFloat32, cpu));
    rejected(flat, Tensor(output.data(), {4}, DataType::kFloat32, cpu, {0}));
    rejected(flat, Tensor(output.data(), {3}, DataType::kFloat32, cpu));
    std::array<uint16_t, 4> half{};
    call(flat, Tensor(half.data(), {4}, DataType::kFloat16, cpu));
    if (half != std::array<uint16_t, 4>{0xc200, 0xc000, 0xbc00, 0})
      throw std::runtime_error("FP16 rounding changed");
    ++checks;
    Tensor empty_in(nullptr, {0, 4}, DataType::kFloat32, cpu),
        empty_out(nullptr, {0}, DataType::kFloat16, cpu);
    call(empty_in, empty_out);
    ++checks;
    std::cout << "{\"passed\":true,\"checks\":" << checks << "}\n";
  } catch (const std::exception&) {
    std::cout << "{\"passed\":false,\"checks\":" << checks << "}\n";
    return 1;
  }
}
