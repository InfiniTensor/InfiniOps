#include <infini/ops.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

using infini::ops::Cast;
using infini::ops::Config;
using infini::ops::DataType;
using infini::ops::Device;
using infini::ops::Handle;
using infini::ops::Tensor;

namespace {

void Check(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

template <class F>
void Rejected(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "Expected rejection before accessing tensor storage.");
}

}  // namespace

int main() {
  const Device cpu{Device::Type::kCpu, 0};
  Config config;
  config.set_implementation_index(0);
  std::array<float, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 8> output{};
  const auto x = Tensor(input.data(), {4}, DataType::kFloat32, cpu);
  auto y = Tensor(output.data(), {4}, DataType::kFloat32, cpu);
  auto op = Cast::Make(config, x, y);
  (*op)(Handle{}, x, y);
  Check(output[3] == 4, "The valid descriptor did not copy its input.");
  const auto before_input = input;
  const auto before_output = output;

  Rejected([&] {
    (*op)(Handle{}, x, Tensor(input.data() + 1, {4}, DataType::kFloat32, cpu));
  });
  Rejected([&] {
    (*op)(Handle{}, x, Tensor(output.data(), {4}, DataType::kInt32, cpu));
  });
  Rejected([&] {
    (*op)(Handle{}, x, Tensor(output.data(), {2, 2}, DataType::kFloat32, cpu));
  });
  Rejected([&] {
    (*op)(Handle{}, x,
          Tensor(output.data(), {4}, DataType::kFloat32, cpu, {2}));
  });
  Check(input == before_input && output == before_output,
        "Rejected descriptor reuse modified storage.");

  const auto unsupported = [&](Tensor view) {
    Check(!Cast::CanMake(config, view, y),
          "Invalid storage was reported as supported.");
    Rejected([&] { Cast::Make(config, view, y); });
  };
  unsupported(Tensor(nullptr, {4}, DataType::kFloat32, cpu));
  unsupported(Tensor(reinterpret_cast<char*>(input.data()) + 1, {4},
                     DataType::kFloat32, cpu));
  unsupported(Tensor(
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 3),
      {4}, DataType::kFloat32, cpu));
  unsupported(Tensor(input.data(), {4}, DataType::kFloat32, cpu, {-1}));
  unsupported(Tensor(input.data(), {4}, DataType::kFloat32, cpu,
                     {std::numeric_limits<std::ptrdiff_t>::max()}));

  // Empty tensors need no storage, while rank-zero tensors contain one value.
  auto empty_in = Tensor(nullptr, {0, 3}, DataType::kFloat32, cpu);
  auto empty_out = Tensor(nullptr, {0, 3}, DataType::kFloat16, cpu);
  Check(Cast::CanMake(config, empty_in, empty_out),
        "An empty tensor unexpectedly required storage.");
  Cast::Call(Handle{}, config, empty_in, empty_out);
  auto scalar_in = Tensor(input.data(), {}, DataType::kFloat32, cpu);
  auto scalar_out = Tensor(output.data(), {}, DataType::kFloat32, cpu);
  Cast::Call(Handle{}, config, scalar_in, scalar_out);
  Check(output[0] == input[0], "Scalar conversion failed.");
  std::puts("CPU Cast descriptor and storage checks passed.");
}
