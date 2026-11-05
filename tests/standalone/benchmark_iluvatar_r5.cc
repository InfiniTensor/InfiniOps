#include <cuda_runtime_api.h>
#include <infini/ops.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using infini::ops::DataType;
using infini::ops::Tensor;
const infini::ops::Device kDevice{infini::ops::Device::Type::kIluvatar, 0};

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("COREX runtime call failed: ") +
                             cudaGetErrorString(status));
  }
}

class Stream {
 public:
  Stream() { Check(cudaStreamCreate(&value)); }
  ~Stream() { cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t value{};
};

class Buffer {
 public:
  explicit Buffer(size_t bytes) : bytes_(bytes) {
    Check(cudaMalloc(&data_, bytes));
  }
  ~Buffer() { cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void* data() const { return data_; }
  void Upload(const void* host) {
    Check(cudaMemcpy(data_, host, bytes_, cudaMemcpyHostToDevice));
  }
  void Download(void* host) const {
    Check(cudaMemcpy(host, data_, bytes_, cudaMemcpyDeviceToHost));
  }

 private:
  void* data_{};
  size_t bytes_;
};

class Event {
 public:
  Event() { Check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t value{};
};

float ToFloat(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t ToBfloat(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

std::vector<uint16_t> Values(size_t size, size_t offset = 0) {
  std::vector<uint16_t> values(size);
  for (size_t i = 0; i < size; ++i) {
    values[i] = ToBfloat((static_cast<int>((i + offset) % 29) - 14) / 16.0f);
  }
  return values;
}

uint64_t Checksum(const std::vector<uint16_t>& values) {
  uint64_t hash = 14695981039346656037ull;
  for (uint16_t value : values) {
    hash = (hash ^ (value & 255)) * 1099511628211ull;
    hash = (hash ^ (value >> 8)) * 1099511628211ull;
  }
  return hash;
}

float Validate(const std::vector<uint16_t>& actual,
               const std::vector<float>& expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("Size mismatch.");
  float largest = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(ToFloat(actual[i]) - expected[i]);
    if (!std::isfinite(error) ||
        error > 1e-6f + 0.01f * std::abs(expected[i])) {
      throw std::runtime_error("Synthetic numerical reference check failed.");
    }
    largest = std::max(largest, error);
  }
  return largest;
}

void Measure(const char* operation, size_t rows, size_t width,
             cudaStream_t stream, const std::function<void()>& call,
             uint64_t checksum, float max_abs_error) {
  constexpr int kIterations = 20;
  constexpr int kRepeats = 5;
  for (int i = 0; i < 3; ++i) call();
  Check(cudaStreamSynchronize(stream));
  std::vector<double> wall, device;
  Event begin, end;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    const auto started = std::chrono::steady_clock::now();
    Check(cudaEventRecord(begin.value, stream));
    for (int i = 0; i < kIterations; ++i) call();
    Check(cudaEventRecord(end.value, stream));
    Check(cudaEventSynchronize(end.value));
    wall.push_back(std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - started)
                       .count() /
                   kIterations);
    float ms = 0;
    Check(cudaEventElapsedTime(&ms, begin.value, end.value));
    device.push_back(ms * 1000 / kIterations);
  }
  auto sorted = wall;
  std::sort(sorted.begin(), sorted.end());
  std::cout
      << "{\"operation\":\"" << operation << "\",\"shape\":[" << rows << ','
      << width
      << "],\"dtype\":\"bfloat16\",\"warmup\":3,\"iterations\":20,\"repeat\":5,"
      << "\"output_checksum_fnv1a64\":\"" << checksum
      << "\",\"max_abs_error\":" << max_abs_error
      << ",\"median_wall_us\":" << sorted[kRepeats / 2] << ",\"wall_us\":[";
  for (size_t i = 0; i < wall.size(); ++i)
    std::cout << (i ? "," : "") << wall[i];
  std::cout << "],\"device_interval_us\":[";
  for (size_t i = 0; i < device.size(); ++i)
    std::cout << (i ? "," : "") << device[i];
  std::cout << "]}" << std::endl;
}

void Run(const std::string& operation, size_t rows, size_t width,
         cudaStream_t stream) {
  const bool swiglu = operation == "swiglu";
  const size_t input_width = swiglu ? 2 * width : width;
  auto x = Values(rows * input_width), other = Values(rows * width, 7);
  std::vector<uint16_t> y(rows * width);
  std::vector<float> reference(y.size());
  Buffer x_buffer(x.size() * 2), other_buffer(other.size() * 2),
      y_buffer(y.size() * 2);
  x_buffer.Upload(x.data());
  other_buffer.Upload(other.data());
  const Tensor input{
      x_buffer.data(), {rows, input_width}, DataType::kBFloat16, kDevice};
  const Tensor rhs{
      other_buffer.data(), {rows, width}, DataType::kBFloat16, kDevice};
  const Tensor output{
      y_buffer.data(), {rows, width}, DataType::kBFloat16, kDevice};
  infini::ops::Handle handle;
  handle.set_stream(stream);
  const auto config = infini::ops::Config::Automatic();
  const auto call = [&] {
    if (operation == "softmax")
      infini::ops::Softmax::Call(handle, config, input, int64_t{-1},
                                 std::optional<DataType>{}, output);
    else if (swiglu)
      infini::ops::SiluAndMul::Call(handle, config, input, output);
    else
      infini::ops::Add::Call(handle, config, input, rhs, output);
  };
  call();
  Check(cudaStreamSynchronize(stream));
  y_buffer.Download(y.data());
  for (size_t row = 0; row < rows; ++row) {
    float sum = 0;
    if (operation == "softmax") {
      for (size_t column = 0; column < width; ++column)
        sum += std::exp(ToFloat(x[row * width + column]));
    }
    for (size_t column = 0; column < width; ++column) {
      const float value = ToFloat(x[row * input_width + column]);
      reference[row * width + column] =
          operation == "softmax" ? std::exp(value) / sum
          : swiglu               ? value / (1 + std::exp(-value)) *
                         ToFloat(x[row * input_width + width + column])
                   : value + ToFloat(other[row * width + column]);
    }
  }
  Measure(operation.c_str(), rows, width, stream, call, Checksum(y),
          Validate(y, reference));
}
}  // namespace

int main() {
  try {
    const char* prefix = std::getenv("INFINI_ROOT");
    if (!prefix)
      throw std::runtime_error("Set `INFINI_ROOT` to the selected SDK.");
    const std::string expected = std::string(prefix) + "/lib/libinfiniops.so";
    std::ifstream maps("/proc/self/maps");
    bool found = false;
    for (std::string line; std::getline(maps, line);) {
      if (line.find("/libinfiniops.so") == std::string::npos) continue;
      if (line.find(expected) == std::string::npos)
        throw std::runtime_error("An unexpected InfiniOps library is loaded.");
      found = true;
    }
    if (!found)
      throw std::runtime_error("The selected InfiniOps library was not found.");
    Check(cudaSetDevice(0));
    Stream stream;
    for (const auto& shape :
         std::vector<std::vector<size_t>>{{16, 128}, {512, 512}, {2048, 2048}})
      Run("softmax", shape[0], shape[1], stream.value);
    for (size_t rows : {4, 512, 2048}) Run("swiglu", rows, 18944, stream.value);
    for (size_t rows : {16, 2048}) Run("add", rows, 3584, stream.value);
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  return 0;
}
