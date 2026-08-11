#include "triton/ops/add/jit.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "runtime.h"

namespace infini::ops {

namespace {

using triton::jit::AutoTuningOptions;
using triton::jit::Config;
using triton::jit::Grid;

enum class MetadataSection : std::size_t {
  kOutputShape,
  kInputStrides,
  kOtherStrides,
  kOutputStrides,
  kCount,
};

std::size_t MetadataOffset(MetadataSection section,
                           std::size_t metadata_extent) {
  return static_cast<std::size_t>(section) * metadata_extent;
}

const Config& DefaultConfig() {
  static const Config config{4u, 3u, {{"BLOCK_SIZE", 1024}}};
  return config;
}

std::vector<Config> DefaultAutoTuningCandidates() {
  return {
      {4u, 3u, {{"BLOCK_SIZE", 256}}},
      {4u, 3u, {{"BLOCK_SIZE", 512}}},
      {8u, 4u, {{"BLOCK_SIZE", 1024}}},
      {8u, 4u, {{"BLOCK_SIZE", 2048}}},
  };
}

std::vector<int64_t> BuildMetadata(int ndim, const Tensor::Shape& output_shape,
                                   const Tensor::Strides& input_strides,
                                   const Tensor::Strides& other_strides,
                                   const Tensor::Strides& output_strides) {
  const auto metadata_extent = static_cast<std::size_t>(std::max(ndim, 1));
  std::vector<int64_t> metadata(
      static_cast<std::size_t>(MetadataSection::kCount) * metadata_extent, 0);

  for (int dimension = 0; dimension < ndim; ++dimension) {
    const auto index = static_cast<std::size_t>(dimension);
    metadata[MetadataOffset(MetadataSection::kOutputShape, metadata_extent) +
             index] = static_cast<int64_t>(output_shape[index]);
    metadata[MetadataOffset(MetadataSection::kInputStrides, metadata_extent) +
             index] = static_cast<int64_t>(input_strides[index]);
    metadata[MetadataOffset(MetadataSection::kOtherStrides, metadata_extent) +
             index] = static_cast<int64_t>(other_strides[index]);
    metadata[MetadataOffset(MetadataSection::kOutputStrides, metadata_extent) +
             index] = static_cast<int64_t>(output_strides[index]);
  }

  return metadata;
}

template <Device::Type kDev>
class DeviceBuffer {
 public:
  using Runtime = ::infini::rt::runtime::Runtime<kDev>;

  explicit DeviceBuffer(void* stream_ptr)
      : stream_ptr_(static_cast<typename Runtime::Stream>(stream_ptr)) {}

  DeviceBuffer(const DeviceBuffer&) = delete;

  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ptr_ == nullptr) return;

    const auto status = Runtime::FreeAsync(data_ptr_, stream_ptr_);
    assert(status == Runtime::kSuccess &&
           "Triton JIT `Add` failed to release its metadata buffer.");
    (void)status;
  }

  void Upload(const void* source_ptr, std::size_t size) {
    const auto allocation_status = Runtime::Malloc(&data_ptr_, size);
    assert(allocation_status == Runtime::kSuccess &&
           "Triton JIT `Add` failed to allocate its metadata buffer.");
    if (allocation_status != Runtime::kSuccess) {
      data_ptr_ = nullptr;
      return;
    }

    const auto copy_status = Runtime::Memcpy(data_ptr_, source_ptr, size,
                                             Runtime::kMemcpyHostToDevice);
    assert(copy_status == Runtime::kSuccess &&
           "Triton JIT `Add` failed to upload tensor metadata.");
    if (copy_status != Runtime::kSuccess) {
      const auto release_status = Runtime::FreeAsync(data_ptr_, stream_ptr_);
      assert(release_status == Runtime::kSuccess &&
             "Triton JIT `Add` failed to release its metadata buffer.");
      data_ptr_ = nullptr;
    }
  }

  void* data_ptr() const { return data_ptr_; }

 private:
  void* data_ptr_{nullptr};

  typename Runtime::Stream stream_ptr_;
};

struct MetadataTensorViews {
  Tensor output_shape;

  Tensor input_strides;

  Tensor other_strides;

  Tensor output_strides;
};

MetadataTensorViews MakeMetadataTensorViews(void* metadata_ptr, int ndim,
                                            const Tensor& output) {
  const auto metadata_extent = static_cast<std::size_t>(std::max(ndim, 1));
  const Tensor::Shape metadata_shape{
      static_cast<Tensor::Size>(metadata_extent)};
  auto* base_ptr = static_cast<char*>(metadata_ptr);
  const auto make_tensor = [&](MetadataSection section) {
    const auto byte_offset =
        MetadataOffset(section, metadata_extent) * sizeof(int64_t);
    char* tensor_data_ptr = nullptr;
    if (base_ptr != nullptr) tensor_data_ptr = base_ptr + byte_offset;
    return Tensor{tensor_data_ptr, metadata_shape, DataType::kInt64,
                  output.device()};
  };

  return {
      make_tensor(MetadataSection::kOutputShape),
      make_tensor(MetadataSection::kInputStrides),
      make_tensor(MetadataSection::kOtherStrides),
      make_tensor(MetadataSection::kOutputStrides),
  };
}

std::vector<Tensor::Size> ResolveAutoTuningKeys(
    const std::vector<std::string>& requested_keys, Tensor::Size n_elements,
    Tensor::Size ndim) {
  std::vector<Tensor::Size> values;
  values.reserve(requested_keys.size());
  for (const auto& key : requested_keys) {
    if (key == "n_elements") {
      values.push_back(n_elements);
    } else if (key == "ndim") {
      values.push_back(ndim);
    } else {
      assert(
          false &&
          "Triton JIT `Add` auto-tuning keys must be `n_elements` or `ndim`.");
      return {};
    }
  }

  return values;
}

AutoTuningOptions NormalizeAutoTuningOptions(const AutoTuningOptions& options,
                                             const Config& default_config) {
  AutoTuningOptions normalized = options;
  if (normalized.keys.empty()) normalized.keys = {"n_elements"};
  if (normalized.candidates.empty()) {
    normalized.candidates = DefaultAutoTuningCandidates();
  }

  for (auto& candidate : normalized.candidates) {
    candidate = candidate.WithDefaultConstexprs(default_config);
  }

  return normalized;
}

std::optional<Grid> MakeGrid(const Config& config, Tensor::Size n_elements) {
  const auto block_size = config.FindConstexpr("BLOCK_SIZE");
  assert(block_size.has_value() && *block_size > 0 &&
         "Triton JIT `Add` requires a positive `BLOCK_SIZE`.");
  if (!block_size.has_value() || *block_size <= 0) return std::nullopt;

  const auto block_size_value = static_cast<Tensor::Size>(*block_size);
  const auto block_count =
      n_elements / block_size_value +
      static_cast<Tensor::Size>(n_elements % block_size_value != 0);
  assert(block_count <= std::numeric_limits<unsigned>::max() &&
         "Triton JIT `Add` grid does not fit in an unsigned integer.");
  if (block_count > std::numeric_limits<unsigned>::max()) return std::nullopt;

  return Grid{static_cast<unsigned>(block_count)};
}

}  // namespace

template <Device::Type kDev>
void Operator<Add, kDev, 10>::operator()(const Tensor input, const Tensor other,
                                         const double alpha, Tensor out) const {
  const Device device = out.device();
  const bool devices_match = input.device().type() == device.type() &&
                             input.device().index() == device.index() &&
                             other.device().type() == device.type() &&
                             other.device().index() == device.index();
  assert(devices_match &&
         "Triton JIT `Add` inputs and output must be on the same device.");
  if (!devices_match) return;

  const Tensor::Size n_elements = out.numel();
  if (n_elements == 0) return;

  auto device_guard = triton::jit::ScopedDevice<kDev>::Create(device.index());
  assert(device_guard.has_value() &&
         "Triton JIT `Add` failed to select a device.");
  if (!device_guard.has_value()) return;

  assert(this->ndim_ <=
             static_cast<Tensor::Size>(std::numeric_limits<int>::max()) &&
         "Triton JIT `Add` does not support this tensor rank.");
  if (this->ndim_ >
      static_cast<Tensor::Size>(std::numeric_limits<int>::max())) {
    return;
  }
  const int ndim = static_cast<int>(this->ndim_);

  DeviceBuffer<kDev> metadata_buffer(this->stream_);
  const bool needs_metadata = !this->is_input_contiguous_ ||
                              !this->is_other_contiguous_ ||
                              !this->is_out_contiguous_;
  if (needs_metadata) {
    const auto metadata =
        BuildMetadata(ndim, this->out_shape_, this->input_strides_,
                      this->other_strides_, this->out_strides_);
    metadata_buffer.Upload(metadata.data(), metadata.size() * sizeof(int64_t));
    if (metadata_buffer.data_ptr() == nullptr) return;
  }
  const auto metadata_views =
      MakeMetadataTensorViews(metadata_buffer.data_ptr(), ndim, out);

  const auto& default_config = DefaultConfig();
  const auto& config = this->config(default_config);
  const auto* auto_tuning_options_ptr = config.auto_tuning_options();
  if (auto_tuning_options_ptr != nullptr) {
    const auto options =
        NormalizeAutoTuningOptions(*auto_tuning_options_ptr, default_config);
    assert(options.warmup_milliseconds >= 0 &&
           options.repetition_milliseconds > 0 &&
           "Triton JIT `Add` auto-tuning durations are invalid.");
    if (options.warmup_milliseconds < 0 ||
        options.repetition_milliseconds <= 0) {
      return;
    }

    const auto key_values =
        ResolveAutoTuningKeys(options.keys, n_elements, this->ndim_);
    triton::jit::LaunchWithAutoTuning<kDev>(
        "add", this->stream_, options, key_values,
        [n_elements](const Config& candidate) {
          return MakeGrid(candidate, n_elements);
        },
        input, other, out, metadata_views.output_shape,
        metadata_views.input_strides, metadata_views.other_strides,
        metadata_views.output_strides, this->is_input_contiguous_,
        this->is_other_contiguous_, this->is_out_contiguous_, ndim, n_elements,
        alpha);
    return;
  }

  const auto effective_config = config.WithDefaultConstexprs(default_config);
  const auto grid = MakeGrid(effective_config, n_elements);
  assert(grid.has_value() && "Triton JIT `Add` grid is invalid.");
  if (!grid.has_value()) return;
  triton::jit::Launch<kDev>(
      "add", this->stream_, *grid, effective_config, input, other, out,
      metadata_views.output_shape, metadata_views.input_strides,
      metadata_views.other_strides, metadata_views.output_strides,
      this->is_input_contiguous_, this->is_other_contiguous_,
      this->is_out_contiguous_, ndim, n_elements, alpha);
}

template class Operator<Add, Device::Type::kNvidia, 10>;

}  // namespace infini::ops
