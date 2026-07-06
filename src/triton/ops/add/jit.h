#ifndef INFINI_OPS_TRITON_JIT_ADD_H_
#define INFINI_OPS_TRITON_JIT_ADD_H_

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <vector>

#include "base/add.h"
#include "data_type.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia, 7> : public Add {
 public:
  using Add::Add;
  using Add::operator();

  static config_t default_config() { return {4u, 3u, {{"BLOCK_SIZE", 1024}}}; }

  static std::vector<config_t> autotune_configs() {
    return {
        {4u, 3u, {{"BLOCK_SIZE", 256}}},
        {4u, 3u, {{"BLOCK_SIZE", 512}}},
        {8u, 4u, {{"BLOCK_SIZE", 1024}}},
        {8u, 4u, {{"BLOCK_SIZE", 2048}}},
    };
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    const int ndim = static_cast<int>(ndim_);

    std::vector<int64_t> h_meta(4 * std::max(ndim, 1), 0);
    for (int i = 0; i < ndim; ++i) {
      h_meta[0 * ndim + i] = static_cast<int64_t>(out_shape_[i]);
      h_meta[1 * ndim + i] = static_cast<int64_t>(input_strides_[i]);
      h_meta[2 * ndim + i] = static_cast<int64_t>(other_strides_[i]);
      h_meta[3 * ndim + i] = static_cast<int64_t>(out_strides_[i]);
    }
    const size_t meta_bytes = h_meta.size() * sizeof(int64_t);
    CUdeviceptr d_meta;
    cuMemAlloc(&d_meta, meta_bytes);
    cuMemcpyHtoD(d_meta, h_meta.data(), meta_bytes);
    const size_t stride_bytes = ndim * sizeof(int64_t);

    auto meta_shape =
        std::vector<Tensor::Size>{static_cast<Tensor::Size>(std::max(ndim, 1))};
    Tensor d_out_shape{reinterpret_cast<void*>(d_meta + 0 * stride_bytes),
                       meta_shape, DataType::kInt64, out.device()};
    Tensor d_input_strides{reinterpret_cast<void*>(d_meta + 1 * stride_bytes),
                           meta_shape, DataType::kInt64, out.device()};
    Tensor d_other_strides{reinterpret_cast<void*>(d_meta + 2 * stride_bytes),
                           meta_shape, DataType::kInt64, out.device()};
    Tensor d_out_strides{reinterpret_cast<void*>(d_meta + 3 * stride_bytes),
                         meta_shape, DataType::kInt64, out.device()};

    const size_t n_elements = out.numel();

    auto extension = config_.extension();
    static const config_t defaults = default_config();
    const auto* config_ptr = static_cast<const config_t*>(extension.get());
    config_t config = config_ptr ? *config_ptr : defaults;
    if (extension) config.apply_defaults(defaults);

    int result;
    if (config.is_autotune()) {
      if (config.configs.empty()) config.configs = autotune_configs();
      for (auto& c : config.configs) c.apply_defaults(defaults);
      result = launch_jit_autotune(
          "add", stream_, config, {n_elements}, out.dtype(),
          [&](const config_t& c) {
            int block_size = c.at("BLOCK_SIZE");
            return grid_t{static_cast<unsigned>((n_elements + block_size - 1) /
                                                block_size)};
          },
          input, other, out, d_out_shape, d_input_strides, d_other_strides,
          d_out_strides, is_input_contiguous_, is_other_contiguous_,
          is_out_contiguous_, ndim, n_elements);
    } else {
      const int block_size = config.at("BLOCK_SIZE");
      grid_t grid{
          static_cast<unsigned>((n_elements + block_size - 1) / block_size)};
      result = launch_jit(
          "add", stream_, grid, config, input, other, out, d_out_shape,
          d_input_strides, d_other_strides, d_out_strides, is_input_contiguous_,
          is_other_contiguous_, is_out_contiguous_, ndim, n_elements);
    }

    cuMemFreeAsync(d_meta, static_cast<CUstream>(stream_));

    assert(result == 0 && "Triton JIT `Add` launch failed");
  }
};

}  // namespace infini::ops

#endif
