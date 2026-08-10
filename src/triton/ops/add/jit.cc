#include "triton/ops/add/jit.h"

#include <cassert>
#include <memory>
#include <unordered_map>

#include "runtime.h"
#include "triton/jit/jit.h"

namespace infini::ops {

template <Device::Type kDev>
void Operator<Add, kDev, 10>::operator()(const Tensor input, const Tensor other,
                                         const double alpha, Tensor out) const {
  const int ndim = static_cast<int>(this->ndim_);

  std::vector<int64_t> h_meta(4 * std::max(ndim, 1), 0);
  for (int i = 0; i < ndim; ++i) {
    h_meta[0 * ndim + i] = static_cast<int64_t>(this->out_shape_[i]);
    h_meta[1 * ndim + i] = static_cast<int64_t>(this->input_strides_[i]);
    h_meta[2 * ndim + i] = static_cast<int64_t>(this->other_strides_[i]);
    h_meta[3 * ndim + i] = static_cast<int64_t>(this->out_strides_[i]);
  }
  const size_t meta_bytes = h_meta.size() * sizeof(int64_t);
  void* d_meta = nullptr;
  Runtime<kDev>::Malloc(&d_meta, meta_bytes);
  Runtime<kDev>::Memcpy(d_meta, h_meta.data(), meta_bytes,
                        Runtime<kDev>::kMemcpyHostToDevice);
  const size_t stride_bytes = ndim * sizeof(int64_t);

  std::vector<Tensor::Size> meta_shape{
      static_cast<Tensor::Size>(std::max(ndim, 1))};
  char* base = static_cast<char*>(d_meta);
  Tensor d_out_shape{base + stride_bytes * 0, meta_shape, DataType::kInt64,
                     out.device()};
  Tensor d_input_strides{base + stride_bytes * 1, meta_shape, DataType::kInt64,
                         out.device()};
  Tensor d_other_strides{base + stride_bytes * 2, meta_shape, DataType::kInt64,
                         out.device()};
  Tensor d_out_strides{base + stride_bytes * 3, meta_shape, DataType::kInt64,
                       out.device()};

  const size_t n_elements = out.numel();

  static const JitConfig defaults = DefaultConfig();
  std::shared_ptr<Config> extension = this->config_.extension();
  auto cfg = std::static_pointer_cast<JitConfig>(extension);

  const std::unordered_map<std::string, Tensor::Size> args{
      {"n_elements", n_elements},
      {"ndim", ndim},
  };

  int result;
  if (cfg && cfg->autotune) {
    auto tune = std::static_pointer_cast<AutotuneConfig>(extension);
    if (tune->candidates.empty()) tune->candidates = AutotuneConfigs();
    for (auto& c : tune->candidates) c.ApplyDefaults(defaults);

    auto key_names = tune->key.empty() ? DefaultKey() : tune->key;
    std::vector<Tensor::Size> key_vals;
    for (const auto& name : key_names) key_vals.push_back(args.at(name));

    result = LaunchJitAutotune<kDev>(
        "add", this->stream_, *tune, key_vals,
        {input.dtype(), other.dtype(), out.dtype()},
        [&](const JitConfig& c) {
          int block_size = c.At("BLOCK_SIZE");
          return Grid{static_cast<unsigned>((n_elements + block_size - 1) /
                                            block_size)};
        },
        input, other, out, d_out_shape, d_input_strides, d_other_strides,
        d_out_strides, this->is_input_contiguous_, this->is_other_contiguous_,
        this->is_out_contiguous_, ndim, n_elements, alpha);
  } else {
    JitConfig config = cfg ? *cfg : defaults;
    if (cfg) config.ApplyDefaults(defaults);
    const int block_size = config.At("BLOCK_SIZE");
    Grid grid{
        static_cast<unsigned>((n_elements + block_size - 1) / block_size)};
    result = LaunchJit<kDev>("add", this->stream_, grid, config, input, other,
                             out, d_out_shape, d_input_strides, d_other_strides,
                             d_out_strides, this->is_input_contiguous_,
                             this->is_other_contiguous_,
                             this->is_out_contiguous_, ndim, n_elements, alpha);
  }

  Runtime<kDev>::FreeAsync(
      d_meta, static_cast<typename Runtime<kDev>::Stream>(this->stream_));

  assert(result == 0 && "Triton JIT `Add` launch failed");
}

template class Operator<Add, Device::Type::kNvidia, 10>;

}  // namespace infini::ops
