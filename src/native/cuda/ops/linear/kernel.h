#ifndef INFINI_OPS_CUDA_LINEAR_KERNEL_H_
#define INFINI_OPS_CUDA_LINEAR_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <optional>
#include <utility>

#include "base/add.h"
#include "base/gemm.h"
#include "base/linear.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/caster.cuh"
#include "runtime.h"

namespace infini::ops {

template <Device::Type kDev, typename T, unsigned int block_size>
__global__ void LinearKernel(T* out, int64_t stride_out, const T* input,
                             int64_t stride_input, const T* weight,
                             int64_t stride_weight_n, int64_t stride_weight_k,
                             const T* bias, int64_t stride_bias,
                             int64_t in_features) {
  const int64_t column = blockIdx.x;
  float sum = 0;

  for (int64_t i = threadIdx.x; i < in_features; i += block_size) {
    sum += Caster<kDev>::template Cast<float>(input[i * stride_input]) *
           Caster<kDev>::template Cast<float>(
               weight[column * stride_weight_n + i * stride_weight_k]);
  }

  using BlockReduce = cub::BlockReduce<float, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    if (bias) {
      sum += Caster<kDev>::template Cast<float>(bias[column * stride_bias]);
    }
    out[column * stride_out] = Caster<kDev>::template Cast<T>(sum);
  }
}

template <Device::Type device_type>
class CudaLinear : public Linear {
 public:
  CudaLinear(const Tensor input, const Tensor weight,
             std::optional<Tensor> bias, Tensor out)
      : Linear{input, weight, bias, out},
        can_flatten_rows_{input.ndim() > 1 && input.IsContiguous() &&
                          out.IsContiguous()},
        can_use_row_gemm_{input.ndim() > 1 && input.stride(-1) == 1 &&
                          out.stride(-1) == 1},
        gemm_{can_flatten_rows_ ? FlatInputView(input, weight, out)
                                : RowGemmInputView(input, weight),
              weight,
              can_flatten_rows_ ? FlatOutputView(out) : RowGemmOutputView(out)},
        add_{} {
    assert(weight.ndim() == 2 && "operator `Linear` requires a 2D `weight`");

    if (bias) add_.emplace(BroadcastBias(*bias, out), out, out);
  }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, Tensor out) const override {
    assert(has_bias_ == bias.has_value());

    if (can_flatten_rows_) {
      auto a = FlatInputView(input, weight, out);
      auto y = FlatOutputView(out);
      ConfigureSubOperator(gemm_);
      gemm_(a, weight, std::nullopt, std::optional<float>{1.0F},
            std::optional<float>{0.0F}, std::nullopt, std::optional<int>{1}, y);
    } else {
      for (Tensor::Size row = 0; row < rows_; ++row) {
        if (can_use_row_gemm_) {
          auto a = RowGemmInputView(input, weight, row);
          auto y = RowGemmOutputView(out, row);

          ConfigureSubOperator(gemm_);
          gemm_(a, weight, std::nullopt, std::optional<float>{1.0F},
                std::optional<float>{0.0F}, std::nullopt, std::optional<int>{1},
                y);
          continue;
        }

        RunLinearKernelRow(input, weight, bias, out, row);
      }
    }

    if (bias && add_ && (can_flatten_rows_ || can_use_row_gemm_)) {
      auto bias_view = BroadcastBias(*bias, out);
      ConfigureSubOperator(*add_);
      (*add_)(bias_view, out, 1.0, out);
    }
  }

 private:
  static Tensor FlatInputView(const Tensor input, const Tensor weight,
                              const Tensor out) {
    assert(input.IsContiguous() && out.IsContiguous());

    const auto in_features = input.size(-1);
    const Tensor::Stride row_stride = input.ndim() == 1 ? 0 : input.stride(-2);

    return Tensor{const_cast<void*>(input.data()),
                  Tensor::Shape{out.numel() / weight.size(0), in_features},
                  input.dtype(), input.device(),
                  Tensor::Strides{row_stride, input.stride(-1)}};
  }

  static Tensor FlatOutputView(const Tensor out) {
    assert(out.IsContiguous());

    const auto columns = out.size(-1);
    const Tensor::Stride row_stride = out.ndim() == 1 ? 0 : out.stride(-2);

    return Tensor{const_cast<void*>(out.data()),
                  Tensor::Shape{out.numel() / columns, columns}, out.dtype(),
                  out.device(), Tensor::Strides{row_stride, out.stride(-1)}};
  }

  static Tensor RowGemmInputView(const Tensor input, const Tensor weight,
                                 Tensor::Size row = 0) {
    auto* data = static_cast<std::byte*>(const_cast<void*>(input.data())) +
                 static_cast<std::size_t>(LeadingOffset(input, row)) *
                     input.element_size();

    return Tensor{
        data, Tensor::Shape{1, weight.size(1)}, input.dtype(), input.device(),
        Tensor::Strides{static_cast<Tensor::Stride>(weight.size(1)), 1}};
  }

  static Tensor RowGemmOutputView(const Tensor out, Tensor::Size row = 0) {
    auto* data =
        static_cast<std::byte*>(const_cast<void*>(out.data())) +
        static_cast<std::size_t>(LeadingOffset(out, row)) * out.element_size();

    return Tensor{
        data, Tensor::Shape{1, out.size(-1)}, out.dtype(), out.device(),
        Tensor::Strides{static_cast<Tensor::Stride>(out.size(-1)), 1}};
  }

  static Tensor::Stride LeadingOffset(const Tensor tensor, Tensor::Size row) {
    Tensor::Stride offset = 0;
    auto remaining = row;

    for (std::size_t reverse = 1; reverse < tensor.ndim(); ++reverse) {
      const auto axis = tensor.ndim() - reverse - 1;
      const auto coordinate = remaining % tensor.size(axis);
      remaining /= tensor.size(axis);
      offset += static_cast<Tensor::Stride>(coordinate) * tensor.stride(axis);
    }

    return offset;
  }

  static Tensor BroadcastBias(const Tensor bias, const Tensor out) {
    Tensor::Strides strides(out.ndim(), 0);
    strides.back() = bias.stride(0);

    return Tensor{const_cast<void*>(bias.data()), out.shape(), bias.dtype(),
                  bias.device(), strides};
  }

  template <typename Op>
  void ConfigureSubOperator(Op& op) const {
    if (handle_ptr_) op.set_handle(*handle_ptr_);

    op.set_stream(stream_);
    op.set_workspace(workspace_);
    op.set_workspace_size_in_bytes(workspace_size_in_bytes_);
  }

  void RunLinearKernelRow(const Tensor input, const Tensor weight,
                          const std::optional<Tensor>& bias, const Tensor out,
                          Tensor::Size row) const {
    auto cuda_stream = static_cast<typename Runtime<device_type>::Stream>(
        stream_ ? stream_ : 0);

    DispatchFunc<device_type,
                 ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          LinearKernelRow<T>(input, weight, bias, out, row, cuda_stream);
        },
        "CudaLinear::RunLinearKernelRow()");
  }

  template <typename T, typename Stream>
  void LinearKernelRow(const Tensor input, const Tensor weight,
                       const std::optional<Tensor>& bias, const Tensor out,
                       Tensor::Size row, Stream stream) const {
    constexpr unsigned int kBlockSize = 256;
    auto* input_data = AdvanceData<T>(input, LeadingOffset(input, row));
    auto* weight_data = AdvanceData<T>(weight, 0);
    auto* bias_data = bias ? AdvanceData<T>(*bias, 0) : nullptr;
    auto* out_data = AdvanceData<T>(out, LeadingOffset(out, row));

    LinearKernel<device_type, T, kBlockSize>
        <<<static_cast<unsigned int>(out.size(-1)), kBlockSize, 0, stream>>>(
            out_data, out.stride(-1), input_data, input.stride(-1), weight_data,
            weight.stride(0), weight.stride(1), bias_data,
            bias ? bias->stride(0) : 0, static_cast<int64_t>(input.size(-1)));
  }

  template <typename T>
  static T* AdvanceData(const Tensor tensor, Tensor::Stride offset) {
    auto* data = static_cast<std::byte*>(const_cast<void*>(tensor.data())) +
                 static_cast<std::size_t>(offset) * tensor.element_size();

    return reinterpret_cast<T*>(data);
  }

  bool can_flatten_rows_;

  bool can_use_row_gemm_;

  mutable Operator<Gemm, device_type> gemm_;

  mutable std::optional<Operator<Add, device_type>> add_;
};

}  // namespace infini::ops

#endif
