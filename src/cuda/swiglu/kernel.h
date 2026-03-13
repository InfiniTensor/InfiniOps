#ifndef INFINI_OPS_CUDA_SWIGLU_KERNEL_H_
#define INFINI_OPS_CUDA_SWIGLU_KERNEL_H_

#include <cassert>
#include <cstdint>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "base/swiglu.h"
#include "common/generic_utils.h"
#include "cuda/swiglu/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaSwiglu : public Swiglu {
 public:
  CudaSwiglu(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu{input, gate, out} {
    const auto required_workspace_size{workspace_size_in_bytes()};
    if (required_workspace_size == 0) {
      return;
    }

    Backend::malloc((void**)&default_workspace_, required_workspace_size);
    InitializeDefaultWorkspace();
  }

  ~CudaSwiglu() override {
    if (default_workspace_ != nullptr) {
      Backend::free(default_workspace_);
    }
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream_ ? stream_ : 0);
    auto* workspace{workspace_ ? workspace_ : default_workspace_};
    const auto required_workspace_size{workspace_size_in_bytes()};
    const auto workspace_size{workspace_ && workspace_size_in_bytes_
                                  ? workspace_size_in_bytes_
                                  : required_workspace_size};
    auto layout{WorkspaceLayout{}};

    if (required_workspace_size != 0) {
      assert(workspace != nullptr && "`Swiglu` requires a workspace buffer.");
      assert(workspace_size >= required_workspace_size &&
             "`workspace_size_in_bytes` is insufficient for `Swiglu`.");

      layout = ResolveWorkspaceLayout(workspace);
      if (workspace_ != nullptr) {
        CopyMetadataAsync(layout, cuda_stream);
      }
    }

    DispatchFunc<AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          int block_size = GetOptimalBlockSize();
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));
          size_t step = gridDims.x * blockDims.x;

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          const T* d_gate = reinterpret_cast<const T*>(gate.data());

#define LAUNCH_SWIGLU_KERNEL(BLOCK_SIZE)                                  \
  for (size_t i = 0; i < output_size_; i += step) {                       \
    SwigluKernel<T, BLOCK_SIZE><<<gridDims, blockDims, 0, cuda_stream>>>( \
        d_out, d_input, d_gate, layout.out_shape, layout.input_shape,     \
        layout.gate_shape, layout.out_strides, layout.input_strides,      \
        layout.gate_strides, output_size_, ndim_, i, is_out_contiguous_,  \
        is_input_contiguous_, is_gate_contiguous_);                       \
  }

          if (block_size == CUDA_BLOCK_SIZE_1024) {
            LAUNCH_SWIGLU_KERNEL(CUDA_BLOCK_SIZE_1024)
          } else if (block_size == CUDA_BLOCK_SIZE_512) {
            LAUNCH_SWIGLU_KERNEL(CUDA_BLOCK_SIZE_512)
          } else if (block_size == CUDA_BLOCK_SIZE_256) {
            LAUNCH_SWIGLU_KERNEL(CUDA_BLOCK_SIZE_256)
          } else {
            LAUNCH_SWIGLU_KERNEL(CUDA_BLOCK_SIZE_128)
          }

#undef LAUNCH_SWIGLU_KERNEL
        },
        "CudaSwiglu::operator()");
  }

  std::size_t workspace_size_in_bytes() const override {
    if (output_size_ == 0 || ndim_ == 0) {
      return 0;
    }

    return MetadataWorkspaceSizeInBytes();
  }

 private:
  void InitializeDefaultWorkspace() {
    CopyMetadata(ResolveWorkspaceLayout(default_workspace_));
  }

  void CopyMetadata(const WorkspaceLayout& layout) const {
    const auto shape_size{ndim_ * sizeof(Tensor::Size)};
    const auto strides_size{ndim_ * sizeof(Tensor::Stride)};

    Backend::memcpy(layout.input_shape, input_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(layout.gate_shape, gate_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(layout.out_shape, out_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(layout.input_strides, input_strides_.data(), strides_size,
                    Backend::memcpyH2D);
    Backend::memcpy(layout.gate_strides, gate_strides_.data(), strides_size,
                    Backend::memcpyH2D);
    Backend::memcpy(layout.out_strides, out_strides_.data(), strides_size,
                    Backend::memcpyH2D);
  }

  void CopyMetadataAsync(const WorkspaceLayout& layout,
                         typename Backend::stream_t stream) const {
    const auto shape_size{ndim_ * sizeof(Tensor::Size)};
    const auto strides_size{ndim_ * sizeof(Tensor::Stride)};

    Backend::memcpyAsync(layout.input_shape, input_shape_.data(), shape_size,
                         Backend::memcpyH2D, stream);
    Backend::memcpyAsync(layout.gate_shape, gate_shape_.data(), shape_size,
                         Backend::memcpyH2D, stream);
    Backend::memcpyAsync(layout.out_shape, out_shape_.data(), shape_size,
                         Backend::memcpyH2D, stream);
    Backend::memcpyAsync(layout.input_strides, input_strides_.data(),
                         strides_size, Backend::memcpyH2D, stream);
    Backend::memcpyAsync(layout.gate_strides, gate_strides_.data(),
                         strides_size, Backend::memcpyH2D, stream);
    Backend::memcpyAsync(layout.out_strides, out_strides_.data(), strides_size,
                         Backend::memcpyH2D, stream);
  }

  void* default_workspace_{nullptr};
};

}  // namespace infini::ops

#endif
