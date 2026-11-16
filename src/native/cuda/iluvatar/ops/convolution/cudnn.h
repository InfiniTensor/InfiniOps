#ifndef INFINI_OPS_ILUVATAR_CONVOLUTION_CUDNN_H_
#define INFINI_OPS_ILUVATAR_CONVOLUTION_CUDNN_H_

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cassert>
#include <cstdlib>
#include <limits>
#include <vector>

#include "base/convolution.h"
#include "native/cuda/iluvatar/ops/convolution/workspace.h"

namespace infini::ops::iluvatar_conv_detail {

class CudnnState {
 public:
  CudnnState() = default;

  CudnnState(const CudnnState&) = delete;

  CudnnState& operator=(const CudnnState&) = delete;

  ~CudnnState() {
    int previous = -1;
    cudaGetDevice(&previous);
    if (device_ >= 0 && previous != device_) cudaSetDevice(device_);
    if (workspace_) cudaFree(workspace_);
    if (convolution_) cudnnDestroyConvolutionDescriptor(convolution_);
    if (weight_) cudnnDestroyFilterDescriptor(weight_);
    if (output_) cudnnDestroyTensorDescriptor(output_);
    if (input_) cudnnDestroyTensorDescriptor(input_);
    if (handle_) cudnnDestroy(handle_);
    if (previous >= 0 && previous != device_) cudaSetDevice(previous);
  }

  static bool Eligible(const Tensor input, const Tensor weight,
                       const std::optional<Tensor>& bias,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool transposed,
                       const std::vector<int64_t>& output_padding,
                       int64_t groups, const Tensor out) {
    if (bias || transposed || groups != 1 ||
        input.device() != weight.device() || input.device() != out.device() ||
        stride.size() != 2 || padding.size() != 2 ||
        dilation != std::vector<int64_t>({1, 1}) ||
        output_padding != std::vector<int64_t>({0, 0}))
      return false;
    for (const auto tensor : {input, weight, out}) {
      if (tensor.dtype() != DataType::kFloat32 || tensor.ndim() != 4 ||
          !tensor.IsContiguous())
        return false;
      size_t size = 1;
      for (const auto extent : tensor.shape()) {
        if (extent == 0 ||
            extent >
                static_cast<size_t>(std::numeric_limits<int>::max()) / size)
          return false;
        size *= extent;
      }
    }
    for (size_t axis = 0; axis < 2; ++axis) {
      if (stride[axis] <= 0 || stride[axis] > std::numeric_limits<int>::max() ||
          padding[axis] < 0 || padding[axis] > std::numeric_limits<int>::max())
        return false;
    }
    return true;
  }

  bool Initialize(const Tensor input, const Tensor weight,
                  const std::vector<int64_t>& stride,
                  const std::vector<int64_t>& padding, const Tensor out) {
    const char* setting = std::getenv("INFINIOPS_ILUVATAR_CONV_WORKSPACE_MB");
    const auto limit = WorkspaceLimit(setting ? setting : "");
    if (!limit || cudaGetDevice(&device_) != cudaSuccess ||
        device_ != input.device().index())
      return false;
    if (cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&input_) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&output_) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&weight_) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&convolution_) != CUDNN_STATUS_SUCCESS)
      return false;
    const auto x = input.shape(), w = weight.shape(), y = out.shape();
    if (cudnnSetTensor4dDescriptor(input_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   x[0], x[1], x[2],
                                   x[3]) != CUDNN_STATUS_SUCCESS ||
        cudnnSetTensor4dDescriptor(output_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   y[0], y[1], y[2],
                                   y[3]) != CUDNN_STATUS_SUCCESS ||
        cudnnSetFilter4dDescriptor(weight_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                   w[0], w[1], w[2],
                                   w[3]) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolution2dDescriptor(
            convolution_, padding[0], padding[1], stride[0], stride[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionMathType(convolution_, CUDNN_DEFAULT_MATH) !=
            CUDNN_STATUS_SUCCESS)
      return false;
    cudnnConvolutionFwdAlgoPerf_t candidates[8]{};
    int count = 0;
    if (cudnnGetConvolutionForwardAlgorithm_v7(
            handle_, input_, weight_, convolution_, output_, 8, &count,
            candidates) != CUDNN_STATUS_SUCCESS ||
        count < 1 || count > 8)
      return false;
    for (int i = 0; i < count; ++i) {
      const auto& candidate = candidates[i];
      if (candidate.status != CUDNN_STATUS_SUCCESS ||
          candidate.mathType != CUDNN_DEFAULT_MATH ||
          candidate.determinism != CUDNN_DETERMINISTIC ||
          candidate.memory > *limit)
        continue;
      size_t bytes = 0;
      if (cudnnGetConvolutionForwardWorkspaceSize(
              handle_, input_, weight_, convolution_, output_, candidate.algo,
              &bytes) != CUDNN_STATUS_SUCCESS ||
          bytes > *limit)
        continue;
      if (bytes && cudaMalloc(&workspace_, bytes) != cudaSuccess) {
        workspace_ = nullptr;
        cudaGetLastError();
        continue;
      }
      bytes_ = bytes;
      algorithm_ = candidate.algo;
      return true;
    }
    return false;
  }

  bool Run(const Tensor input, const Tensor weight, void* stream,
           Tensor out) const {
    if (cudnnSetStream(handle_, static_cast<cudaStream_t>(stream)) !=
        CUDNN_STATUS_SUCCESS)
      return false;
    const float alpha = 1.0f, beta = 0.0f;
    return cudnnConvolutionForward(handle_, &alpha, input_, input.data(),
                                   weight_, weight.data(), convolution_,
                                   algorithm_, workspace_, bytes_, &beta,
                                   output_, out.data()) == CUDNN_STATUS_SUCCESS;
  }

 private:
  int device_ = -1;

  cudnnHandle_t handle_ = nullptr;

  cudnnTensorDescriptor_t input_ = nullptr;

  cudnnTensorDescriptor_t output_ = nullptr;

  cudnnFilterDescriptor_t weight_ = nullptr;

  cudnnConvolutionDescriptor_t convolution_ = nullptr;

  void* workspace_ = nullptr;

  size_t bytes_ = 0;

  cudnnConvolutionFwdAlgo_t algorithm_{};
};

}  // namespace infini::ops::iluvatar_conv_detail

#endif
