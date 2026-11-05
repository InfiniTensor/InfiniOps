#ifndef INFINI_OPS_CAMBRICON_MATMUL_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MATMUL_KERNEL_H_

#include <cstdint>
#include <stdexcept>
#include <string>

#include "base/matmul.h"
#include "native/cambricon/cnnl_utils.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kCambricon> : public Matmul {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Matmul{input, other, out} {
    if (input.device() != other.device() || input.device() != out.device() ||
        input_type_ != other_type_ || input_type_ != out_type_ ||
        (out_type_ != DataType::kFloat16 && out_type_ != DataType::kFloat32) ||
        input.ndim() < 2 || other.ndim() < 2 || out.ndim() < 2 ||
        !out.IsContiguous()) {
      throw std::invalid_argument(
          "Cambricon Matmul requires matching FP16 or FP32 matrix tensors");
    }
    if (out.numel() == 0) return;
    if (input.size(-1) == 0 || input.size(-1) != other.size(-2) ||
        input.size(-2) != out.size(-2) || other.size(-1) != out.size(-1)) {
      throw std::invalid_argument("Unsupported Cambricon Matmul dimensions");
    }
    CheckBatch(input, out);
    CheckBatch(other, out);

    handle_ = Create<cnnlHandle_t, cnnlDestroy>(cnnlCreate, "create handle");
    descriptor_ = Create<cnnlMatMulDescriptor_t, cnnlMatMulDescDestroy>(
        cnnlMatMulDescCreate, "create matmul descriptor");
    algo_ = Create<cnnlMatMulAlgo_t, cnnlMatMulAlgoDestroy>(
        cnnlMatMulAlgoCreate, "create algorithm");
    input_ = MatrixDescriptor(input, out.ndim(), CNNL_MATMUL_DESC_TRANSA);
    other_ = MatrixDescriptor(other, out.ndim(), CNNL_MATMUL_DESC_TRANSB);
    output_ = TensorDescriptor(Tensor::Shape(out.shape()), out_type_, out.ndim());
    const auto compute_type = CNNL_DTYPE_FLOAT;
    const int32_t allow_tf32 = 0;
    Check(cnnlSetMatMulDescAttr(descriptor_.get(), CNNL_MATMUL_DESC_COMPUTE_TYPE,
                              &compute_type, sizeof(compute_type)), "set accumulation type");
    Check(cnnlSetMatMulDescAttr(descriptor_.get(), CNNL_MATMUL_ALLOW_TF32,
                              &allow_tf32, sizeof(allow_tf32)), "disable TF32");

    auto heuristic = Create<cnnlMatMulHeuristicResult_t, cnnlDestroyMatMulHeuristicResult>(
        cnnlCreateMatMulHeuristicResult, "create heuristic");
    auto result = heuristic.get();
    // Some CNNL releases fill the result without writing return_algo_count.
    int count = -1;
    Check(cnnlGetBatchMatMulExAlgoHeuristic(handle_.get(), descriptor_.get(),
                                        input_.get(), other_.get(), output_.get(),
                                        nullptr, 1, &result, &count), "select algorithm");
    if (count != -1 && count != 1)
      throw std::runtime_error("Cambricon Matmul: no CNNL algorithm");
    Check(cnnlGetBatchMatMulExHeuristicResult(heuristic.get(), algo_.get(), &workspace_size_),
          "get workspace size");
    if (workspace_size_ != 0) {
      void* ptr = nullptr;
      if (cnrtMalloc(&ptr, workspace_size_) != cnrtSuccess)
        throw std::runtime_error("Cambricon Matmul: workspace allocation failed");
      scratch_.reset(ptr);
    }
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor other, Tensor out) const override {
    if (!output_) return;
    Check(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)), "set queue");
    const float alpha = 1.0f, beta = 0.0f;
    auto* workspace = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                          ? workspace_ : scratch_.get();
    Check(cnnlBatchMatMulEx(handle_.get(), descriptor_.get(), algo_.get(),
                                 &alpha, input_.get(), input.data(), other_.get(),
                                 other.data(), &beta, output_.get(), out.data(),
                                 workspace, workspace_size_), "execute");
  }

 private:
  template <typename T, auto Destroy>
  using Resource = std::unique_ptr<std::remove_pointer_t<T>, decltype(Destroy)>;

  template <typename T, auto Destroy, typename Creator>
  static Resource<T, Destroy> Create(Creator creator, const char* operation) {
    T value = nullptr;
    Check(creator(&value), operation);
    return Resource<T, Destroy>{value, Destroy};
  }

  static void Check(cnnlStatus_t status, const char* operation) {
    if (status != CNNL_STATUS_SUCCESS)
      throw std::runtime_error(std::string("Cambricon Matmul: ") + operation +
                               ": " + cnnlGetErrorString(status));
  }

  static void CheckBatch(const Tensor& input, const Tensor& out) {
    if (input.ndim() > out.ndim())
      throw std::invalid_argument("Cambricon Matmul: invalid broadcast rank");
    const auto offset = out.ndim() - input.ndim();
    for (size_t i = 0; i + 2 < input.ndim(); ++i) {
      if (input.shape()[i] != 1 && input.shape()[i] != out.shape()[offset + i])
        throw std::invalid_argument("Cambricon Matmul: incompatible batch dimensions");
    }
  }

  static Resource<cnnlTensorDescriptor_t, cnnlDestroyTensorDescriptor>
  TensorDescriptor(const Tensor::Shape& shape, DataType dtype, size_t rank) {
    auto result = Create<cnnlTensorDescriptor_t, cnnlDestroyTensorDescriptor>(
        cnnlCreateTensorDescriptor, "create tensor descriptor");
    std::vector<int64_t> dims(rank - shape.size(), 1);
    dims.insert(dims.end(), shape.begin(), shape.end());
    Check(cnnlSetTensorDescriptor_v2(result.get(), CNNL_LAYOUT_ARRAY,
                                    cnnl_utils::GetDataType(dtype), dims.size(), dims.data()),
          "set tensor descriptor");
    return result;
  }

  Resource<cnnlTensorDescriptor_t, cnnlDestroyTensorDescriptor>
  MatrixDescriptor(const Tensor& tensor, size_t rank,
                   cnnlMatMulDescAttribute_t attribute) {
    Tensor::Shape shape(tensor.shape());
    const auto strides = tensor.strides();
    const size_t last = shape.size() - 1;
    const auto rows = static_cast<Tensor::Stride>(shape[last - 1]);
    const auto cols = static_cast<Tensor::Stride>(shape[last]);
    const bool normal = (cols == 1 || strides[last] == 1) &&
                        (rows == 1 || strides[last - 1] == cols);
    const bool transposed = (rows == 1 || strides[last - 1] == 1) &&
                            (cols == 1 || strides[last] == rows);
    if (!normal && !transposed)
      throw std::invalid_argument("Cambricon Matmul: unsupported matrix strides");
    auto stride = rows * cols;
    for (size_t i = last - 1; i-- > 0;) {
      if (shape[i] > 1 && strides[i] != stride)
        throw std::invalid_argument("Cambricon Matmul: unsupported batch strides");
      stride *= shape[i];
    }
    const int32_t transpose = normal ? 0 : 1;
    if (transpose) std::swap(shape[last - 1], shape[last]);
    Check(cnnlSetMatMulDescAttr(descriptor_.get(), attribute, &transpose, sizeof(transpose)),
          "set transpose");
    // Preserve leading singleton dimensions even when the batch is one.
    return TensorDescriptor(shape, tensor.dtype(), rank);
  }

  Resource<cnnlHandle_t, cnnlDestroy> handle_{nullptr, cnnlDestroy};
  Resource<cnnlMatMulDescriptor_t, cnnlMatMulDescDestroy> descriptor_{nullptr, cnnlMatMulDescDestroy};
  Resource<cnnlMatMulAlgo_t, cnnlMatMulAlgoDestroy> algo_{nullptr, cnnlMatMulAlgoDestroy};
  Resource<cnnlTensorDescriptor_t, cnnlDestroyTensorDescriptor>
      input_{nullptr, cnnlDestroyTensorDescriptor},
      other_{nullptr, cnnlDestroyTensorDescriptor},
      output_{nullptr, cnnlDestroyTensorDescriptor};
  std::size_t workspace_size_ = 0;
  std::unique_ptr<void, decltype(&cnrtFree)> scratch_{nullptr, cnrtFree};
};

}  // namespace infini::ops

#endif
