#ifndef INFINI_OPS_CAMBRICON_ADD_H_
#define INFINI_OPS_CAMBRICON_ADD_H_

#include "base/add.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"
#include "native/cambricon/op_tensor.h"

namespace infini::ops {

template <typename T>
void AddUnion(void* workspace, int core_per_cluster, int cluster_count,
              cnrtQueue_t queue, const void* input, const void* other,
              void* out, const size_t* out_shape, const ptrdiff_t* out_strides,
              const size_t* input_shape, const ptrdiff_t* input_strides,
              const size_t* other_shape, const ptrdiff_t* other_strides,
              size_t output_size, int ndim, bool fast_path, bool out_contiguous,
              double alpha);

template <>
class Operator<Add, Device::Type::kCambricon> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, const double alpha,
           Tensor out)
      : Add{input, other, alpha, out} {
    if (output_size_ == 0) return;
    if ((out_type_ == DataType::kFloat16 || out_type_ == DataType::kFloat32) &&
        input_type_ == out_type_ && other_type_ == out_type_ &&
        input.device() == out.device() && other.device() == out.device() &&
        input.IsContiguous() && other.IsContiguous() && out.IsContiguous()) {
      cnnl_ = std::make_unique<cnnl_utils::OpTensor>(CNNL_OP_TENSOR_ADD, input, other, out);
      return;
    }
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster,
                                &cluster_count);
    cnrtMalloc(&default_workspace_, workspace_size_in_bytes());
  }

  Operator(const Tensor input, const Tensor other, Tensor out)
      : Operator{input, other, 1.0, out} {}

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const override {
    if (output_size_ == 0) return;
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    if (cnnl_) {
      cnnl_->Run(input, other, out, static_cast<float>(alpha), queue,
                 workspace_, workspace_size_in_bytes_);
      return;
    }
    auto workspace{workspace_ ? workspace_ : default_workspace_};

    bool fast_path = is_input_contiguous_ && is_other_contiguous_ &&
                     is_out_contiguous_ && input_shape_ == out_shape_ &&
                     other_shape_ == out_shape_;

    DispatchFunc<Device::Type::kCambricon,
                 CambriconBangFloatTypes<DataType::kInt32, DataType::kInt64>>(
        {out_type_},
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          AddUnion<T>(workspace, core_per_cluster, cluster_count, queue,
                      input.data(), other.data(), out.data(), out_shape_.data(),
                      out_strides_.data(), input_shape_.data(),
                      input_strides_.data(), other_shape_.data(),
                      other_strides_.data(), output_size_, ndim_, fast_path,
                      is_out_contiguous_, alpha);
        },
        "CambriconAdd::operator() - output dispatch");
  }

  ~Operator() {
    if (default_workspace_) cnrtFree(default_workspace_);
  }

  std::size_t workspace_size_in_bytes() const override {
    if (cnnl_) return cnnl_->WorkspaceSize();
    if (output_size_ == 0) return 0;
    return ndim_ * (3 * sizeof(size_t) + 3 * sizeof(ptrdiff_t));
  }

  void* default_workspace_{nullptr};
  int core_per_cluster = 0;
  int cluster_count = 0;

 private:
  std::unique_ptr<cnnl_utils::OpTensor> cnnl_;
};

}  // namespace infini::ops

#endif
