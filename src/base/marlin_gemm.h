#ifndef INFINI_OPS_BASE_MARLIN_GEMM_H_
#define INFINI_OPS_BASE_MARLIN_GEMM_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM's low-level `marlin_gemm` operator.
// Packed weights, scales, bias, and zero points require vLLM preprocessing.
class MarlinGemm : public Operator<MarlinGemm> {
 public:
  MarlinGemm(const Tensor a, const Tensor b_q_weight,
             const std::optional<Tensor> b_bias, const Tensor b_scales,
             const std::optional<Tensor> a_scales,
             const std::optional<Tensor> global_scale,
             const std::optional<Tensor> b_zeros,
             const std::optional<Tensor> g_idx,
             const std::optional<Tensor> perm, const Tensor workspace,
             const int64_t b_type_id, const int64_t size_m,
             const int64_t size_n, const int64_t size_k, const bool is_k_full,
             const bool use_atomic_add, const bool use_fp32_reduce,
             const bool is_zp_float, Tensor out)
      : size_m_{size_m},
        size_n_{size_n},
        size_k_{size_k},
        num_groups_{b_scales.ndim() == 2 ? b_scales.size(0) : 0},
        b_type_id_{b_type_id},
        is_k_full_{is_k_full},
        use_atomic_add_{use_atomic_add},
        use_fp32_reduce_{use_fp32_reduce},
        is_zp_float_{is_zp_float},
        device_type_{a.device().type()},
        device_index_{a.device().index()},
        a_metadata_{a},
        b_q_weight_metadata_{b_q_weight},
        b_bias_metadata_{b_bias},
        b_scales_metadata_{b_scales},
        a_scales_metadata_{a_scales},
        global_scale_metadata_{global_scale},
        b_zeros_metadata_{b_zeros},
        g_idx_metadata_{g_idx},
        perm_metadata_{perm},
        workspace_metadata_{workspace},
        out_metadata_{out} {
    Validate(a, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros,
             g_idx, perm, workspace, out);
  }

  virtual void operator()(
      const Tensor a, const Tensor b_q_weight,
      const std::optional<Tensor> b_bias, const Tensor b_scales,
      const std::optional<Tensor> a_scales,
      const std::optional<Tensor> global_scale,
      const std::optional<Tensor> b_zeros, const std::optional<Tensor> g_idx,
      const std::optional<Tensor> perm, const Tensor workspace,
      const int64_t b_type_id, const int64_t size_m, const int64_t size_n,
      const int64_t size_k, const bool is_k_full, const bool use_atomic_add,
      const bool use_fp32_reduce, const bool is_zp_float, Tensor out) const = 0;

 protected:
  void ValidateCallMetadata(
      const Tensor a, const Tensor b_q_weight,
      const std::optional<Tensor> b_bias, const Tensor b_scales,
      const std::optional<Tensor> a_scales,
      const std::optional<Tensor> global_scale,
      const std::optional<Tensor> b_zeros, const std::optional<Tensor> g_idx,
      const std::optional<Tensor> perm, const Tensor workspace,
      const int64_t b_type_id, const int64_t size_m, const int64_t size_n,
      const int64_t size_k, const bool is_k_full, const bool use_atomic_add,
      const bool use_fp32_reduce, const bool is_zp_float,
      const Tensor out) const {
    assert(b_type_id == b_type_id_ && size_m == size_m_ && size_n == size_n_ &&
           size_k == size_k_ && is_k_full == is_k_full_ &&
           use_atomic_add == use_atomic_add_ &&
           use_fp32_reduce == use_fp32_reduce_ && is_zp_float == is_zp_float_ &&
           "`MarlinGemm` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    assert(same_metadata(a_metadata_, a) &&
           same_metadata(b_q_weight_metadata_, b_q_weight) &&
           SameOptionalMetadata(b_bias_metadata_, b_bias) &&
           same_metadata(b_scales_metadata_, b_scales) &&
           SameOptionalMetadata(a_scales_metadata_, a_scales) &&
           SameOptionalMetadata(global_scale_metadata_, global_scale) &&
           SameOptionalMetadata(b_zeros_metadata_, b_zeros) &&
           SameOptionalMetadata(g_idx_metadata_, g_idx) &&
           SameOptionalMetadata(perm_metadata_, perm) &&
           same_metadata(workspace_metadata_, workspace) &&
           same_metadata(out_metadata_, out) &&
           "`MarlinGemm` tensor metadata differs from its descriptor");
  }

  int64_t size_m_{0};

  int64_t size_n_{0};

  int64_t size_k_{0};

  Tensor::Size num_groups_{0};

  int64_t b_type_id_{0};

  bool is_k_full_{false};

  bool use_atomic_add_{false};

  bool use_fp32_reduce_{false};

  bool is_zp_float_{false};

  Device::Type device_type_;

  int device_index_{0};

 private:
  static bool SameOptionalMetadata(const std::optional<Tensor>& expected,
                                   const std::optional<Tensor>& actual) {
    if (expected.has_value() != actual.has_value()) {
      return false;
    }
    if (!expected) {
      return true;
    }

    return std::equal_to<Tensor>{}(*expected, *actual);
  }

  void Validate(const Tensor a, const Tensor b_q_weight,
                const std::optional<Tensor> b_bias, const Tensor b_scales,
                const std::optional<Tensor> a_scales,
                const std::optional<Tensor> global_scale,
                const std::optional<Tensor> b_zeros,
                const std::optional<Tensor> g_idx,
                const std::optional<Tensor> perm, const Tensor workspace,
                const Tensor out) const {
    assert(size_m_ >= 0 && size_n_ > 0 && size_k_ > 0 && size_k_ % 16 == 0 &&
           size_n_ % 64 == 0 &&
           "`MarlinGemm` requires non-negative `size_m`, positive `size_n` "
           "and `size_k`, `size_k` divisible by 16, and `size_n` divisible "
           "by 64");
    assert(size_m_ <= std::numeric_limits<int>::max() &&
           size_n_ <= std::numeric_limits<int>::max() &&
           size_k_ <= std::numeric_limits<int>::max() &&
           "`MarlinGemm` dimensions exceed CUDA kernel limits");
    assert(b_type_id_ > 0 && "`MarlinGemm` requires a valid `b_type_id`");

    assert(a.ndim() == 2 && a.size(0) == static_cast<Tensor::Size>(size_m_) &&
           a.size(1) == static_cast<Tensor::Size>(size_k_) &&
           "`MarlinGemm` requires `a` shape [`size_m`, `size_k`]");
    assert((a.dtype() == DataType::kFloat16 ||
            a.dtype() == DataType::kBFloat16 || a.dtype() == DataType::kInt8) &&
           "`MarlinGemm` requires float16, bfloat16, or int8 activations");
    assert(a.stride(1) == 1 && a.stride(0) % 8 == 0 &&
           "`MarlinGemm` requires a unit inner stride and an outer stride "
           "divisible by 8 for `a`");

    assert(b_q_weight.ndim() == 2 &&
           b_q_weight.size(0) == static_cast<Tensor::Size>(size_k_ / 16) &&
           b_q_weight.dtype() == DataType::kInt32 &&
           b_q_weight.IsContiguous() &&
           "`MarlinGemm` requires contiguous int32 Marlin weights with "
           "shape [`size_k / 16`, packed_n]");
    assert(b_scales.ndim() == 2 && b_scales.size(0) > 0 &&
           b_scales.size(1) == static_cast<Tensor::Size>(size_n_) &&
           (b_scales.dtype() == DataType::kFloat16 ||
            b_scales.dtype() == DataType::kBFloat16) &&
           b_scales.IsContiguous() &&
           "`MarlinGemm` requires contiguous scales with shape "
           "[`num_groups`, `size_n`] and a 16-bit floating-point dtype");

    assert(out.ndim() == 2 &&
           out.size(0) == static_cast<Tensor::Size>(size_m_) &&
           out.size(1) == static_cast<Tensor::Size>(size_n_) &&
           out.IsContiguous() &&
           (out.dtype() == DataType::kFloat16 ||
            out.dtype() == DataType::kBFloat16) &&
           "`MarlinGemm` requires contiguous float16 or bfloat16 output with "
           "shape [`size_m`, `size_n`]");
    assert(b_scales.dtype() == out.dtype() &&
           "`MarlinGemm` requires scales to match the output dtype");
    if (a.dtype() == DataType::kFloat16 || a.dtype() == DataType::kBFloat16) {
      assert(a.dtype() == out.dtype() && !a_scales &&
             "`MarlinGemm` requires matching activation/output dtypes and no "
             "`a_scales` for 16-bit activations");
    } else {
      assert(a_scales &&
             "`MarlinGemm` requires `a_scales` for 8-bit activations");
    }

    assert(workspace.ndim() == 1 && workspace.numel() > 0 &&
           workspace.dtype() == DataType::kInt32 && workspace.IsContiguous() &&
           "`MarlinGemm` requires a non-empty contiguous int32 workspace");

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == device_type_ &&
             tensor.device().index() == device_index_;
    };
    assert(same_device(b_q_weight) && same_device(b_scales) &&
           same_device(workspace) && same_device(out) &&
           "`MarlinGemm` requires all tensors on the same device");

    const auto validate_optional = [&](const std::optional<Tensor>& tensor) {
      return !tensor || same_device(*tensor);
    };
    assert(validate_optional(b_bias) && validate_optional(a_scales) &&
           validate_optional(global_scale) && validate_optional(b_zeros) &&
           validate_optional(g_idx) && validate_optional(perm) &&
           "`MarlinGemm` requires all optional tensors on the same device");

    if (b_bias) {
      assert(b_bias->ndim() == 1 &&
             b_bias->size(0) == static_cast<Tensor::Size>(size_n_) &&
             b_bias->dtype() == out.dtype() && b_bias->IsContiguous() &&
             "`MarlinGemm` requires contiguous `b_bias` with shape "
             "[`size_n`] and the output dtype");
    }
    if (a_scales) {
      assert(a_scales->dtype() == DataType::kFloat32 &&
             a_scales->IsContiguous() &&
             "`MarlinGemm` requires contiguous float32 `a_scales`");
    }
    if (global_scale) {
      assert(global_scale->dtype() == DataType::kFloat32 &&
             global_scale->IsContiguous() &&
             "`MarlinGemm` requires contiguous float32 `global_scale`");
    }
    if (b_zeros && b_zeros->numel() > 0) {
      assert(b_zeros->ndim() == 2 && b_zeros->size(0) == num_groups_ &&
             b_zeros->IsContiguous() &&
             "`MarlinGemm` requires contiguous `b_zeros` with one row per "
             "scale group");
      assert(((is_zp_float_ && b_zeros->dtype() == a.dtype()) ||
              (!is_zp_float_ && b_zeros->dtype() == DataType::kInt32) ||
              b_zeros->numel() == 0) &&
             "`MarlinGemm` zero-point dtype is incompatible with "
             "`is_zp_float`");
    }

    assert(g_idx.has_value() == perm.has_value() &&
           "`MarlinGemm` requires `g_idx` and `perm` together");
    if (g_idx) {
      assert(g_idx->ndim() == 1 && perm->ndim() == 1 &&
             g_idx->dtype() == DataType::kInt32 &&
             perm->dtype() == DataType::kInt32 && g_idx->IsContiguous() &&
             perm->IsContiguous() && g_idx->numel() == perm->numel() &&
             (g_idx->numel() == 0 ||
              g_idx->numel() == static_cast<Tensor::Size>(size_k_)) &&
             "`MarlinGemm` requires empty `g_idx`/`perm` or contiguous int32 "
             "vectors of length `size_k`");
    }
    assert((!is_zp_float_ || (b_zeros && b_zeros->numel() > 0)) &&
           "`MarlinGemm` requires non-empty `b_zeros` when `is_zp_float` is "
           "true");

    ValidateCallMetadata(a, b_q_weight, b_bias, b_scales, a_scales,
                         global_scale, b_zeros, g_idx, perm, workspace,
                         b_type_id_, size_m_, size_n_, size_k_, is_k_full_,
                         use_atomic_add_, use_fp32_reduce_, is_zp_float_, out);
  }

  Tensor a_metadata_;

  Tensor b_q_weight_metadata_;

  std::optional<Tensor> b_bias_metadata_;

  Tensor b_scales_metadata_;

  std::optional<Tensor> a_scales_metadata_;

  std::optional<Tensor> global_scale_metadata_;

  std::optional<Tensor> b_zeros_metadata_;

  std::optional<Tensor> g_idx_metadata_;

  std::optional<Tensor> perm_metadata_;

  Tensor workspace_metadata_;

  Tensor out_metadata_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MARLIN_GEMM_H_
