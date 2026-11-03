#ifndef INFINI_OPS_ASCEND_FIA_COMMON__H_
#define INFINI_OPS_ASCEND_FIA_COMMON__H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "tensor.h"

namespace infini::ops::ascend::fia {

inline void AssertSupportedDtype(const DataType dtype, const char* op_name) {
  (void)op_name;
  assert((dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) &&
         "`FIA`: only `float16` and `bfloat16` are supported");
}

inline std::vector<int64_t> ReadIntTensor(const Tensor& tensor,
                                          aclrtStream stream) {
  assert(tensor.ndim() == 1 && "`FIA`: sequence tensor must be 1D");

  const auto n = tensor.numel();
  std::vector<int64_t> result(n);

  if (tensor.dtype() == DataType::kInt32) {
    std::vector<int32_t> tmp(n);
    const int32_t* src = nullptr;

    if (tensor.device().type() == Device::Type::kCpu) {
      src = static_cast<const int32_t*>(tensor.data());
    } else {
      auto ret = aclrtMemcpyAsync(tmp.data(), n * sizeof(int32_t),
                                  tensor.data(), n * sizeof(int32_t),
                                  ACL_MEMCPY_DEVICE_TO_HOST, stream);
      assert(ret == ACL_SUCCESS && "`FIA`: D2H copy failed");
      ret = aclrtSynchronizeStream(stream);
      assert(ret == ACL_SUCCESS && "`FIA`: stream synchronize failed");
      src = tmp.data();
    }

    for (std::size_t i = 0; i < n; ++i) {
      result[i] = static_cast<int64_t>(src[i]);
    }

    return result;
  }

  if (tensor.dtype() == DataType::kInt64) {
    if (tensor.device().type() == Device::Type::kCpu) {
      const auto* src = static_cast<const int64_t*>(tensor.data());
      result.assign(src, src + n);
    } else {
      auto ret = aclrtMemcpyAsync(result.data(), n * sizeof(int64_t),
                                  tensor.data(), n * sizeof(int64_t),
                                  ACL_MEMCPY_DEVICE_TO_HOST, stream);
      assert(ret == ACL_SUCCESS && "`FIA`: D2H copy failed");
      ret = aclrtSynchronizeStream(stream);
      assert(ret == ACL_SUCCESS && "`FIA`: stream synchronize failed");
    }

    return result;
  }

  assert(false && "`FIA`: sequence tensor must be `int32` or `int64`");

  return result;
}

inline aclIntArray* CreateCumSeqLengths(const Tensor& cu_seqlens,
                                        aclrtStream stream) {
  auto values = ReadIntTensor(cu_seqlens, stream);
  assert(values.size() > 1 && "`FIA`: `cu_seqlens` must contain a batch");

  return aclCreateIntArray(values.data() + 1,
                           static_cast<int64_t>(values.size() - 1));
}

inline aclIntArray* CreateDiffSeqLengths(const Tensor& cu_seqlens,
                                         aclrtStream stream) {
  auto values = ReadIntTensor(cu_seqlens, stream);
  assert(values.size() > 1 && "`FIA`: `cu_seqlens` must contain a batch");

  std::vector<int64_t> lengths(values.size() - 1);
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    lengths[i] = values[i + 1] - values[i];
  }

  return aclCreateIntArray(lengths.data(),
                           static_cast<int64_t>(lengths.size()));
}

inline aclIntArray* CreateSeqLengths(const Tensor& seqlens,
                                     aclrtStream stream) {
  auto values = ReadIntTensor(seqlens, stream);

  return aclCreateIntArray(values.data(), static_cast<int64_t>(values.size()));
}

inline aclTensor* MakeCausalMask(void** mask_buf) {
  constexpr int64_t kMaskDim = 2048;
  const int64_t mask_elems = kMaskDim * kMaskDim;
  const auto mask_bytes = static_cast<size_t>(mask_elems);

  auto ret = aclrtMalloc(mask_buf, mask_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
  assert(ret == ACL_SUCCESS && "`FIA`: causal mask allocation failed");

  std::vector<uint8_t> host_mask(mask_elems);
  for (int64_t row = 0; row < kMaskDim; ++row) {
    for (int64_t col = 0; col < kMaskDim; ++col) {
      host_mask[row * kMaskDim + col] = (col > row) ? 1 : 0;
    }
  }

  ret = aclrtMemcpy(*mask_buf, mask_bytes, host_mask.data(), mask_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  assert(ret == ACL_SUCCESS && "`FIA`: causal mask upload failed");

  std::vector<int64_t> shape = {kMaskDim, kMaskDim};
  std::vector<int64_t> strides = {kMaskDim, 1};
  std::vector<int64_t> storage = {mask_elems};

  return aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()),
                         ACL_UINT8, strides.data(), 0, ACL_FORMAT_ND,
                         storage.data(), static_cast<int64_t>(storage.size()),
                         *mask_buf);
}

inline void ResolveSparseMode(bool is_causal, int64_t window_left,
                              int64_t window_right, int64_t& sparse_mode,
                              int64_t& pre_tokens, int64_t& next_tokens) {
  sparse_mode = 0;
  pre_tokens = 2147483647;
  next_tokens = 2147483647;

  if (is_causal) {
    if (window_left >= 0) {
      sparse_mode = 4;
      pre_tokens = window_left;
    } else {
      sparse_mode = 3;
    }
    next_tokens = 0;

    return;
  }

  if (window_left >= 0) pre_tokens = window_left;
  if (window_right >= 0) next_tokens = window_right;
}

}  // namespace infini::ops::ascend::fia

#endif
