#ifndef INFINI_OPS_ASCEND_COMPRESSOR_METADATA_KERNEL_H_
#define INFINI_OPS_ASCEND_COMPRESSOR_METADATA_KERNEL_H_

#include <stdexcept>

#include "base/compressor_metadata.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_compressor_metadata.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER

template <>
class Operator<CompressorMetadata, Device::Type::kAscend, 0>
    : public CompressorMetadata {
 public:
  static bool supports(
      Tensor compress_cos, Tensor compress_sin, Tensor slot_mapping,
      const Tensor rope_cos, const Tensor rope_sin, const Tensor cu_seqlens,
      const Tensor start_pos, const Tensor kv_block_table,
      int64_t kv_block_size, int64_t slot_mapping_format,
      int64_t compress_ratio, int64_t num_reqs_actual) {
    if (rope_cos.device().type() != Device::Type::kAscend ||
        rope_cos.ndim() != 2 || rope_sin.shape() != rope_cos.shape() ||
        rope_sin.dtype() != rope_cos.dtype() || cu_seqlens.ndim() != 1 ||
        start_pos.ndim() != 1 || kv_block_table.ndim() != 2 ||
        compress_cos.device() != rope_cos.device() ||
        compress_sin.device() != rope_cos.device() ||
        slot_mapping.device() != rope_cos.device() ||
        rope_sin.device() != rope_cos.device() ||
        cu_seqlens.device() != rope_cos.device() ||
        start_pos.device() != rope_cos.device() ||
        kv_block_table.device() != rope_cos.device() ||
        compress_cos.dtype() != rope_cos.dtype() ||
        compress_sin.dtype() != rope_cos.dtype() ||
        slot_mapping.dtype() != DataType::kInt32 || kv_block_size <= 0 ||
        (slot_mapping_format != 1 && slot_mapping_format != 2) ||
        compress_ratio <= 0 || num_reqs_actual <= 0 ||
        cu_seqlens.shape()[0] <= static_cast<uint64_t>(num_reqs_actual) ||
        start_pos.shape()[0] < static_cast<uint64_t>(num_reqs_actual) ||
        kv_block_table.shape()[0] < static_cast<uint64_t>(num_reqs_actual) ||
        !compress_cos.IsContiguous() || !compress_sin.IsContiguous() ||
        !slot_mapping.IsContiguous() || !rope_cos.IsContiguous() ||
        !rope_sin.IsContiguous() || !cu_seqlens.IsContiguous() ||
        !start_pos.IsContiguous() || !kv_block_table.IsContiguous()) {
      return false;
    }

    const auto rows = compress_cos.shape()[0];
    if (compress_cos.shape() !=
            Tensor::Shape{rows, 1, 1, rope_cos.shape()[1]} ||
        compress_sin.shape() != compress_cos.shape()) {
      return false;
    }
    return slot_mapping_format == 2
               ? slot_mapping.shape() == Tensor::Shape{rows, 2}
               : slot_mapping.shape() == Tensor::Shape{rows};
  }

  Operator(Tensor compress_cos, Tensor compress_sin, Tensor slot_mapping,
           const Tensor rope_cos, const Tensor rope_sin,
           const Tensor cu_seqlens, const Tensor start_pos,
           const Tensor kv_block_table, int64_t kv_block_size,
           int64_t slot_mapping_format, int64_t compress_ratio,
           int64_t num_reqs_actual)
      : CompressorMetadata(compress_cos, compress_sin, slot_mapping, rope_cos,
                           rope_sin, cu_seqlens, start_pos, kv_block_table,
                           kv_block_size, slot_mapping_format, compress_ratio,
                           num_reqs_actual),
        compress_cos_cache_(compress_cos),
        compress_sin_cache_(compress_sin),
        slot_mapping_cache_(slot_mapping),
        rope_cos_cache_(rope_cos),
        rope_sin_cache_(rope_sin),
        cu_seqlens_cache_(cu_seqlens),
        start_pos_cache_(start_pos),
        kv_block_table_cache_(kv_block_table),
        kv_block_size_(kv_block_size),
        slot_mapping_format_(slot_mapping_format),
        compress_ratio_(compress_ratio),
        num_reqs_actual_(num_reqs_actual) {
    if (!supports(compress_cos, compress_sin, slot_mapping, rope_cos,
                  rope_sin, cu_seqlens, start_pos, kv_block_table,
                  kv_block_size, slot_mapping_format, compress_ratio,
                  num_reqs_actual)) {
      throw std::invalid_argument(
          "unsupported Ascend CompressorMetadata operands");
    }
  }

  bool capture_safe() const override { return false; }

  void operator()(
      Tensor compress_cos, Tensor compress_sin, Tensor slot_mapping,
      const Tensor rope_cos, const Tensor rope_sin, const Tensor cu_seqlens,
      const Tensor start_pos, const Tensor kv_block_table, int64_t,
      int64_t, int64_t, int64_t) const override {
    auto* t_compress_cos = compress_cos_cache_.get(compress_cos.data());
    auto* t_compress_sin = compress_sin_cache_.get(compress_sin.data());
    auto* t_slot_mapping = slot_mapping_cache_.get(slot_mapping.data());
    auto* t_rope_cos = rope_cos_cache_.get(const_cast<void*>(rope_cos.data()));
    auto* t_rope_sin = rope_sin_cache_.get(const_cast<void*>(rope_sin.data()));
    auto* t_cu_seqlens =
        cu_seqlens_cache_.get(const_cast<void*>(cu_seqlens.data()));
    auto* t_start_pos =
        start_pos_cache_.get(const_cast<void*>(start_pos.data()));
    auto* t_kv_block_table =
        kv_block_table_cache_.get(const_cast<void*>(kv_block_table.data()));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ascend::CheckAcl(
        aclnnCompressorMetadataGetWorkspaceSize(
            t_rope_cos, t_rope_sin, t_cu_seqlens, t_start_pos,
            t_kv_block_table, kv_block_size_, slot_mapping_format_,
            compress_ratio_, num_reqs_actual_, t_compress_cos,
            t_compress_sin, t_slot_mapping, &workspace_size, &executor),
        "CompressorMetadata workspace query");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size, "compressor_metadata");
    ascend::CheckAcl(
        aclnnCompressorMetadata(arena.buf, workspace_size, executor, stream),
        "CompressorMetadata execution");
  }

 private:
  mutable ascend::AclTensorCache compress_cos_cache_;
  mutable ascend::AclTensorCache compress_sin_cache_;
  mutable ascend::AclTensorCache slot_mapping_cache_;
  mutable ascend::AclTensorCache rope_cos_cache_;
  mutable ascend::AclTensorCache rope_sin_cache_;
  mutable ascend::AclTensorCache cu_seqlens_cache_;
  mutable ascend::AclTensorCache start_pos_cache_;
  mutable ascend::AclTensorCache kv_block_table_cache_;
  int64_t kv_block_size_;
  int64_t slot_mapping_format_;
  int64_t compress_ratio_;
  int64_t num_reqs_actual_;
};

#else

template <>
class Operator<CompressorMetadata, Device::Type::kAscend, 0>
    : public CompressorMetadata {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor compress_cos, Tensor compress_sin, Tensor slot_mapping,
           const Tensor rope_cos, const Tensor rope_sin,
           const Tensor cu_seqlens, const Tensor start_pos,
           const Tensor kv_block_table, int64_t kv_block_size,
           int64_t slot_mapping_format, int64_t compress_ratio,
           int64_t num_reqs_actual)
      : CompressorMetadata(compress_cos, compress_sin, slot_mapping, rope_cos,
                           rope_sin, cu_seqlens, start_pos, kv_block_table,
                           kv_block_size, slot_mapping_format, compress_ratio,
                           num_reqs_actual) {
    throw std::runtime_error(
        "CompressorMetadata requires INFINI_ASCEND_CUSTOM_OPAPI_ROOT");
  }

  void operator()(Tensor, Tensor, Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, const Tensor, int64_t, int64_t,
                  int64_t, int64_t) const override {
    throw std::runtime_error(
        "CompressorMetadata Ascend provider is unavailable");
  }
};

#endif

}  // namespace infini::ops

#endif
