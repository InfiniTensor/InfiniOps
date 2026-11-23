#ifndef INFINI_OPS_ASCEND_COMPRESSOR_KERNEL_H_
#define INFINI_OPS_ASCEND_COMPRESSOR_KERNEL_H_

#include <stdexcept>

#include "base/compressor.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_compressor.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <>
class Operator<Compressor, Device::Type::kAscend, 0> : public Compressor {
 public:
  static bool supports(
      Tensor cmp_kv, Tensor state_cache, const Tensor x, const Tensor wkv,
      const Tensor wgate, const Tensor ape, const Tensor norm_weight,
      const Tensor rope_sin, const Tensor rope_cos,
      const Tensor state_block_table, const Tensor cu_seqlens,
      const Tensor start_pos, int64_t rope_head_dim, int64_t compress_ratio,
      int64_t coff, double norm_eps, int64_t rotary_mode,
      int64_t cache_mode) {
    if (x.device().type() != Device::Type::kAscend ||
        x.dtype() != DataType::kBFloat16 ||
        cmp_kv.dtype() != DataType::kBFloat16 ||
        wkv.dtype() != DataType::kBFloat16 ||
        wgate.dtype() != DataType::kBFloat16 ||
        state_cache.dtype() != DataType::kFloat32 ||
        ape.dtype() != DataType::kFloat32 ||
        norm_weight.dtype() != DataType::kBFloat16 ||
        (rope_sin.dtype() != DataType::kBFloat16 &&
         rope_sin.dtype() != DataType::kFloat32) ||
        rope_cos.dtype() != rope_sin.dtype() ||
        state_block_table.dtype() != DataType::kInt32 ||
        cu_seqlens.dtype() != DataType::kInt32 ||
        start_pos.dtype() != DataType::kInt32 || x.ndim() != 2 ||
        wkv.ndim() != 2 || wgate.ndim() != 2 || state_cache.ndim() != 3 ||
        ape.ndim() != 2 || norm_weight.ndim() != 1 ||
        rope_sin.ndim() != 2 || rope_cos.ndim() != 2 ||
        state_block_table.ndim() != 2 || cu_seqlens.ndim() != 1 ||
        start_pos.ndim() != 1 || rope_head_dim <= 0 ||
        compress_ratio <= 0 || coff <= 0 || norm_eps < 0.0 ||
        rotary_mode < 0 || rotary_mode > 3 || cache_mode != 1) {
      return false;
    }

    const auto device = x.device();
    if (cmp_kv.device() != device || state_cache.device() != device ||
        wkv.device() != device || wgate.device() != device ||
        ape.device() != device || norm_weight.device() != device ||
        rope_sin.device() != device || rope_cos.device() != device ||
        state_block_table.device() != device || cu_seqlens.device() != device ||
        start_pos.device() != device || !cmp_kv.IsContiguous() ||
        !x.IsContiguous() || !wkv.IsContiguous() || !wgate.IsContiguous() ||
        !ape.IsContiguous() || !norm_weight.IsContiguous() ||
        !rope_sin.IsContiguous() || !rope_cos.IsContiguous() ||
        !state_block_table.IsContiguous() || !cu_seqlens.IsContiguous() ||
        !start_pos.IsContiguous()) {
      return false;
    }

    const auto hidden = x.shape()[1];
    const auto head_dim = norm_weight.shape()[0];
    const auto compressed_dim = static_cast<uint64_t>(coff) * head_dim;
    if (rope_head_dim > static_cast<int64_t>(head_dim) ||
        wkv.shape() != Tensor::Shape{compressed_dim, hidden} ||
        wgate.shape() != wkv.shape() ||
        ape.shape() != Tensor::Shape{static_cast<uint64_t>(compress_ratio),
                                     compressed_dim} ||
        rope_sin.shape() != rope_cos.shape() ||
        rope_sin.shape()[1] != static_cast<uint64_t>(rope_head_dim) ||
        cmp_kv.shape() != Tensor::Shape{rope_sin.shape()[0], head_dim} ||
        state_cache.shape()[2] != 2 * compressed_dim ||
        cu_seqlens.shape()[0] < 2 || start_pos.shape()[0] < 1 ||
        state_block_table.shape()[0] < start_pos.shape()[0]) {
      return false;
    }
    return true;
  }

  Operator(Tensor cmp_kv, Tensor state_cache, const Tensor x,
           const Tensor wkv, const Tensor wgate, const Tensor ape,
           const Tensor norm_weight, const Tensor rope_sin,
           const Tensor rope_cos, const Tensor state_block_table,
           const Tensor cu_seqlens, const Tensor start_pos,
           int64_t rope_head_dim, int64_t compress_ratio, int64_t coff,
           double norm_eps, int64_t rotary_mode, int64_t cache_mode)
      : Compressor(cmp_kv, state_cache, x, wkv, wgate, ape, norm_weight,
                   rope_sin, rope_cos, state_block_table, cu_seqlens,
                   start_pos, rope_head_dim, compress_ratio, coff, norm_eps,
                   rotary_mode, cache_mode),
        cmp_kv_cache_(cmp_kv),
        state_cache_cache_(state_cache),
        x_cache_(x),
        wkv_cache_(wkv),
        wgate_cache_(wgate),
        ape_cache_(ape),
        norm_weight_cache_(norm_weight),
        rope_sin_cache_(rope_sin),
        rope_cos_cache_(rope_cos),
        state_block_table_cache_(state_block_table),
        cu_seqlens_cache_(cu_seqlens),
        start_pos_cache_(start_pos),
        t_cmp_kv_(cmp_kv_cache_.get(cmp_kv.data())),
        t_state_cache_(state_cache_cache_.get(state_cache.data())),
        t_x_(x_cache_.get(const_cast<void*>(x.data()))),
        t_wkv_(wkv_cache_.get(const_cast<void*>(wkv.data()))),
        t_wgate_(wgate_cache_.get(const_cast<void*>(wgate.data()))),
        t_ape_(ape_cache_.get(const_cast<void*>(ape.data()))),
        t_norm_weight_(
            norm_weight_cache_.get(const_cast<void*>(norm_weight.data()))),
        t_rope_sin_(rope_sin_cache_.get(const_cast<void*>(rope_sin.data()))),
        t_rope_cos_(rope_cos_cache_.get(const_cast<void*>(rope_cos.data()))),
        t_state_block_table_(state_block_table_cache_.get(
            const_cast<void*>(state_block_table.data()))),
        t_cu_seqlens_(
            cu_seqlens_cache_.get(const_cast<void*>(cu_seqlens.data()))),
        t_start_pos_(start_pos_cache_.get(const_cast<void*>(start_pos.data()))) {
    if (!supports(cmp_kv, state_cache, x, wkv, wgate, ape, norm_weight,
                  rope_sin, rope_cos, state_block_table, cu_seqlens,
                  start_pos, rope_head_dim, compress_ratio, coff, norm_eps,
                  rotary_mode, cache_mode)) {
      throw std::invalid_argument("unsupported Ascend Compressor operands");
    }

    const auto state_stride_dim0 =
        static_cast<int64_t>(state_cache.strides()[0]);
    ascend::CheckAcl(
        aclnnCompressorGetWorkspaceSize(
            t_x_, t_wkv_, t_wgate_, t_state_cache_, t_ape_, t_norm_weight_,
            t_rope_sin_, t_rope_cos_, t_state_block_table_, t_cu_seqlens_,
            nullptr, t_start_pos_, rope_head_dim, compress_ratio, coff,
            norm_eps, rotary_mode, cache_mode, state_stride_dim0, t_cmp_kv_,
            &workspace_size_, &executor_),
        "Compressor workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "Compressor repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    cmp_kv_cache_.release();
    state_cache_cache_.release();
    x_cache_.release();
    wkv_cache_.release();
    wgate_cache_.release();
    ape_cache_.release();
    norm_weight_cache_.release();
    rope_sin_cache_.release();
    rope_cos_cache_.release();
    state_block_table_cache_.release();
    cu_seqlens_cache_.release();
    start_pos_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(
      Tensor cmp_kv, Tensor state_cache, const Tensor x, const Tensor wkv,
      const Tensor wgate, const Tensor ape, const Tensor norm_weight,
      const Tensor rope_sin, const Tensor rope_cos,
      const Tensor state_block_table, const Tensor cu_seqlens,
      const Tensor start_pos, int64_t rope_head_dim, int64_t compress_ratio,
      int64_t coff, double norm_eps, int64_t rotary_mode,
      int64_t cache_mode) const override {
    if (!supports(cmp_kv, state_cache, x, wkv, wgate, ape, norm_weight,
                  rope_sin, rope_cos, state_block_table, cu_seqlens,
                  start_pos, rope_head_dim, compress_ratio, coff, norm_eps,
                  rotary_mode, cache_mode)) {
      throw std::invalid_argument("unsupported Ascend Compressor operands");
    }

    set_input(0, t_x_, x, "x");
    set_input(1, t_wkv_, wkv, "wkv");
    set_input(2, t_wgate_, wgate, "wgate");
    ascend::CheckAcl(AclSetInputTensorAddr(executor_, 3, t_state_cache_,
                                           state_cache.data()),
                     "Compressor state_cache address");
    set_input(4, t_ape_, ape, "ape");
    set_input(5, t_norm_weight_, norm_weight, "norm_weight");
    set_input(6, t_rope_sin_, rope_sin, "rope_sin");
    set_input(7, t_rope_cos_, rope_cos, "rope_cos");
    set_input(8, t_state_block_table_, state_block_table,
              "state_block_table");
    set_input(9, t_cu_seqlens_, cu_seqlens, "cu_seqlens");
    set_input(11, t_start_pos_, start_pos, "start_pos");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 0, t_cmp_kv_, cmp_kv.data()),
        "Compressor output address");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "compressor");
    ascend::CheckAcl(
        aclnnCompressor(arena.buf, workspace_size_, executor_, stream),
        "Compressor execution");
  }

 private:
  void set_input(int64_t index, aclTensor* tensor, const Tensor value,
                 const char* name) const {
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, index, tensor,
                         const_cast<void*>(value.data())),
                     name);
  }

  mutable ascend::AclTensorCache cmp_kv_cache_;
  mutable ascend::AclTensorCache state_cache_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache wkv_cache_;
  mutable ascend::AclTensorCache wgate_cache_;
  mutable ascend::AclTensorCache ape_cache_;
  mutable ascend::AclTensorCache norm_weight_cache_;
  mutable ascend::AclTensorCache rope_sin_cache_;
  mutable ascend::AclTensorCache rope_cos_cache_;
  mutable ascend::AclTensorCache state_block_table_cache_;
  mutable ascend::AclTensorCache cu_seqlens_cache_;
  mutable ascend::AclTensorCache start_pos_cache_;
  aclTensor* t_cmp_kv_{nullptr};
  aclTensor* t_state_cache_{nullptr};
  aclTensor* t_x_{nullptr};
  aclTensor* t_wkv_{nullptr};
  aclTensor* t_wgate_{nullptr};
  aclTensor* t_ape_{nullptr};
  aclTensor* t_norm_weight_{nullptr};
  aclTensor* t_rope_sin_{nullptr};
  aclTensor* t_rope_cos_{nullptr};
  aclTensor* t_state_block_table_{nullptr};
  aclTensor* t_cu_seqlens_{nullptr};
  aclTensor* t_start_pos_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <>
class Operator<Compressor, Device::Type::kAscend, 0> : public Compressor {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor cmp_kv, Tensor state_cache, const Tensor x,
           const Tensor wkv, const Tensor wgate, const Tensor ape,
           const Tensor norm_weight, const Tensor rope_sin,
           const Tensor rope_cos, const Tensor state_block_table,
           const Tensor cu_seqlens, const Tensor start_pos,
           int64_t rope_head_dim, int64_t compress_ratio, int64_t coff,
           double norm_eps, int64_t rotary_mode, int64_t cache_mode)
      : Compressor(cmp_kv, state_cache, x, wkv, wgate, ape, norm_weight,
                   rope_sin, rope_cos, state_block_table, cu_seqlens,
                   start_pos, rope_head_dim, compress_ratio, coff, norm_eps,
                   rotary_mode, cache_mode) {
    throw std::runtime_error(
        "Compressor requires INFINI_ASCEND_CUSTOM_OPAPI_ROOT");
  }

  void operator()(Tensor, Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, const Tensor, int64_t, int64_t,
                  int64_t, double, int64_t, int64_t) const override {
    throw std::runtime_error("Compressor Ascend provider is unavailable");
  }
};
#endif

}  // namespace infini::ops

#endif
