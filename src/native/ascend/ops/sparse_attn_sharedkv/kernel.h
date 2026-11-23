#ifndef INFINI_OPS_ASCEND_SPARSE_ATTN_SHAREDKV_KERNEL_H_
#define INFINI_OPS_ASCEND_SPARSE_ATTN_SHAREDKV_KERNEL_H_
#include <stdexcept>
#include <string>
#include "base/sparse_attn_sharedkv.h"
#include "operator.h"
#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_sparse_attn_sharedkv.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif
namespace infini::ops {
#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <> class Operator<SparseAttnSharedkv, Device::Type::kAscend, 0> : public SparseAttnSharedkv {
 public:
  static bool supports(Tensor output, Tensor softmax_lse, const Tensor q,
      const Tensor ori_kv, const Tensor cmp_kv, const Tensor ori_sparse_indices,
      const Tensor cmp_sparse_indices, const Tensor ori_block_table,
      const Tensor cmp_block_table, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv, const Tensor sinks,
      const Tensor metadata, double, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t, std::string layout_q, std::string layout_kv,
      bool) {
    return output.device().type() == Device::Type::kAscend &&
      q.device() == output.device() && output.IsContiguous() && q.IsContiguous() &&
      output.dtype() == q.dtype() && output.shape() == q.shape() &&
      ori_kv.device() == output.device() && cmp_kv.device() == output.device() &&
      ori_sparse_indices.device() == output.device() && cmp_sparse_indices.device() == output.device() &&
      ori_block_table.device() == output.device() && cmp_block_table.device() == output.device() &&
      cu_seqlens_q.device() == output.device() && cu_seqlens_ori_kv.device() == output.device() &&
      cu_seqlens_cmp_kv.device() == output.device() && seqused_q.device() == output.device() &&
      seqused_kv.device() == output.device() && sinks.device() == output.device() &&
      metadata.device() == output.device() && softmax_lse.device() == output.device() &&
      softmax_lse.IsContiguous() && metadata.dtype() == DataType::kInt32 &&
      layout_q == "TND" && layout_kv == "PA_ND";
  }
  Operator(Tensor output, Tensor softmax_lse, const Tensor q,
      const Tensor ori_kv, const Tensor cmp_kv, const Tensor ori_sparse_indices,
      const Tensor cmp_sparse_indices, const Tensor ori_block_table,
      const Tensor cmp_block_table, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv, const Tensor sinks,
      const Tensor metadata, double softmax_scale, int64_t cmp_ratio,
      int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_kv_stride,
      int64_t cmp_kv_stride, int64_t ori_win_left, int64_t ori_win_right,
      std::string layout_q, std::string layout_kv, bool return_softmax_lse)
      : SparseAttnSharedkv(output, softmax_lse, q, ori_kv, cmp_kv,
          ori_sparse_indices, cmp_sparse_indices, ori_block_table, cmp_block_table,
          cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
          seqused_kv, sinks, metadata, softmax_scale, cmp_ratio, ori_mask_mode,
          cmp_mask_mode, ori_kv_stride, cmp_kv_stride, ori_win_left,
          ori_win_right, layout_q, layout_kv, return_softmax_lse),
        output_cache_(output), lse_cache_(softmax_lse), q_cache_(q),
        ori_kv_cache_(ori_kv), cmp_kv_cache_(cmp_kv), ori_sparse_cache_(ori_sparse_indices),
        cmp_sparse_cache_(cmp_sparse_indices), ori_block_cache_(ori_block_table),
        cmp_block_cache_(cmp_block_table), qlens_cache_(cu_seqlens_q),
        ori_lens_cache_(cu_seqlens_ori_kv), cmp_lens_cache_(cu_seqlens_cmp_kv),
        seqused_q_cache_(seqused_q), seqused_kv_cache_(seqused_kv), sinks_cache_(sinks),
        metadata_cache_(metadata), t_output_(output_cache_.get(output.data())),
        t_lse_(lse_cache_.get(softmax_lse.data())), t_q_(q_cache_.get(const_cast<void*>(q.data()))),
        t_ori_kv_(ori_kv.numel() == 0 ? nullptr : ori_kv_cache_.get(const_cast<void*>(ori_kv.data()))),
        t_cmp_kv_(cmp_kv.numel() == 0 ? nullptr : cmp_kv_cache_.get(const_cast<void*>(cmp_kv.data()))),
        t_ori_sparse_(ori_sparse_indices.numel() == 0 ? nullptr : ori_sparse_cache_.get(const_cast<void*>(ori_sparse_indices.data()))),
        t_cmp_sparse_(cmp_sparse_indices.numel() == 0 ? nullptr : cmp_sparse_cache_.get(const_cast<void*>(cmp_sparse_indices.data()))),
        t_ori_block_(ori_block_table.numel() == 0 ? nullptr : ori_block_cache_.get(const_cast<void*>(ori_block_table.data()))),
        t_cmp_block_(cmp_block_table.numel() == 0 ? nullptr : cmp_block_cache_.get(const_cast<void*>(cmp_block_table.data()))),
        t_qlens_(cu_seqlens_q.numel() == 0 ? nullptr : qlens_cache_.get(const_cast<void*>(cu_seqlens_q.data()))),
        t_ori_lens_(cu_seqlens_ori_kv.numel() == 0 ? nullptr : ori_lens_cache_.get(const_cast<void*>(cu_seqlens_ori_kv.data()))),
        t_cmp_lens_(cu_seqlens_cmp_kv.numel() == 0 ? nullptr : cmp_lens_cache_.get(const_cast<void*>(cu_seqlens_cmp_kv.data()))),
        t_seqused_q_(seqused_q.numel() == 0 ? nullptr : seqused_q_cache_.get(const_cast<void*>(seqused_q.data()))),
        t_seqused_kv_(seqused_kv_cache_.get(const_cast<void*>(seqused_kv.data()))),
        t_sinks_(sinks_cache_.get(const_cast<void*>(sinks.data()))),
        t_metadata_(metadata_cache_.get(const_cast<void*>(metadata.data()))),
        softmax_scale_(softmax_scale), cmp_ratio_(cmp_ratio), ori_mask_mode_(ori_mask_mode),
        cmp_mask_mode_(cmp_mask_mode), ori_kv_stride_(ori_kv_stride),
        cmp_kv_stride_(cmp_kv_stride), ori_win_left_(ori_win_left),
        ori_win_right_(ori_win_right), layout_q_(std::move(layout_q)),
        layout_kv_(std::move(layout_kv)), return_softmax_lse_(return_softmax_lse) {
    if (!supports(output, softmax_lse, q, ori_kv, cmp_kv, ori_sparse_indices,
        cmp_sparse_indices, ori_block_table, cmp_block_table, cu_seqlens_q,
        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv, sinks,
        metadata, softmax_scale_, cmp_ratio_, ori_mask_mode_, cmp_mask_mode_,
        ori_kv_stride_, cmp_kv_stride_, ori_win_left_, ori_win_right_, layout_q_,
        layout_kv_, return_softmax_lse_)) {
      throw std::invalid_argument("unsupported Ascend SparseAttnSharedkv operands");
    }
    ascend::CheckAcl(aclnnSparseAttnSharedkvGetWorkspaceSize(
        t_q_, t_ori_kv_, t_cmp_kv_, t_ori_sparse_, t_cmp_sparse_, t_ori_block_,
        t_cmp_block_, t_qlens_, t_ori_lens_, t_cmp_lens_, t_seqused_q_,
        t_seqused_kv_, t_sinks_, t_metadata_, softmax_scale_, cmp_ratio_,
        ori_mask_mode_, cmp_mask_mode_, ori_kv_stride_, cmp_kv_stride_,
        ori_win_left_, ori_win_right_, const_cast<char*>(layout_q_.c_str()),
        const_cast<char*>(layout_kv_.c_str()), return_softmax_lse_, t_output_,
        t_lse_, &workspace_size_, &executor_), "SparseAttnSharedkv workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
        "SparseAttnSharedkv repeatable executor");
  }
  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    output_cache_.release(); lse_cache_.release(); q_cache_.release();
    ori_kv_cache_.release(); cmp_kv_cache_.release(); ori_sparse_cache_.release();
    cmp_sparse_cache_.release(); ori_block_cache_.release(); cmp_block_cache_.release();
    qlens_cache_.release(); ori_lens_cache_.release(); cmp_lens_cache_.release();
    seqused_q_cache_.release(); seqused_kv_cache_.release(); sinks_cache_.release();
    metadata_cache_.release();
  }
  bool capture_safe() const override { return true; }
  void operator()(Tensor output, Tensor softmax_lse, const Tensor q,
      const Tensor ori_kv, const Tensor cmp_kv, const Tensor ori_sparse_indices,
      const Tensor cmp_sparse_indices, const Tensor ori_block_table,
      const Tensor cmp_block_table, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv, const Tensor sinks,
      const Tensor metadata, double, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t, std::string, std::string, bool) const override {
    set_input(0, t_q_, q, "SparseAttnSharedkv q");
    set_input(1, t_ori_kv_, ori_kv, "SparseAttnSharedkv ori_kv");
    set_input(2, t_cmp_kv_, cmp_kv, "SparseAttnSharedkv cmp_kv");
    set_input(3, t_ori_sparse_, ori_sparse_indices, "SparseAttnSharedkv ori_sparse");
    set_input(4, t_cmp_sparse_, cmp_sparse_indices, "SparseAttnSharedkv cmp_sparse");
    set_input(5, t_ori_block_, ori_block_table, "SparseAttnSharedkv ori_block");
    set_input(6, t_cmp_block_, cmp_block_table, "SparseAttnSharedkv cmp_block");
    set_input(7, t_qlens_, cu_seqlens_q, "SparseAttnSharedkv qlens");
    set_input(8, t_ori_lens_, cu_seqlens_ori_kv, "SparseAttnSharedkv ori_lens");
    set_input(9, t_cmp_lens_, cu_seqlens_cmp_kv, "SparseAttnSharedkv cmp_lens");
    set_input(10, t_seqused_q_, seqused_q, "SparseAttnSharedkv seqused_q");
    set_input(11, t_seqused_kv_, seqused_kv, "SparseAttnSharedkv seqused_kv");
    set_input(12, t_sinks_, sinks, "SparseAttnSharedkv sinks");
    set_input(13, t_metadata_, metadata, "SparseAttnSharedkv metadata");
    ascend::CheckAcl(AclSetOutputTensorAddr(executor_, 0, t_output_, output.data()),
        "SparseAttnSharedkv output");
    ascend::CheckAcl(AclSetOutputTensorAddr(executor_, 1, t_lse_, softmax_lse.data()),
        "SparseAttnSharedkv lse");
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size_, "sparse_attn_sharedkv");
    ascend::CheckAcl(aclnnSparseAttnSharedkv(arena.buf, workspace_size_, executor_, stream),
        "SparseAttnSharedkv execution");
  }
 private:
  void set_input(int index, aclTensor* tensor, const Tensor& value, const char* name) const {
    if (tensor != nullptr) {
      ascend::CheckAcl(AclSetInputTensorAddr(executor_, index, tensor, const_cast<void*>(value.data())), name);
    }
  }
  mutable ascend::AclTensorCache output_cache_, lse_cache_, q_cache_, ori_kv_cache_, cmp_kv_cache_,
      ori_sparse_cache_, cmp_sparse_cache_, ori_block_cache_, cmp_block_cache_, qlens_cache_,
      ori_lens_cache_, cmp_lens_cache_, seqused_q_cache_, seqused_kv_cache_, sinks_cache_, metadata_cache_;
  aclTensor *t_output_{nullptr}, *t_lse_{nullptr}, *t_q_{nullptr}, *t_ori_kv_{nullptr}, *t_cmp_kv_{nullptr},
      *t_ori_sparse_{nullptr}, *t_cmp_sparse_{nullptr}, *t_ori_block_{nullptr}, *t_cmp_block_{nullptr},
      *t_qlens_{nullptr}, *t_ori_lens_{nullptr}, *t_cmp_lens_{nullptr}, *t_seqused_q_{nullptr},
      *t_seqused_kv_{nullptr}, *t_sinks_{nullptr}, *t_metadata_{nullptr};
  double softmax_scale_;
  int64_t cmp_ratio_, ori_mask_mode_, cmp_mask_mode_, ori_kv_stride_, cmp_kv_stride_, ori_win_left_, ori_win_right_;
  std::string layout_q_, layout_kv_;
  bool return_softmax_lse_;
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <> class Operator<SparseAttnSharedkv, Device::Type::kAscend, 0> : public SparseAttnSharedkv {
 public:
  template <typename... Args> static bool supports(const Args&...) { return false; }
  Operator(Tensor, Tensor, const Tensor, const Tensor, const Tensor, const Tensor, const Tensor,
      const Tensor, const Tensor, const Tensor, const Tensor, const Tensor, const Tensor,
      const Tensor, const Tensor, const Tensor, const Tensor, double, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t, int64_t, std::string, std::string, bool) {
    throw std::runtime_error("SparseAttnSharedkv requires custom transformer");
  }
  void operator()(Tensor, Tensor, const Tensor, const Tensor, const Tensor, const Tensor, const Tensor,
      const Tensor, const Tensor, const Tensor, const Tensor, const Tensor, const Tensor, const Tensor,
      const Tensor, const Tensor, const Tensor, double, int64_t, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, std::string, std::string, bool) const override {
    throw std::runtime_error("SparseAttnSharedkv unavailable");
  }
};
#endif
}
#endif
