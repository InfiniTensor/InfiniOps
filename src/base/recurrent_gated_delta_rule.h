#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

#include "operator.h"
namespace infini::ops {
// Gated delta rule recurrence over a state pool, in two forms.
//
// Batched single-token decode: `q`, `k`, `v` and `out` are `[B, 1, H, D]`.
// Batch item b reads `initial_state[initial_state_indices[b]]`, or
// `initial_state[b]` without indices, and ends in
// `initial_state[final_state_indices[b]]` or in `final_state[b]`.
//
// Packed sequences with an FP32 state: `q`, `k`, `v` and `out` are
// `[1, T, H, 128]` — the head dim of the Clover model, and of both
// providers. Request r owns the tokens `[cu_seqlens[r], cu_seqlens[r+1])`
// and runs the recurrence token by token. State and gates are F32 whatever
// the data dtype is.
//
// State-slot contract of both forms:
// - The initial slot is only read. The final state is written to the final
//   slot; when the two differ, the initial slot is left unchanged, so a
//   request forks a state instead of moving it. Equal slots update in place.
// - Requests run concurrently. Within one call, final slots must be pairwise
//   distinct and must not be the initial slot of any request; several requests
//   may read the same initial slot. A separate `final_state` tensor leaves the
//   pool read-only.
// - A request with an out-of-range initial or final slot writes nothing to the
//   pool and zeroes its output. A packed request with a malformed token range
//   leaves its output untouched.
//
// Each step applies `state *= exp(g)`, `state += (v - state @ k) * beta * k`
// and `out = state @ q`. `use_qk_l2norm` L2-normalizes q and k with eps 1e-6;
// q is always scaled by 1/sqrt(D).
class RecurrentGatedDeltaRule : public Operator<RecurrentGatedDeltaRule> {
 public:
  RecurrentGatedDeltaRule(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out, std::optional<Tensor> final_state,
                          bool use_qk_l2norm)
      : norm_(use_qk_l2norm),
        metadata_{q,
                  k,
                  v,
                  g,
                  beta,
                  initial_state,
                  initial_state_indices,
                  final_state_indices,
                  out,
                  final_state} {
    const auto& views = metadata_;
    Validate(views);
    device_ = q.device();
    data_type_ = q.dtype();
    gate_type_ = g.dtype();
    B_ = q.shape()[0];
    Hk_ = q.shape()[2];
    Hv_ = v.shape()[2];
    D_ = q.shape()[3];
    pool_ = initial_state.shape()[0];
  }
  // Packed form; `metadata_` holds [q, k, v, g, beta, state, cu_seqlens,
  // initial indices, final indices, out].
  RecurrentGatedDeltaRule(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state, const Tensor cu_seqlens,
                          const Tensor initial_slots,
                          const Tensor final_slots, Tensor out,
                          bool use_qk_l2norm)
      : norm_(use_qk_l2norm),
        metadata_{q,
                  k,
                  v,
                  g,
                  beta,
                  initial_state,
                  cu_seqlens,
                  initial_slots,
                  final_slots,
                  out} {
    ValidatePacked(metadata_);
    device_ = q.device();
    data_type_ = q.dtype();
    gate_type_ = g.dtype();
    tokens_ = q.shape()[1];
    Hk_ = q.shape()[2];
    Hv_ = v.shape()[2];
    D_ = q.shape()[3];
    pool_ = initial_state.shape()[0];
    requests_ = cu_seqlens.numel() - 1;
  }
  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out, std::optional<Tensor> final_state,
                          bool use_qk_l2norm) const = 0;
  // Packed form. Providers that do not implement it keep the default.
  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state, const Tensor cu_seqlens,
                          const Tensor initial_slots,
                          const Tensor final_slots, Tensor out,
                          bool use_qk_l2norm) const {
    throw std::invalid_argument(
        "this RecurrentGatedDeltaRule provider does not support packed "
        "sequences");
  }

 protected:
  using Views = std::array<std::optional<Tensor>, 10>;
  Device device_;
  DataType data_type_, gate_type_;
  bool norm_;
  size_t B_{0}, Hk_{0}, Hv_{0}, D_{0}, pool_{0};
  size_t tokens_{0}, requests_{0};
  const Views metadata_;
  static bool floating(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32;
  }
  static void Validate(const Views& a) {
    const auto &q = *a[0], &k = *a[1], &v = *a[2], &g = *a[3], &beta = *a[4],
               &state = *a[5], &out = *a[8];
    assert(floating(q.dtype()) && floating(g.dtype()) &&
           beta.dtype() == g.dtype() &&
           "recurrent delta data/gates support F16/BF16/F32");
    assert(q.ndim() == 4 && k.shape() == q.shape() && v.ndim() == 4 &&
           out.shape() == v.shape() && state.ndim() == 4 && g.ndim() == 3 &&
           beta.shape() == g.shape() && "recurrent delta rank/shape mismatch");
    const auto B = q.shape()[0], Hk = q.shape()[2], Hv = v.shape()[2],
               D = q.shape()[3], Dv = v.shape()[3], T = q.shape()[1];
    assert((v.shape()[0] == B && v.shape()[1] == T && Hk && Hv &&
            Hv % Hk == 0 && D && Dv && g.shape() == Tensor::Shape{B, T, Hv}) &&
           "recurrent delta data/gate shape or head grouping mismatch");
    assert(a[7].has_value() != a[9].has_value() &&
           "recurrent delta requires exactly one final destination mode");
    assert((state.shape()[1] == Hv && state.shape()[2] == Dv &&
            state.shape()[3] == D && (a[6] || state.shape()[0] >= B) &&
            (!a[9] || a[9]->shape() == Tensor::Shape{B, Hv, Dv, D})) &&
           "recurrent delta state layout must be pool/B,Hv,Dv,Dk");
    for (size_t i = 0; i < a.size(); ++i) {
      if (!a[i]) continue;
      assert(a[i]->device() == q.device() && "recurrent delta device mismatch");
      for (auto stride : a[i]->strides())
        assert(stride >= 0 && "recurrent delta negative stride unsupported");
      if (i == 6 || i == 7) {
        assert((a[i]->shape() == Tensor::Shape{B} && a[i]->IsContiguous() &&
                (a[i]->dtype() == DataType::kInt32 ||
                 a[i]->dtype() == DataType::kInt64)) &&
               "recurrent delta indices require contiguous I32/I64 B-vector");
      } else if (i != 3 && i != 4) {
        assert(a[i]->dtype() == q.dtype() &&
               "recurrent delta data dtype mismatch");
      }
      assert((!(i <= 2 || i == 8) || a[i]->strides()[3] == 1) &&
             "recurrent delta innermost data dimension must be contiguous");
      assert((!((i == 5 && a[7]) || i == 8 || i == 9) ||
              !a[i]->HasBroadcastDim()) &&
             "recurrent delta broadcast destination unsupported");
    }
    if (state.shape()[0] >
        static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error(
          "recurrent delta pool exceeds int64 index extent");
  }
  static void ValidatePacked(const Views& a) {
    const auto &q = *a[0], &k = *a[1], &v = *a[2], &g = *a[3], &beta = *a[4],
               &state = *a[5], &out = *a[9];
    assert(a[6] && a[7] && a[8] &&
           "packed recurrent delta requires offsets and slot indices");
    assert(q.ndim() == 4 && k.shape() == q.shape() && v.ndim() == 4 &&
           out.shape() == v.shape() && state.ndim() == 4 && g.ndim() == 3 &&
           beta.shape() == g.shape() &&
           "packed recurrent delta rank/shape mismatch");
    const auto T = q.shape()[1], Hk = q.shape()[2], Hv = v.shape()[2],
               D = q.shape()[3];
    if (D != 128)
      throw std::invalid_argument(
          "packed recurrent delta requires a head dim of 128");
    assert((q.shape()[0] == 1 && v.shape()[0] == 1 && v.shape()[1] == T && T &&
            Hk && Hv && Hv % Hk == 0 && D && v.shape()[3] == D &&
            g.shape() == Tensor::Shape{1, T, Hv} &&
            state.shape()[1] == Hv && state.shape()[2] == D &&
            state.shape()[3] == D) &&
           "packed recurrent delta requires [1,T,H,D] data, Dk == Dv and a "
           "[pool,Hv,D,D] state");
    assert(floating(q.dtype()) && state.dtype() == DataType::kFloat32 &&
           g.dtype() == DataType::kFloat32 && beta.dtype() == g.dtype() &&
           "packed recurrent delta keeps state and gates in F32");
    assert(a[6]->numel() >= 2 && a[7]->numel() + 1 == a[6]->numel() &&
           a[8]->shape() == a[7]->shape() &&
           "packed recurrent delta requires [N+1] offsets and [N] slot indices");
    for (size_t i = 0; i < a.size(); ++i) {
      const auto& t = *a[i];
      assert(t.device() == q.device() &&
             "packed recurrent delta device mismatch");
      for (auto stride : t.strides())
        assert(stride >= 0 &&
               "packed recurrent delta negative stride unsupported");
      if (i == 6 || i == 7 || i == 8) {
        assert((t.ndim() == 1 && t.IsContiguous() &&
                (t.dtype() == DataType::kInt32 ||
                 t.dtype() == DataType::kInt64)) &&
               "packed recurrent delta sequence indices require contiguous "
               "I32/I64 vectors");
      } else if (i != 3 && i != 4) {
        assert(t.dtype() == q.dtype() &&
               "packed recurrent delta data dtype mismatch");
      }
      assert((!(i <= 2 || i == 9) || t.strides()[3] == 1) &&
             "packed recurrent delta innermost data dimension must be "
             "contiguous");
      assert((!(i == 5 || i == 9) || !t.HasBroadcastDim()) &&
             "packed recurrent delta broadcast destination unsupported");
    }
    if (state.shape()[0] >
        static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error(
          "recurrent delta pool exceeds int64 index extent");
  }
  void CheckInvocation(const Views& views, bool norm) const {
    if (norm != norm_)
      throw std::invalid_argument(
          "recurrent delta normalization changed after Make");
    for (size_t i = 0; i < views.size(); ++i) {
      if (views[i].has_value() != metadata_[i].has_value() ||
          (views[i] && !std::equal_to<Tensor>{}(*views[i], *metadata_[i])))
        throw std::invalid_argument(
            "recurrent delta metadata changed after Make");
      if (views[i] && views[i]->numel() && !views[i]->data())
        throw std::invalid_argument(
            "recurrent delta nonempty tensor has null data");
    }
  }
};
}  // namespace infini::ops
