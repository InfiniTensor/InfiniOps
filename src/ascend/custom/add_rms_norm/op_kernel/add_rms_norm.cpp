#include "data_type.h"
#include "kernel_operator.h"

constexpr int32_t kBufferNum = 2;

template <typename T>
class KernelAddRmsNorm {
 public:
  __aicore__ inline KernelAddRmsNorm() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR residual, GM_ADDR weight,
                              int64_t total_rows, int64_t dim_length,
                              int64_t dim_length_align, int64_t former_num,
                              int64_t former_length, int64_t tail_length,
                              float eps, GM_ADDR out, GM_ADDR residual_out) {
    dim_length_ = dim_length;
    dim_length_align_ = dim_length_align;
    eps_ = eps;

    // Block-level tiling: determine row range for this core.
    int64_t block_idx = AscendC::GetBlockIdx();
    int64_t row_offset;

    if (block_idx < former_num) {
      block_rows_ = former_length;
      row_offset = former_length * block_idx;
    } else {
      block_rows_ = tail_length;
      int64_t tail_idx = block_idx - former_num;
      row_offset = former_length * former_num + tail_length * tail_idx;
    }

    // Global memory pointers.
    input_gm_.SetGlobalBuffer((__gm__ T*)input + row_offset * dim_length_align,
                              block_rows_ * dim_length_align);
    residual_gm_.SetGlobalBuffer(
        (__gm__ T*)residual + row_offset * dim_length_align,
        block_rows_ * dim_length_align);
    out_gm_.SetGlobalBuffer((__gm__ T*)out + row_offset * dim_length_align,
                            block_rows_ * dim_length_align);
    residual_out_gm_.SetGlobalBuffer(
        (__gm__ T*)residual_out + row_offset * dim_length_align,
        block_rows_ * dim_length_align);
    weight_gm_.SetGlobalBuffer((__gm__ float*)weight, dim_length_align);

    int32_t dim_len_align = static_cast<int32_t>(dim_length_align_);

    // I/O queues (double-buffered).
    pipe_.InitBuffer(in_queue_input_, kBufferNum,
                     dim_len_align * static_cast<int32_t>(sizeof(T)));
    pipe_.InitBuffer(in_queue_residual_, kBufferNum,
                     dim_len_align * static_cast<int32_t>(sizeof(T)));
    pipe_.InitBuffer(out_queue_out_, kBufferNum,
                     dim_len_align * static_cast<int32_t>(sizeof(T)));
    pipe_.InitBuffer(out_queue_residual_out_, kBufferNum,
                     dim_len_align * static_cast<int32_t>(sizeof(T)));

    // Weight buffer (fp32, loaded once, reused for all rows).
    pipe_.InitBuffer(weight_buf_,
                     dim_len_align * static_cast<int32_t>(sizeof(float)));

    // FP16/BF16 path needs extra fp32 compute buffers.
    // `fp32_buf1_`: holds `x_out` in fp32 (reused from `x1_fp32` after Add).
    // `fp32_buf2_`: holds `x2_fp32` initially, then `x_out^2`, then final
    // result.
    if constexpr (sizeof(T) == 2) {
      pipe_.InitBuffer(fp32_buf1_,
                       dim_len_align * static_cast<int32_t>(sizeof(float)));
      pipe_.InitBuffer(fp32_buf2_,
                       dim_len_align * static_cast<int32_t>(sizeof(float)));
    }

    // `ReduceSum` temporary buffer (size per API formula).
    constexpr int32_t kElemsPerRepeat = 256 / sizeof(float);
    constexpr int32_t kElemsPerBlock = 32 / sizeof(float);
    int32_t first_max_repeat =
        (dim_len_align + kElemsPerRepeat - 1) / kElemsPerRepeat;
    int32_t reduce_tmp_size =
        ((first_max_repeat + kElemsPerBlock - 1) / kElemsPerBlock) *
        kElemsPerBlock;
    pipe_.InitBuffer(reduce_tmp_buf_,
                     reduce_tmp_size * static_cast<int32_t>(sizeof(float)));

    // Scalar buffer for reduction result (8 floats = 32 bytes).
    pipe_.InitBuffer(sum_buf_, 32);

    // Load weight (fp32) from GM into `weight_buf_`.
    AscendC::LocalTensor<float> w_local = weight_buf_.Get<float>();
    AscendC::DataCopyExtParams w_params{
        1, static_cast<uint32_t>(dim_len_align * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> w_pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(w_local, weight_gm_, w_params, w_pad);

    // Ensure weight DMA completes before compute.
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Process() {
    for (int64_t row = 0; row < block_rows_; ++row) {
      CopyIn(row);
      Compute(row);
      CopyOut(row);
    }
  }

 private:
  __aicore__ inline void CopyIn(int64_t row) {
    AscendC::LocalTensor<T> input_local = in_queue_input_.AllocTensor<T>();
    AscendC::LocalTensor<T> residual_local =
        in_queue_residual_.AllocTensor<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(dim_length_align_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(input_local, input_gm_[row * dim_length_align_],
                         params, pad);
    AscendC::DataCopyPad(residual_local, residual_gm_[row * dim_length_align_],
                         params, pad);
    in_queue_input_.EnQue(input_local);
    in_queue_residual_.EnQue(residual_local);
  }

  __aicore__ inline void Compute(int64_t row) {
    AscendC::LocalTensor<T> input_local = in_queue_input_.DeQue<T>();
    AscendC::LocalTensor<T> residual_local = in_queue_residual_.DeQue<T>();
    AscendC::LocalTensor<T> out_local = out_queue_out_.AllocTensor<T>();
    AscendC::LocalTensor<T> residual_out_local =
        out_queue_residual_out_.AllocTensor<T>();

    AscendC::LocalTensor<float> w_local = weight_buf_.Get<float>();
    AscendC::LocalTensor<float> r_tmp = reduce_tmp_buf_.Get<float>();
    AscendC::LocalTensor<float> s_local = sum_buf_.Get<float>();

    int32_t dim_len = static_cast<int32_t>(dim_length_);
    int32_t dim_len_align = static_cast<int32_t>(dim_length_align_);

    if constexpr (sizeof(T) == 4) {
      // ---- FP32 path: compute directly. ----

      // Step 1: x_out = x1 + x2.
      AscendC::Add(residual_out_local, input_local, residual_local,
                   dim_len_align);

      // Step 2: x_out^2 into out_local (reuse output buffer temporarily).
      AscendC::Mul(out_local, residual_out_local, residual_out_local,
                   dim_len_align);

      // Step 3: ReduceSum(x_out^2) -> s_local[0].
      // `ReduceSum` may modify `out_local`, but we overwrite it below.
      AscendC::ReduceSum(s_local, out_local, r_tmp, dim_len_align);

      // Step 4-5: scale = 1 / sqrt(mean(x_out^2) + eps).
      float sum_val = s_local.GetValue(0);
      float mean_val = sum_val / static_cast<float>(dim_len) + eps_;
      s_local.SetValue(0, mean_val);
      AscendC::Sqrt(s_local, s_local, 8);
      float scale = 1.0f / s_local.GetValue(0);

      // Step 6: y = x_out * scale.
      AscendC::Muls(out_local, residual_out_local, scale, dim_len_align);

      // Step 7: y = y * weight.
      AscendC::Mul(out_local, out_local, w_local, dim_len_align);

    } else {
      // ---- FP16/BF16 path: cast → fp32 compute → cast back. ----
      AscendC::LocalTensor<float> b1 = fp32_buf1_.Get<float>();
      AscendC::LocalTensor<float> b2 = fp32_buf2_.Get<float>();

      // Cast inputs fp16/bf16 → fp32.
      AscendC::Cast(b1, input_local, AscendC::RoundMode::CAST_NONE,
                    dim_len_align);
      AscendC::Cast(b2, residual_local, AscendC::RoundMode::CAST_NONE,
                    dim_len_align);

      // Step 1: x_out = x1 + x2 (fp32), stored in b1.
      AscendC::Add(b1, b1, b2, dim_len_align);

      // Cast `x_out` fp32 → fp16/bf16 for the residual output.
      AscendC::Cast(residual_out_local, b1, AscendC::RoundMode::CAST_RINT,
                    dim_len_align);

      // Step 2: x_out^2 in fp32, stored in b2.
      AscendC::Mul(b2, b1, b1, dim_len_align);

      // Step 3: ReduceSum(x_out^2) -> s_local[0].
      AscendC::ReduceSum(s_local, b2, r_tmp, dim_len_align);

      // Step 4-5: scale = 1 / sqrt(mean(x_out^2) + eps).
      float sum_val = s_local.GetValue(0);
      float mean_val = sum_val / static_cast<float>(dim_len) + eps_;
      s_local.SetValue(0, mean_val);
      AscendC::Sqrt(s_local, s_local, 8);
      float scale = 1.0f / s_local.GetValue(0);

      // Step 6: y = x_out * scale (fp32), reuse b2.
      AscendC::Muls(b2, b1, scale, dim_len_align);

      // Step 7: y = y * weight (fp32).
      AscendC::Mul(b2, b2, w_local, dim_len_align);

      AscendC::Cast(out_local, b2, AscendC::RoundMode::CAST_RINT,
                    dim_len_align);
    }

    in_queue_input_.FreeTensor(input_local);
    in_queue_residual_.FreeTensor(residual_local);
    out_queue_out_.EnQue(out_local);
    out_queue_residual_out_.EnQue(residual_out_local);
  }

  __aicore__ inline void CopyOut(int64_t row) {
    AscendC::LocalTensor<T> out_local = out_queue_out_.DeQue<T>();
    AscendC::LocalTensor<T> residual_out_local =
        out_queue_residual_out_.DeQue<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(dim_length_align_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(out_gm_[row * dim_length_align_], out_local, params);
    AscendC::DataCopyPad(residual_out_gm_[row * dim_length_align_],
                         residual_out_local, params);
    out_queue_out_.FreeTensor(out_local);
    out_queue_residual_out_.FreeTensor(residual_out_local);
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> in_queue_input_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> in_queue_residual_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> out_queue_out_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> out_queue_residual_out_;

  AscendC::TBuf<AscendC::TPosition::VECCALC> weight_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buf1_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buf2_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_tmp_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buf_;

  AscendC::GlobalTensor<T> input_gm_, residual_gm_, out_gm_, residual_out_gm_;
  AscendC::GlobalTensor<float> weight_gm_;

  int64_t block_rows_;
  int64_t dim_length_;
  int64_t dim_length_align_;
  float eps_;
};

// `dtype_code` is `static_cast<int64_t>(infini::ops::DataType)` forwarded
// by the host launcher.  fp16 and bf16 both have `sizeof == 2` but need
// distinct numeric paths, so dispatch is on the `DataType` tag rather
// than the byte size.
//
// Parameters follow the C2 convention: inputs first, attributes between,
// outputs last.  The kernel symbol is prefixed with `aclrtlaunch_` by the
// `AscendC` toolchain, yielding `aclrtlaunch_AddRmsNorm` which matches the
// base `AddRmsNorm` class name.
extern "C" __global__ __aicore__ void AddRmsNorm(
    GM_ADDR input, GM_ADDR residual, GM_ADDR weight, int64_t total_rows,
    int64_t dim_length, int64_t dim_length_align, int64_t former_num,
    int64_t former_length, int64_t tail_length, float eps, int64_t dtype_code,
    GM_ADDR out, GM_ADDR residual_out) {
  switch (static_cast<infini::ops::DataType>(dtype_code)) {
    case infini::ops::DataType::kFloat16: {
      KernelAddRmsNorm<half> op;
      op.Init(input, residual, weight, total_rows, dim_length, dim_length_align,
              former_num, former_length, tail_length, eps, out, residual_out);
      op.Process();
      break;
    }
    case infini::ops::DataType::kBFloat16: {
      KernelAddRmsNorm<bfloat16_t> op;
      op.Init(input, residual, weight, total_rows, dim_length, dim_length_align,
              former_num, former_length, tail_length, eps, out, residual_out);
      op.Process();
      break;
    }
    case infini::ops::DataType::kFloat32:
    default: {
      KernelAddRmsNorm<float> op;
      op.Init(input, residual, weight, total_rows, dim_length, dim_length_align,
              former_num, former_length, tail_length, eps, out, residual_out);
      op.Process();
      break;
    }
  }
}
