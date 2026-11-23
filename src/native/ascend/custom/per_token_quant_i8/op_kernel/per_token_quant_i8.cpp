#include "kernel_operator.h"

template <typename T>
class KernelPerTokenQuantI8 {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR packed, GM_ADDR scale,
                              int64_t rows, int64_t cols) {
    rows_ = rows;
    cols_ = cols;

    const int64_t block_idx = AscendC::GetBlockIdx();
    const int64_t block_count = AscendC::GetBlockNum();
    const int64_t rows_per_block = (rows + block_count - 1) / block_count;
    rows_per_block_ = ((rows_per_block + 7) / 8) * 8;
    row_begin_ = block_idx * rows_per_block_;
    row_end_ = row_begin_ + rows_per_block_;
    if (row_end_ > rows) row_end_ = rows;

    x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), rows * cols);
    packed_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(packed),
                               rows * cols);
    scale_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), rows);

    const int32_t col_count = static_cast<int32_t>(cols);
    pipe_.InitBuffer(input_queue_, 1, col_count * sizeof(T));
    pipe_.InitBuffer(output_queue_, 1, col_count * sizeof(int8_t));
    pipe_.InitBuffer(scale_queue_, 1, rows_per_block_ * sizeof(float));
    pipe_.InitBuffer(fp32_buffer_, col_count * sizeof(float));
    pipe_.InitBuffer(abs_buffer_, col_count * sizeof(float));
    pipe_.InitBuffer(fp16_buffer_, col_count * sizeof(half));

    constexpr int32_t elems_per_repeat = 256 / sizeof(float);
    constexpr int32_t elems_per_block = 32 / sizeof(float);
    const int32_t first_repeat =
        (col_count + elems_per_repeat - 1) / elems_per_repeat;
    const int32_t reduce_tmp_elems =
        ((first_repeat + elems_per_block - 1) / elems_per_block) *
        elems_per_block;
    pipe_.InitBuffer(reduce_tmp_buffer_, reduce_tmp_elems * sizeof(float));
    pipe_.InitBuffer(max_buffer_, 32);
  }

  __aicore__ inline void Process() {
    if (row_begin_ >= row_end_) return;

    auto scale_output = scale_queue_.AllocTensor<float>();
    for (int64_t row = row_begin_; row < row_end_; ++row) {
      CopyIn(row);
      Compute(row, scale_output);
      CopyOut(row);
    }
    scale_queue_.EnQue(scale_output);
    CopyScaleOut();
  }

 private:
  __aicore__ inline void CopyIn(int64_t row) {
    auto input = input_queue_.AllocTensor<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(input, x_gm_[row * cols_], params, pad);
    input_queue_.EnQue(input);
  }

  __aicore__ inline void Compute(
      int64_t row, AscendC::LocalTensor<float>& scale_output) {
    auto input = input_queue_.DeQue<T>();
    auto output = output_queue_.AllocTensor<int8_t>();
    auto fp32 = fp32_buffer_.Get<float>();
    auto abs_value = abs_buffer_.Get<float>();
    auto fp16 = fp16_buffer_.Get<half>();
    auto reduce_tmp = reduce_tmp_buffer_.Get<float>();
    auto max_value = max_buffer_.Get<float>();
    const int32_t count = static_cast<int32_t>(cols_);

    AscendC::Cast(fp32, input, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs(abs_value, fp32, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(max_value, abs_value, reduce_tmp, count);

    const float abs_max = max_value.GetValue(0);
    const float scale = abs_max > 0.0f ? abs_max / 127.0f : 1.0f;
    const float multiplier = abs_max > 0.0f ? 127.0f / abs_max : 0.0f;
    AscendC::Muls(fp32, fp32, multiplier, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(fp16, fp32, AscendC::RoundMode::CAST_ROUND, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(output, fp16, AscendC::RoundMode::CAST_RINT, count);
    scale_output.SetValue(row - row_begin_, scale);

    input_queue_.FreeTensor(input);
    output_queue_.EnQue(output);
  }

  __aicore__ inline void CopyOut(int64_t row) {
    auto output = output_queue_.DeQue<int8_t>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(cols_ * sizeof(int8_t)), 0, 0, 0};
    AscendC::DataCopyPad(packed_gm_[row * cols_], output, params);
    output_queue_.FreeTensor(output);
  }

  __aicore__ inline void CopyScaleOut() {
    auto scale_output = scale_queue_.DeQue<float>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>((row_end_ - row_begin_) * sizeof(float)),
        0, 0, 0};
    AscendC::DataCopyPad(scale_gm_[row_begin_], scale_output, params);
    scale_queue_.FreeTensor(scale_output);
  }

  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> input_queue_;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> output_queue_;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> scale_queue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> abs_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_tmp_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> max_buffer_;
  AscendC::GlobalTensor<T> x_gm_;
  AscendC::GlobalTensor<int8_t> packed_gm_;
  AscendC::GlobalTensor<float> scale_gm_;
  int64_t rows_{0};
  int64_t cols_{0};
  int64_t rows_per_block_{0};
  int64_t row_begin_{0};
  int64_t row_end_{0};
};

extern "C" __global__ __aicore__ void per_token_quant_i8(
    GM_ADDR x, GM_ADDR packed, GM_ADDR scale, int64_t rows, int64_t cols,
    int64_t dtype_code) {
  if (dtype_code == 0) {
    KernelPerTokenQuantI8<half> op;
    op.Init(x, packed, scale, rows, cols);
    op.Process();
  } else {
    KernelPerTokenQuantI8<bfloat16_t> op;
    op.Init(x, packed, scale, rows, cols);
    op.Process();
  }
}
