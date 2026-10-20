#include <cstddef>

#include "kernel_operator.h"

// One token's contiguous 128-element heads are normalized together.
// Reduction and broadcast stay in UB; packed QKV token strides are preserved.
class KernelRmsNormQk {
 public:
  __aicore__ inline void Init(GM_ADDR y, GM_ADDR x, GM_ADDR weight,
                              size_t heads, ptrdiff_t y_token,
                              ptrdiff_t x_token, float epsilon) {
    heads_ = heads;
    y_token_ = y_token;
    x_token_ = x_token;
    epsilon_ = epsilon;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    weight_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(weight));
    const size_t elements = heads * 128;
    pipe_.InitBuffer(input_, 2, elements * sizeof(bfloat16_t));
    pipe_.InitBuffer(output_, 2, elements * sizeof(bfloat16_t));
    pipe_.InitBuffer(weight_input_, 1, 128 * sizeof(bfloat16_t));
    pipe_.InitBuffer(x_f_, elements * sizeof(float));
    pipe_.InitBuffer(square_f_, elements * sizeof(float));
    pipe_.InitBuffer(out_f_, elements * sizeof(float));
    pipe_.InitBuffer(weight_f_, 128 * sizeof(float));
    pipe_.InitBuffer(rstd_, 32 * sizeof(float));
    pipe_.InitBuffer(variance_, 32 * sizeof(float));
    pipe_.InitBuffer(ones_, 32 * sizeof(float));
    pipe_.InitBuffer(rstd_blocks_, 32 * 8 * sizeof(float));
  }

  __aicore__ inline void Process(size_t tokens) {
    AscendC::LocalTensor<bfloat16_t> weight =
        weight_input_.AllocTensor<bfloat16_t>();
    AscendC::DataCopy(weight, weight_, 128);
    weight_input_.EnQue(weight);
    weight = weight_input_.DeQue<bfloat16_t>();
    AscendC::LocalTensor<float> wf = weight_f_.Get<float>();
    AscendC::Cast(wf, weight, AscendC::RoundMode::CAST_NONE, 128);
    weight_input_.FreeTensor(weight);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<float> ones = ones_.Get<float>();
    AscendC::Duplicate(ones, 1.0f, 32);
    AscendC::PipeBarrier<PIPE_V>();
    for (size_t token = AscendC::GetBlockIdx(); token < tokens;
         token += AscendC::GetBlockNum()) {
      AscendC::LocalTensor<bfloat16_t> input = input_.AllocTensor<bfloat16_t>();
      AscendC::DataCopy(input, x_[token * x_token_], heads_ * 128);
      input_.EnQue(input);
      input = input_.DeQue<bfloat16_t>();
      AscendC::LocalTensor<bfloat16_t> output =
          output_.AllocTensor<bfloat16_t>();
      AscendC::LocalTensor<float> x = x_f_.Get<float>();
      AscendC::LocalTensor<float> squares = square_f_.Get<float>();
      AscendC::LocalTensor<float> out = out_f_.Get<float>();
      AscendC::LocalTensor<float> rstd = rstd_.Get<float>();
      AscendC::LocalTensor<float> variance = variance_.Get<float>();
      AscendC::LocalTensor<float> factors = rstd_blocks_.Get<float>();
      AscendC::Cast(x, input, AscendC::RoundMode::CAST_NONE, heads_ * 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(squares, x, x, heads_ * 128);
      AscendC::PipeBarrier<PIPE_V>();
      // Sum corresponding elements of the two halves, then reduce 64.
      AscendC::BinaryRepeatParams halves{1, 1, 1, 16, 16, 16};
      AscendC::Add(out, squares, squares[64], uint64_t(64),
                   static_cast<uint8_t>(heads_), halves);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum(rstd, out, 64, static_cast<int32_t>(heads_), 1, 1,
                              16);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(variance, rstd, 1.0f / 128.0f, heads_);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(variance, variance, epsilon_, heads_);
      AscendC::PipeBarrier<PIPE_V>();
      // Match the native sqrt/divide sequence before BF16 rounding.
      AscendC::Sqrt(rstd, variance, heads_);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(rstd, ones, rstd, heads_);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Brcb(factors, rstd, static_cast<uint8_t>((heads_ + 7) / 8),
                    {1, 8});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::BinaryRepeatParams normalize{1, 1, 0, 16, 16, 1};
      AscendC::Mul(out, x, factors, uint64_t(64), static_cast<uint8_t>(heads_),
                   normalize);
      AscendC::Mul(out[64], x[64], factors, uint64_t(64),
                   static_cast<uint8_t>(heads_), normalize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::BinaryRepeatParams broadcast_weight{1, 1, 1, 16, 16, 0};
      AscendC::Mul(out, out, wf, uint64_t(64), static_cast<uint8_t>(heads_),
                   broadcast_weight);
      AscendC::Mul(out[64], out[64], wf[64], uint64_t(64),
                   static_cast<uint8_t>(heads_), broadcast_weight);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(output, out, AscendC::RoundMode::CAST_RINT, heads_ * 128);
      output_.EnQue(output);
      input_.FreeTensor(input);
      output = output_.DeQue<bfloat16_t>();
      AscendC::DataCopy(y_[token * y_token_], output, heads_ * 128);
      output_.FreeTensor(output);
    }
  }

 private:
  AscendC::TPipe pipe_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 1> weight_input_;

  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> output_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> x_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> square_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> out_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> weight_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> rstd_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> variance_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> ones_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> rstd_blocks_;

  AscendC::GlobalTensor<bfloat16_t> x_;

  AscendC::GlobalTensor<bfloat16_t> y_;

  AscendC::GlobalTensor<bfloat16_t> weight_;

  size_t heads_;

  ptrdiff_t x_token_;

  ptrdiff_t y_token_;

  float epsilon_;
};

extern "C" __global__ __aicore__ void rms_norm_qk(
    GM_ADDR input, GM_ADDR weight, GM_ADDR out, int64_t tokens, int64_t heads,
    int64_t input_token_stride, int64_t out_token_stride, float epsilon) {
  KernelRmsNormQk op;
  op.Init(out, input, weight, heads, out_token_stride, input_token_stride,
          epsilon);
  op.Process(tokens);
}
