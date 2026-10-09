#include <cstddef>

#include "kernel_operator.h"

// A token's contiguous heads share one DMA transfer and one sin/cos load.
// FP32 vector repeats broadcast the half-head tables across all heads.
class KernelRotaryEmbeddingQk {
 public:
  __aicore__ inline void Init(GM_ADDR positions, GM_ADDR data, GM_ADDR cache,
                              size_t heads, ptrdiff_t token_stride,
                              ptrdiff_t cache_stride) {
    heads_ = heads;
    token_stride_ = token_stride;
    cache_stride_ = cache_stride;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(data));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(data));
    sin_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cache) + 64);
    cos_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cache));
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
    const size_t elements = heads_ * 128;
    pipe_.InitBuffer(input_, 2, elements * sizeof(bfloat16_t));
    pipe_.InitBuffer(output_, 2, elements * sizeof(bfloat16_t));
    pipe_.InitBuffer(sin_input_, 2, 64 * sizeof(bfloat16_t));
    pipe_.InitBuffer(cos_input_, 2, 64 * sizeof(bfloat16_t));
    pipe_.InitBuffer(input_f_, elements * sizeof(float));
    pipe_.InitBuffer(output_f_, elements * sizeof(float));
    pipe_.InitBuffer(sin_f_, 64 * sizeof(float));
    pipe_.InitBuffer(cos_f_, 64 * sizeof(float));
    pipe_.InitBuffer(first_, heads_ * 64 * sizeof(float));
    pipe_.InitBuffer(second_, heads_ * 64 * sizeof(float));
    pipe_.InitBuffer(product1_, heads_ * 64 * sizeof(float));
    pipe_.InitBuffer(product2_, heads_ * 64 * sizeof(float));
    pipe_.InitBuffer(result1_, heads_ * 64 * sizeof(float));
    pipe_.InitBuffer(result2_, heads_ * 64 * sizeof(float));
  }

  __aicore__ inline void Process(size_t tokens) {
    for (size_t work = AscendC::GetBlockIdx(); work < tokens;
         work += AscendC::GetBlockNum()) {
      const size_t token = work;
      const size_t position = static_cast<size_t>(positions_.GetValue(token));
      AscendC::LocalTensor<bfloat16_t> input = input_.AllocTensor<bfloat16_t>();
      AscendC::LocalTensor<bfloat16_t> sin =
          sin_input_.AllocTensor<bfloat16_t>();
      AscendC::LocalTensor<bfloat16_t> cos =
          cos_input_.AllocTensor<bfloat16_t>();
      AscendC::DataCopy(input, x_[token * token_stride_], heads_ * 128);
      AscendC::DataCopy(sin, sin_[position * cache_stride_], 64);
      AscendC::DataCopy(cos, cos_[position * cache_stride_], 64);
      input_.EnQue(input);
      sin_input_.EnQue(sin);
      cos_input_.EnQue(cos);
      input = input_.DeQue<bfloat16_t>();
      sin = sin_input_.DeQue<bfloat16_t>();
      cos = cos_input_.DeQue<bfloat16_t>();
      AscendC::LocalTensor<bfloat16_t> output =
          output_.AllocTensor<bfloat16_t>();
      AscendC::LocalTensor<float> x = input_f_.Get<float>();
      AscendC::LocalTensor<float> out = output_f_.Get<float>();
      AscendC::LocalTensor<float> sf = sin_f_.Get<float>();
      AscendC::LocalTensor<float> cf = cos_f_.Get<float>();
      AscendC::LocalTensor<float> first = first_.Get<float>();
      AscendC::LocalTensor<float> second = second_.Get<float>();
      AscendC::LocalTensor<float> p1 = product1_.Get<float>();
      AscendC::LocalTensor<float> p2 = product2_.Get<float>();
      AscendC::LocalTensor<float> r1 = result1_.Get<float>();
      AscendC::LocalTensor<float> r2 = result2_.Get<float>();
      AscendC::Cast(x, input, AscendC::RoundMode::CAST_NONE, heads_ * 128);
      AscendC::Cast(sf, sin, AscendC::RoundMode::CAST_NONE, 64);
      AscendC::Cast(cf, cos, AscendC::RoundMode::CAST_NONE, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::DataCopyParams split{static_cast<uint16_t>(heads_), 8, 8, 0};
      AscendC::DataCopy(first, x, split);
      AscendC::DataCopy(second, x[64], split);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::BinaryRepeatParams broadcast{1, 1, 1, 8, 8, 0};
      AscendC::Mul(p1, first, cf, uint64_t(64), static_cast<uint8_t>(heads_),
                   broadcast);
      AscendC::Mul(p2, second, sf, uint64_t(64), static_cast<uint8_t>(heads_),
                   broadcast);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(r1, p1, p2, heads_ * 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(p1, first, sf, uint64_t(64), static_cast<uint8_t>(heads_),
                   broadcast);
      AscendC::Mul(p2, second, cf, uint64_t(64), static_cast<uint8_t>(heads_),
                   broadcast);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(r2, p1, p2, heads_ * 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::DataCopyParams merge{static_cast<uint16_t>(heads_), 8, 0, 8};
      AscendC::DataCopy(out, r1, merge);
      AscendC::DataCopy(out[64], r2, merge);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(output, out, AscendC::RoundMode::CAST_RINT, heads_ * 128);
      output_.EnQue(output);
      input_.FreeTensor(input);
      sin_input_.FreeTensor(sin);
      cos_input_.FreeTensor(cos);
      output = output_.DeQue<bfloat16_t>();
      AscendC::DataCopy(y_[token * token_stride_], output, heads_ * 128);
      output_.FreeTensor(output);
    }
  }

 private:
  AscendC::TPipe pipe_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 2> sin_input_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 2> cos_input_;

  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> output_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> input_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> output_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> sin_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> cos_f_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> first_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> second_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> product1_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> product2_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> result1_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> result2_;

  AscendC::GlobalTensor<bfloat16_t> x_;

  AscendC::GlobalTensor<bfloat16_t> y_;

  AscendC::GlobalTensor<bfloat16_t> sin_;

  AscendC::GlobalTensor<bfloat16_t> cos_;

  AscendC::GlobalTensor<int64_t> positions_;

  size_t heads_{0};

  ptrdiff_t token_stride_{0};

  ptrdiff_t cache_stride_{0};
};

extern "C" __global__ __aicore__ void rotary_embedding_qk(
    GM_ADDR positions, GM_ADDR data, GM_ADDR cos_sin_cache, int64_t tokens,
    int64_t heads, int64_t token_stride, int64_t cache_stride) {
  KernelRotaryEmbeddingQk op;
  op.Init(positions, data, cos_sin_cache, heads, token_stride, cache_stride);
  op.Process(tokens);
}
