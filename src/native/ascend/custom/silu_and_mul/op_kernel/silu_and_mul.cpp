#include "kernel_operator.h"

class KernelSiluAndMul {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int64_t hidden,
                              int64_t input_stride, int64_t out_stride) {
    hidden_ = hidden;
    input_stride_ = input_stride;
    out_stride_ = out_stride;
    tile_ = hidden / 8;
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
    pipe_.InitBuffer(gate_queue_, 2, tile_ * sizeof(bfloat16_t));
    pipe_.InitBuffer(up_queue_, 2, tile_ * sizeof(bfloat16_t));
    pipe_.InitBuffer(out_queue_, 2, tile_ * sizeof(bfloat16_t));
    pipe_.InitBuffer(gate_float_, tile_ * sizeof(float));
    pipe_.InitBuffer(up_float_, tile_ * sizeof(float));
    pipe_.InitBuffer(out_float_, tile_ * sizeof(float));
  }

  __aicore__ inline void Process(int64_t rows) {
    int64_t tile_offset = (AscendC::GetBlockIdx() % 8) * tile_;
    for (int64_t row = AscendC::GetBlockIdx() / 8; row < rows;
         row += AscendC::GetBlockNum() / 8) {
      auto gate = gate_queue_.AllocTensor<bfloat16_t>();
      auto up = up_queue_.AllocTensor<bfloat16_t>();
      AscendC::DataCopy(gate, input_[row * input_stride_ + tile_offset], tile_);
      AscendC::DataCopy(up, input_[row * input_stride_ + hidden_ + tile_offset],
                        tile_);
      gate_queue_.EnQue(gate);
      up_queue_.EnQue(up);
      gate = gate_queue_.DeQue<bfloat16_t>();
      up = up_queue_.DeQue<bfloat16_t>();
      auto out = out_queue_.AllocTensor<bfloat16_t>();
      auto gate_float = gate_float_.Get<float>();
      auto up_float = up_float_.Get<float>();
      auto out_float = out_float_.Get<float>();
      AscendC::Cast(gate_float, gate, AscendC::RoundMode::CAST_NONE, tile_);
      AscendC::Cast(up_float, up, AscendC::RoundMode::CAST_NONE, tile_);
      // AscendC takes the multiplier before the SiLU gate.
      AscendC::SwiGLU<float, false>(out_float, up_float, gate_float, 1.0f,
                                    tile_);
      AscendC::Cast(out, out_float, AscendC::RoundMode::CAST_RINT, tile_);
      out_queue_.EnQue(out);
      gate_queue_.FreeTensor(gate);
      up_queue_.FreeTensor(up);
      out = out_queue_.DeQue<bfloat16_t>();
      AscendC::DataCopy(out_[row * out_stride_ + tile_offset], out, tile_);
      out_queue_.FreeTensor(out);
    }
  }

 private:
  AscendC::TPipe pipe_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 2> gate_queue_;

  AscendC::TQue<AscendC::QuePosition::VECIN, 2> up_queue_;

  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> out_queue_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> gate_float_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> up_float_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> out_float_;

  AscendC::GlobalTensor<bfloat16_t> input_;

  AscendC::GlobalTensor<bfloat16_t> out_;

  int64_t hidden_{0};

  int64_t tile_{0};

  int64_t input_stride_{0};

  int64_t out_stride_{0};
};

extern "C" __global__ __aicore__ void silu_and_mul_tiled(
    GM_ADDR input, GM_ADDR out, int64_t rows, int64_t hidden,
    int64_t input_stride, int64_t out_stride) {
  KernelSiluAndMul op;
  op.Init(input, out, hidden, input_stride, out_stride);
  op.Process(rows);
}
