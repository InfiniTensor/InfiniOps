#include "native/ascend/custom/fused_infer_attention_score/attn_infra/base_defs.hpp"
#include "native/ascend/custom/fused_infer_attention_score/attn_infra/detail/alignment.hpp"
namespace NpuArch {
template <typename T, typename U>
__aicore__ inline auto CeilDiv(T dividend, U divisor) {
  return Detail::Alignment::CeilDiv(dividend, divisor);
}
template <typename T, typename U>
__aicore__ inline auto RoundDown(T value, U align) {
  return Detail::Alignment::RoundDown(value, align);
}
template <typename T, typename U>
__aicore__ inline auto RoundUp(T value, U align) {
  return Detail::Alignment::RoundUp(value, align);
}
struct splitNode {
  uint32_t batchIdx[64];
  uint32_t headStartIdx[64];
  uint32_t headEndIdx[64];
  uint32_t qStartIdx[64];
  uint32_t qEndIdx[64];
  uint32_t lseTaskOffset[64];
  uint32_t oTaskOffset[64];
  uint32_t splitNum[64];
};
}  // namespace NpuArch
#include "native/ascend/custom/fused_infer_attention_score/kernel_common.hpp"

struct FAInferTilingData {
  uint64_t mm1OutSize;
  uint64_t smOnlineOutSize;
  uint64_t mm2OutSize;
  uint32_t batch;
  uint32_t numHeads;
  uint32_t kvHeads;
  uint32_t embeddingSize;
  uint32_t embeddingSizeV;
  uint32_t blockSize;
  uint32_t maxNumBlocksPerBatch;
  uint32_t firstBatchTaskNum;
  uint32_t totalTaskNum;
  uint32_t mainLoopTaskNum;
  uint32_t tailLoopTaskNum;
  uint32_t tailStartBatch;
  uint32_t tailStartN2;
  uint32_t tailKvNBlockTile;
  uint32_t maskType;
  float scaleValue;
  uint32_t sparseMode;
  int64_t preToken;
  int64_t nextToken;
  uint64_t qStride;
  uint64_t oStride;
};

#include "native/ascend/custom/fused_infer_attention_score/flash_attention_regular_decode.h"

extern "C" __global__ __aicore__ void infinilm_paged_attention_fai_bf16_i32(
    GM_ADDR out, GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache,
    GM_ADDR block_tables, GM_ADDR cache_lens, float scale, size_t num_heads,
    size_t num_seqs, size_t num_kv_heads, size_t head_size,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t q_stride,
    ptrdiff_t o_stride, GM_ADDR workspace, GM_ADDR tiling) {
  using namespace NpuArch;
  using ArchTag = Arch::AtlasA2;
  using ElementQ = bfloat16_t;
  using LayoutQ = layout::RowMajor;
  using ElementK = bfloat16_t;
  using LayoutK = layout::ColumnMajor;
  using ElementV = bfloat16_t;
  using LayoutV = layout::RowMajor;
  using ElementS = float;
  using LayoutS = layout::RowMajor;
  using ElementSink = bfloat16_t;
  using LayoutSink = layout::RowMajor;
  using ElementP = bfloat16_t;
  using LayoutP = layout::RowMajor;
  using ElementO = bfloat16_t;
  using LayoutO = layout::RowMajor;
  using ElementLse = float;
  using LayoutLse = layout::RowMajor;
  using ElementMask = int8_t;
  using LayoutMask = layout::RowMajor;
  using ElementOTmp = float;
  using LayoutOTmp = layout::RowMajor;
  using ElementUpdate = float;
  using LayoutUpdate = layout::RowMajor;
  using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
  using L0TileShapeQK = GemmShape<128, 128, 128>;
  using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKDecode<true, false>;
  using QType = Gemm::GemmType<ElementQ, LayoutQ>;
  using KType = Gemm::GemmType<ElementK, LayoutK>;
  using SType = Gemm::GemmType<ElementS, LayoutS>;
  using SinkType = Gemm::GemmType<ElementSink, LayoutSink>;
  using BlockMmadQK =
      Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                             QType, KType, SType>;
  using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<
      Epilogue::LseMode::NONE, Epilogue::SinkMode::DISABLE,
      Epilogue::MaskMode::NO_MASK, float>;
  using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
  using pseShiftType = Gemm::GemmType<ElementQ, LayoutQ>;
  using PType = Gemm::GemmType<ElementP, LayoutP>;
  using EpilogueOnlineSoftmax =
      Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType,
                                     maskType, SinkType, pseShiftType>;
  using L1TileShapePV = GemmShape<128, 128, 256>;
  using L0TileShapePV = GemmShape<128, 128, 128>;
  using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVDecode<true, false>;
  using VType = Gemm::GemmType<ElementV, LayoutV>;
  using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
  using BlockMmadPV =
      Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                             PType, VType, OTmpType>;
  using DispatchPolicyRescaleO =
      Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, float>;
  using OType = Gemm::GemmType<ElementO, LayoutO>;
  using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
  using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
  using EpilogueRescaleO =
      Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType,
                                     OUpdateType, LseType>;
  using DispatchPolicyInitOutWhenZero =
      Epilogue::EpilogueAtlasA2InitOutWhenZero<Epilogue::LseMode::NONE>;
  using EpilogueInitOut =
      Epilogue::Block::BlockEpilogue<DispatchPolicyInitOutWhenZero, OType,
                                     LseType>;
  using SplitFuse::FAInferKernelDecoding;
  using FAI =
      FAInferKernelDecoding<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax,
                            EpilogueRescaleO, EpilogueInitOut, true,
                            FaiKernel::MaskType::NO_MASK,
                            FaiKernel::inputLayout::TND>;
  FAIKernelParams params{
      q,          k_cache, v_cache, nullptr,   nullptr, block_tables, nullptr,
      cache_lens, out,     nullptr, workspace, tiling,  nullptr};
  FAI kernel;
  kernel(params);
}
