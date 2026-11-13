#include "kernel_operator.h"

constexpr size_t PAGED_ATTENTION_TILE = 128;
constexpr size_t PAGED_ATTENTION_COPY_TILE = 32;
constexpr size_t BYTE_ALIGN = 32;

template <typename T>
__aicore__ inline size_t alignTileLen(size_t tile_len, size_t byte_align) {
  size_t bytes = tile_len * sizeof(T);
  size_t aligned_bytes = (bytes % byte_align == 0)
                             ? bytes
                             : (bytes + byte_align - bytes % byte_align);
  return aligned_bytes / sizeof(T);
}

#include <type_traits>

using namespace AscendC;

template <typename Tindex>
__aicore__ inline int64_t loadIndex(GlobalTensor<Tindex>& tensor,
                                    ptrdiff_t offset) {
  return static_cast<int64_t>(tensor.GetValue(offset));
}

template <typename Tdata, typename Tindex>
class PagedAttentionKernel {
 public:
  __aicore__ inline PagedAttentionKernel() {}

  __aicore__ inline void init(
      GM_ADDR out, GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache,
      GM_ADDR block_tables, GM_ADDR cache_lens, GM_ADDR alibi_slopes,
      bool has_alibi, size_t num_heads, size_t num_seqs, size_t num_kv_heads,
      size_t head_size, float scale, size_t max_num_blocks_per_seq,
      size_t page_block_size, ptrdiff_t q_stride, ptrdiff_t k_batch_stride,
      ptrdiff_t k_row_stride, ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride,
      ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
      ptrdiff_t block_table_batch_stride, ptrdiff_t cache_lens_stride) {
    _num_heads = num_heads;
    _num_seqs = num_seqs;
    _num_kv_heads = num_kv_heads;
    _head_size = head_size;
    _scale = scale;
    _max_num_blocks_per_seq = max_num_blocks_per_seq;
    _page_block_size = page_block_size;
    _q_stride = q_stride;
    _k_batch_stride = k_batch_stride;
    _k_row_stride = k_row_stride;
    _k_head_stride = k_head_stride;
    _v_batch_stride = v_batch_stride;
    _v_row_stride = v_row_stride;
    _v_head_stride = v_head_stride;
    _o_stride = o_stride;
    _block_table_batch_stride = block_table_batch_stride;
    _cache_lens_stride = cache_lens_stride;
    _has_alibi = has_alibi;

    _out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata*>(out));
    _q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata*>(q));
    _k_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata*>(k_cache));
    _v_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata*>(v_cache));
    _block_tables_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ Tindex*>(block_tables));
    _cache_lens_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ Tindex*>(cache_lens));
    if (_has_alibi) {
      _alibi_slopes_gm.SetGlobalBuffer(
          reinterpret_cast<__gm__ float*>(alibi_slopes));
    }

    _pipe.InitBuffer(
        _q_buf, alignTileLen<float>(_head_size, BYTE_ALIGN) * sizeof(float));
    _pipe.InitBuffer(
        _acc_buf, alignTileLen<float>(_head_size, BYTE_ALIGN) * sizeof(float));
    _pipe.InitBuffer(_score_buf, PAGED_ATTENTION_TILE * sizeof(float));
    _pipe.InitBuffer(
        _out_buf, alignTileLen<Tdata>(_head_size, BYTE_ALIGN) * sizeof(Tdata));

    // Shared float scratchpad for K/V type-casting (used by computeScore and V
    // accumulation)
    _pipe.InitBuffer(_v_f32_buf, alignTileLen<float>(_head_size, BYTE_ALIGN) *
                                     sizeof(float));
    _pipe.InitBuffer(_reduce_buf, alignTileLen<float>(_head_size, BYTE_ALIGN) *
                                      sizeof(float));
    // Double-buffered VECIN queue for K prefetch
    _head_stride = alignTileLen<Tdata>(_head_size, BYTE_ALIGN);
    _pipe.InitBuffer(_k_queue, 2,
                     PAGED_ATTENTION_COPY_TILE * _head_stride * sizeof(Tdata));
    // Double-buffered VECIN queue for V prefetch (overlaps with Score VEC
    // computation)
    _pipe.InitBuffer(_v_queue, 2,
                     PAGED_ATTENTION_COPY_TILE * _head_stride * sizeof(Tdata));
  }

  __aicore__ inline void process() {
    const size_t work_idx = GetBlockIdx();
    const size_t head_idx = work_idx % _num_heads;
    const size_t seq_idx = work_idx / _num_heads;
    if (seq_idx >= _num_seqs) {
      return;
    }

    const int64_t seq_len_i64 = loadIndex(
        _cache_lens_gm, static_cast<ptrdiff_t>(seq_idx) * _cache_lens_stride);
    if (seq_len_i64 <= 0) {
      return;
    }
    const size_t seq_len = static_cast<size_t>(seq_len_i64);
    const size_t num_queries_per_kv = _num_heads / _num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const float alibi_slope =
        _has_alibi ? _alibi_slopes_gm.GetValue(head_idx) : 0.0f;

    LocalTensor<float> q_local = _q_buf.Get<float>();
    LocalTensor<float> acc_local = _acc_buf.Get<float>();
    LocalTensor<float> score_local = _score_buf.Get<float>();

    // Load Q from global memory
    const ptrdiff_t q_base = static_cast<ptrdiff_t>(seq_idx) * _q_stride +
                             static_cast<ptrdiff_t>(head_idx * _head_size);

    LocalTensor<Tdata> q_raw_local = _out_buf.Get<Tdata>();
    DataCopy(q_raw_local, _q_gm[q_base], _head_size);
    PipeBarrier<PIPE_ALL>();

    Cast(q_local, q_raw_local, AscendC::RoundMode::CAST_NONE, _head_size);
    Duplicate(acc_local, 0.0f, _head_size);

    // Online softmax with burst K/V copies.
    float max_score = -3.4028234663852886e38f;
    float sum_exp = 0.0f;
    const uint16_t copy_len =
        static_cast<uint16_t>(_head_stride * sizeof(Tdata) / ONE_BLK_SIZE);
    const auto gap32 = [](ptrdiff_t row, size_t head) {
      const ptrdiff_t value = row - static_cast<ptrdiff_t>(head);
      if (value <= 0) return static_cast<uint16_t>(0);
      return static_cast<uint16_t>(value * sizeof(Tdata) / ONE_BLK_SIZE - 1);
    };
    const uint16_t k_gap = gap32(_k_row_stride, _head_stride);
    const uint16_t v_gap = gap32(_v_row_stride, _head_stride);
    for (size_t block_idx = 0; block_idx < _max_num_blocks_per_seq;
         ++block_idx) {
      const size_t token_start = block_idx * _page_block_size;
      if (token_start >= seq_len) break;
      const int64_t block_id =
          loadIndex(_block_tables_gm, static_cast<ptrdiff_t>(seq_idx) *
                                              _block_table_batch_stride +
                                          static_cast<ptrdiff_t>(block_idx));
      size_t token_end = token_start + _page_block_size;
      if (token_end > seq_len) token_end = seq_len;
      const size_t tile_count = token_end - token_start;
      const ptrdiff_t k_base =
          static_cast<ptrdiff_t>(block_id) * _k_batch_stride +
          static_cast<ptrdiff_t>(kv_head_idx) * _k_head_stride;
      const ptrdiff_t v_base =
          static_cast<ptrdiff_t>(block_id) * _v_batch_stride +
          static_cast<ptrdiff_t>(kv_head_idx) * _v_head_stride;
      size_t offset = 0;
      while (offset < tile_count) {
        const size_t count = (tile_count - offset < PAGED_ATTENTION_COPY_TILE
                                  ? tile_count - offset
                                  : PAGED_ATTENTION_COPY_TILE);
        if (offset == 0) {
          LocalTensor<Tdata> k0 = _k_queue.AllocTensor<Tdata>();
          DataCopy(
              k0, _k_cache_gm[k_base],
              DataCopyParams(static_cast<uint16_t>(count), copy_len, k_gap, 0));
          _k_queue.EnQue(k0);
          LocalTensor<Tdata> v0 = _v_queue.AllocTensor<Tdata>();
          DataCopy(
              v0, _v_cache_gm[v_base],
              DataCopyParams(static_cast<uint16_t>(count), copy_len, v_gap, 0));
          _v_queue.EnQue(v0);
        }
        LocalTensor<Tdata> k_tile = _k_queue.DeQue<Tdata>();
        LocalTensor<Tdata> v_tile = _v_queue.DeQue<Tdata>();
        const size_t next = offset + count;
        if (next < tile_count) {
          const size_t nc = (tile_count - next < PAGED_ATTENTION_COPY_TILE
                                 ? tile_count - next
                                 : PAGED_ATTENTION_COPY_TILE);
          LocalTensor<Tdata> kn = _k_queue.AllocTensor<Tdata>();
          DataCopy(
              kn, _k_cache_gm[k_base + next * _k_row_stride],
              DataCopyParams(static_cast<uint16_t>(nc), copy_len, k_gap, 0));
          _k_queue.EnQue(kn);
          LocalTensor<Tdata> vn = _v_queue.AllocTensor<Tdata>();
          DataCopy(
              vn, _v_cache_gm[v_base + next * _v_row_stride],
              DataCopyParams(static_cast<uint16_t>(nc), copy_len, v_gap, 0));
          _v_queue.EnQue(vn);
        }
        for (size_t i = 0; i < count; ++i) {
          LocalTensor<Tdata> k_raw = k_tile[i * _head_stride];
          LocalTensor<Tdata> v_raw = v_tile[i * _head_stride];
          const size_t t = token_start + offset + i;
          const float score =
              computeScore(q_local, k_raw, t, alibi_slope, seq_len);
          if (score > max_score) {
            score_local.SetValue(0, max_score - score);
            Exp(score_local, score_local, 1);
            const float rescale = score_local.GetValue(0);
            Muls(acc_local, acc_local, rescale, _head_size);
            sum_exp *= rescale;
            max_score = score;
          }
          score_local.SetValue(0, score - max_score);
          Exp(score_local, score_local, 1);
          const float prob = score_local.GetValue(0);
          sum_exp += prob;
          LocalTensor<float> v_f32 = _v_f32_buf.Get<float>();
          Cast(v_f32, v_raw, AscendC::RoundMode::CAST_NONE, _head_size);
          Muls(v_f32, v_f32, prob, _head_size);
          Add(acc_local, acc_local, v_f32, _head_size);
        }
        _k_queue.FreeTensor(k_tile);
        _v_queue.FreeTensor(v_tile);
        offset = next;
      }
    }
    PipeBarrier<PIPE_V>();
    // Normalize accumulator and write back output
    const float inv_sum = 1.0f / (sum_exp + 1e-6f);
    LocalTensor<Tdata> out_local = _out_buf.Get<Tdata>();

    Muls(acc_local, acc_local, inv_sum, _head_size);
    PipeBarrier<PIPE_V>();

    Cast(out_local, acc_local, AscendC::RoundMode::CAST_RINT,
         static_cast<int32_t>(_head_size));
    PipeBarrier<PIPE_V>();

    const ptrdiff_t out_base = static_cast<ptrdiff_t>(seq_idx) * _o_stride +
                               static_cast<ptrdiff_t>(head_idx * _head_size);

    DataCopy(_out_gm[out_base], out_local, _head_size);
  }

 private:
  __aicore__ inline float computeScore(LocalTensor<float>& q_local,
                                       LocalTensor<Tdata>& k_raw_local,
                                       size_t token_idx, float alibi_slope,
                                       size_t seq_len) {
    LocalTensor<float> k_f32_local = _v_f32_buf.Get<float>();

    Cast(k_f32_local, k_raw_local, AscendC::RoundMode::CAST_NONE, _head_size);
    Mul(k_f32_local, q_local, k_f32_local, _head_size);
    LocalTensor<float> temp_workspace = _reduce_buf.Get<float>();
    ReduceSum(k_f32_local, k_f32_local, temp_workspace, _head_size);
    PipeBarrier<PIPE_V>();

    float score = k_f32_local.GetValue(0);

    score *= _scale;
    if (_has_alibi) {
      score +=
          alibi_slope * static_cast<float>(static_cast<int64_t>(token_idx) -
                                           static_cast<int64_t>(seq_len) + 1);
    }
    return score;
  }

  GlobalTensor<Tdata> _out_gm;
  GlobalTensor<Tdata> _q_gm;
  GlobalTensor<Tdata> _k_cache_gm;
  GlobalTensor<Tdata> _v_cache_gm;
  GlobalTensor<Tindex> _block_tables_gm;
  GlobalTensor<Tindex> _cache_lens_gm;
  GlobalTensor<float> _alibi_slopes_gm;

  TPipe _pipe;
  TBuf<TPosition::VECCALC> _q_buf;
  TBuf<TPosition::VECCALC> _acc_buf;
  TBuf<TPosition::VECCALC> _score_buf;
  TBuf<TPosition::VECCALC> _out_buf;
  // Shared float scratchpad for K/V type-casting (used by computeScore and V
  // accumulation)
  TBuf<TPosition::VECCALC> _v_f32_buf;
  TBuf<TPosition::VECCALC> _reduce_buf;
  // Double-buffered VECIN queue for K prefetch
  TQue<QuePosition::VECIN, 2> _k_queue;
  // Double-buffered VECIN queue for V prefetch (overlaps with Score compute)
  TQue<QuePosition::VECIN, 2> _v_queue;

  size_t _num_heads;
  size_t _num_seqs;
  size_t _num_kv_heads;
  size_t _head_size;
  size_t _head_stride;
  float _scale;
  size_t _max_num_blocks_per_seq;
  size_t _page_block_size;
  ptrdiff_t _q_stride;
  ptrdiff_t _k_batch_stride;
  ptrdiff_t _k_row_stride;
  ptrdiff_t _k_head_stride;
  ptrdiff_t _v_batch_stride;
  ptrdiff_t _v_row_stride;
  ptrdiff_t _v_head_stride;
  ptrdiff_t _o_stride;
  ptrdiff_t _block_table_batch_stride;
  ptrdiff_t _cache_lens_stride;
  bool _has_alibi;
};

#define DEFINE_PAGED_ATTENTION_KERNEL(KERNEL_NAME, DATA_TYPE, INDEX_TYPE)      \
  extern "C" __global__ __aicore__ void KERNEL_NAME(                           \
      GM_ADDR out, GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache,                \
      GM_ADDR block_tables, GM_ADDR cache_lens, GM_ADDR alibi_slopes,          \
      bool has_alibi, size_t num_heads, size_t num_seqs, size_t num_kv_heads,  \
      size_t head_size, float scale, size_t max_num_blocks_per_seq,            \
      size_t page_block_size, ptrdiff_t q_stride, ptrdiff_t k_batch_stride,    \
      ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,                         \
      ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,                        \
      ptrdiff_t v_head_stride, ptrdiff_t o_stride,                             \
      ptrdiff_t block_table_batch_stride, ptrdiff_t cache_lens_stride) {       \
    PagedAttentionKernel<DATA_TYPE, INDEX_TYPE> op;                            \
    op.init(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,  \
            has_alibi, num_heads, num_seqs, num_kv_heads, head_size, scale,    \
            max_num_blocks_per_seq, page_block_size, q_stride, k_batch_stride, \
            k_row_stride, k_head_stride, v_batch_stride, v_row_stride,         \
            v_head_stride, o_stride, block_table_batch_stride,                 \
            cache_lens_stride);                                                \
    op.process();                                                              \
  }

DEFINE_PAGED_ATTENTION_KERNEL(infinilm_paged_attention_f16_i64, half, int64_t)
DEFINE_PAGED_ATTENTION_KERNEL(infinilm_paged_attention_f16_i32, half, int32_t)
DEFINE_PAGED_ATTENTION_KERNEL(infinilm_paged_attention_f16_u32, half, uint32_t)
DEFINE_PAGED_ATTENTION_KERNEL(infinilm_paged_attention_bf16_i64, bfloat16_t,
                              int64_t)
DEFINE_PAGED_ATTENTION_KERNEL(infinilm_paged_attention_bf16_i32, bfloat16_t,
                              int32_t)
DEFINE_PAGED_ATTENTION_KERNEL(infinilm_paged_attention_bf16_u32, bfloat16_t,
                              uint32_t)

#undef DEFINE_PAGED_ATTENTION_KERNEL
